"""
 API endpoints for the pynvr application, including authentication,
 camera management, and event streaming.
 
 This module defines the FastAPI application and its routes,
 handling user authentication, camera settings, event logs, and server time.
 It also provides a Server-Sent Events (SSE) endpoint for real-time updates.
"""
import asyncio
import bisect
import logging
import secrets
import time
from collections import deque
from datetime import datetime
from logging import getLogger

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request, Depends, HTTPException, Query, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, EventSourceResponse
from passlib.context import CryptContext

from pynvr.config.config import SystemConfig
from pynvr.logger import event_log
from pynvr.nvr import NVR
from pynvr.webrtc import CameraTrack, MosaicTrack
from pynvr.api.types import (
    CameraDebugResponse,
    CameraResponse,
    CameraSettingResponse,
    CameraSettingsResponse,
    CameraStatus,
    ClassesResponse,
    ClassToggle,
    ClassToggleResponse,
    DimensionsResponse,
    EventsResponse,
    LogsResponse,
    LoginForm,
    LogEntry,
    RecordingEvent,
    ServerTimeResponse,
    SettingValue,
    SettingValueResponse,
    SSEEvent,
    SystemNameResponse,
)
logger = getLogger("pynvr")

# -------------------------
# AUTH SETUP
# -------------------------

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

SESSIONS = {}  # session_id -> username

class AuthStaticFiles(StaticFiles):
    """
    Custom StaticFiles class that checks for a valid session cookie before serving files.
    """
    async def __call__(self, scope, receive, send):
        path = scope["path"]

        # Always allow login page and its assets
        if path.startswith("/login"):
            return await super().__call__(scope, receive, send)

        # Allow SvelteKit build assets
        if path.startswith("/_app"):
            return await super().__call__(scope, receive, send)

        # Allow mosaic.js and related scripts
        if path.startswith("/mosaic") or path.endswith(".js"):
            return await super().__call__(scope, receive, send)

        # Allow static assets
        if path.startswith("/static") or path.startswith("/assets"):
            return await super().__call__(scope, receive, send)

        # Check session cookie
        cookie_header = None
        for k, v in scope["headers"]:
            if k == b"cookie":
                cookie_header = v.decode()
                break

        session_id = None
        if cookie_header:
            for part in cookie_header.split(";"):
                if "session_id=" in part:
                    session_id = part.split("=")[1].strip()

        if session_id not in SESSIONS:
            response = Response(status_code=302, headers={"Location": "/login"})
            await response(scope, receive, send)
            return

        return await super().__call__(scope, receive, send)

def require_user(session_id: str | None = Cookie(None)):
    """Require a valid session cookie for all protected routes."""
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return SESSIONS[session_id]


# -------------------------
# APP FACTORY
# -------------------------

#pylint: disable=too-many-statements
def create_app(system_config: SystemConfig, nvr: NVR):
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------
    # AUTH ROUTES
    # -------------------------
    @app.get("/login")
    async def login_page():
        return FileResponse("pynvr/frontend_dist/index.html")

    @app.post("/login")
    async def login(payload: LoginForm, response: Response):
        if (payload.username != system_config.gui_username
            or not pwd.verify(
                payload.password,
                system_config.gui_password)
                ):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        session_id = secrets.token_hex(32)
        SESSIONS[session_id] = payload.username

        response.set_cookie(
            "session_id",
            session_id,
            httponly=True,
            secure=False,
            samesite="lax",
            max_age=86400,
        )
        return {"ok": True}

    @app.post("/logout")
    async def logout(response: Response, session_id: str | None = Cookie(None)):
        if session_id in SESSIONS:
            del SESSIONS[session_id]
        response.delete_cookie("session_id")
        return {"ok": True}

    @app.get("/whoami")
    async def whoami(user=Depends(require_user)):
        return {"user": user}

    # -------------------------
    # PROTECTED ROUTES
    # -------------------------

    #pylint: disable=unused-argument
    @app.post("/signal")
    async def signal(request: Request, user=Depends(require_user)):
        data = await request.json()
        mode = data.get("mode", "single")
        name = data.get("name")
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        pc = RTCPeerConnection()

        if mode == "mosaic":
            cameras = [camera for camera in nvr.cameras.values() if camera.camera_info.enabled]
            track = MosaicTrack(cameras, system_config.mosaic)
        else:
            camera = nvr.cameras[name]
            track = CameraTrack(camera)

        pc.addTrack(track)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    @app.get("/api/cameras", response_model=list[CameraResponse])
    def get_cameras(user=Depends(require_user)):
        return [CameraResponse(name=camera.camera_info.name, debug=camera.camera_info.debug)
                for camera in nvr.cameras.values() if camera.camera_info.enabled]

    @app.get("/api/classes", response_model=ClassesResponse)
    def get_classes(user=Depends(require_user)):
        return ClassesResponse(classes=system_config.model.classes)

    @app.get("/api/system_name", response_model=SystemNameResponse)
    def get_system_name():
        return SystemNameResponse(system_name=system_config.system_name)

    @app.get("/api/mosaic_dimensions", response_model=DimensionsResponse)
    def get_mosaic_dimensions(user=Depends(require_user)):
        return DimensionsResponse(
            rows=system_config.mosaic.rows,
            columns=system_config.mosaic.columns,
            width=system_config.mosaic.resolution.width,
            height=system_config.mosaic.resolution.height
        )

    @app.get("/api/events", response_model=EventsResponse)
    def api_events(
        start: float | None = Query(None),
        end: float | None = Query(None),
        mobile: bool = False,
        user = Depends(require_user)
    ):
        """import os

        Returns events for the requested time window.

        Modes:
        • mobile=true → last 4 hours (unchanged)
        • start/end provided → return only events overlapping [start, end]
        • no params → return all events (fallback)
        """

        # nvr.recordings MUST be sorted by start_time ascending
        events = list(nvr.recordings)

        # ------------------------------------------------------------
        # MOBILE MODE: last 4 hours (your existing behavior)
        # ------------------------------------------------------------
        if mobile:
            cutoff = datetime.now().timestamp() - 4 * 3600
            end_times = [ev.end_time for ev in events]
            idx = bisect.bisect_left(end_times, cutoff)
            return EventsResponse(events=events[idx:])

        # ------------------------------------------------------------
        # DESKTOP MODE: window-based filtering
        # ------------------------------------------------------------
        if start is not None and end is not None:
            # 1. Find first event whose end_time >= start
            end_times = [ev.end_time for ev in events]
            left = bisect.bisect_left(end_times, start)

            # 2. Find last event whose start_time <= end
            start_times = [ev.start_time for ev in events]
            right = bisect.bisect_right(start_times, end)

            # 3. Slice and return only overlapping events
            window_events = events[left:right]

            # 4. Final overlap check (safety)
            window_events = [
                ev for ev in window_events
                if ev.end_time >= start and ev.start_time <= end
            ]

            return EventsResponse(events=window_events)

        # ------------------------------------------------------------
        # FALLBACK: return everything
        # ------------------------------------------------------------
        return EventsResponse(events=events)


    @app.get("/api/logs", response_model=LogsResponse)
    def get_logs(
        since: float | None = None,
        user=Depends(require_user)
        ):

        log_entries: deque[LogEntry] = event_log

        if since is not None:
            timestamps = [log_entry.timestamp for log_entry in event_log]
            index = bisect.bisect_right(timestamps, since)
            log_entries = event_log[index:]

        return LogsResponse(log_entries=log_entries)

    @app.get("/api/cameras/{camera_name}/settings", response_model=CameraSettingsResponse)
    def get_camera_settings(camera_name: str):
        camera = nvr.cameras[camera_name]
        processor = nvr.processors[camera_name]

        return CameraSettingsResponse(
            yolo_confidence=vars(camera.camera_info.yolo_confidence),
            track_threshold=vars(camera.motion.track_threshold),
            match_threshold=vars(camera.motion.match_threshold),
            track_buffer=vars(camera.motion.track_buffer),
            minimum_relative_motion=vars(camera.motion.minimum_relative_motion),
            classes=processor.classes
        )

    @app.post("/api/cameras/{camera_name}/settings/{setting}", response_model=CameraSettingResponse)
    def update_camera_setting(camera_name: str, setting: str, payload: SettingValue):

        camera = nvr.cameras[camera_name]
        old_value = None

        if setting == "yolo_confidence":
            attr = getattr(camera.camera_info, setting)
            old_value = attr.value
            attr.value = payload.value
        else:
            attr = getattr(camera.motion, setting)
            old_value = attr.value
            attr.value = payload.value
            camera.motion.create_tracker()  # Ensure BYTETracker is recreated with new setting

        logger.info(f"{camera_name} {setting} {old_value:.2f} -> {payload.value:.2f}")

        return CameraSettingResponse(
            status="ok",
            camera=camera_name,
            setting=setting,
            value=payload.value
        )


    @app.post("/api/processor/{camera_name}/class_toggle", response_model=ClassToggleResponse)
    def update_class_toggle(camera_name: str, payload: ClassToggle):

        camera = nvr.cameras[camera_name]

        # Update the boolean toggle
        for processor in nvr.processors.values():
            if processor.camera.camera_info.name == camera_name:
                processor.classes[payload.class_name] = payload.value
                processor.set_selected_classes(processor.classes)
                break

        # Recreate tracker if class filters affect tracking
        camera.motion.create_tracker()

        logger.info(
            camera_name + " class " + payload.class_name + ": " + str(payload.value)
        )

        return ClassToggleResponse(
            status="ok",
            camera=camera_name,
            class_name=payload.class_name,
            value=payload.value
        )

    # camera debug
    @app.get("/api/settings/debug/{camera_name}", response_model=CameraResponse)
    def get_camera_debug(camera_name: str):
        camera = nvr.cameras.get(camera_name)

        return CameraResponse(
            name=camera_name,
            debug=camera.camera_info.debug if camera else False
        )

    @app.post("/api/settings/debug/{camera_name}", response_model=CameraDebugResponse)
    def set_camera_debug(camera_name: str, payload: SettingValue, user=Depends(require_user)):
        camera = nvr.cameras.get(camera_name)
        logger.info(camera_name + " debug " + str(payload.value))
        camera.camera_info.debug = payload.value
        return CameraDebugResponse(
            status="ok",
            camera=camera_name,
            debug=payload.value
            )

    # Verbose debug
    @app.get("/api/settings/debug", response_model=SettingValue)
    def get_debug(user=Depends(require_user)):
        return SettingValue(value=system_config.debug)

    @app.post("/api/settings/debug", response_model=SettingValueResponse)
    def set_debug(payload: SettingValue, user=Depends(require_user)):
        logger.info(f"verbose logging {payload.value}")
        system_config.debug = payload.value

        pynvr_logger = getLogger("pynvr")
        pynvr_logger.setLevel(logging.DEBUG if system_config.debug else logging.INFO)

        for handler in pynvr_logger.handlers:
            handler.setLevel(logging.DEBUG if system_config.debug else logging.INFO)

        logger.debug(f"verbose logging {payload.value} applied to all handlers")

        return SettingValueResponse(
            status="ok",
            value=payload.value
        )

    @app.get("/api/server_time", response_model=ServerTimeResponse)
    def server_time(user=Depends(require_user)):
        return ServerTimeResponse(epoch=time.time())

    async def event_generator(request: Request):
        last_log_time = time.time()
        last_recording_time = time.time()

        try:
            while True:
                if nvr.stop_event.is_set():
                    break

                if await request.is_disconnected():
                    logger.debug("SSE request is disconnected")
                    break

                # CAMERA STATUS
                for processor in nvr.processors.values():
                    camera_status = CameraStatus(
                        ts=time.time(),
                        name=processor.camera.camera_info.name,
                        state=processor.streaming_state.name,
                        state_value=processor.streaming_state.value,
                        objects_dict=processor.camera.motion.get_active_objects(),
                        night=processor.camera.is_night,
                        recording=processor.camera.recording_state.recording,
                        read_fps=processor.reader.fps.as_int(),
                        record_fps=processor.recorder.fps.as_int(),
                    )

                    yield SSEEvent(
                        type="cameraStatus",
                        data=camera_status).to_sse()

                # LOGS
                logs_list = list(event_log)

                if last_log_time is not None:
                    timestamps = [log.timestamp for log in logs_list]
                    index = bisect.bisect_right(timestamps, last_log_time)
                    logs_list = logs_list[index:]

                for log_line in logs_list:
                    yield SSEEvent(
                        type="logLine",
                        data=log_line).to_sse()

                last_log_time = time.time()

                # RECORDINGS
                recordings = list(nvr.recordings)
                if last_recording_time is not None:
                    start_times = [recording.start_time for recording in recordings]
                    index = bisect.bisect_right(start_times, last_recording_time)
                    recordings = recordings[index:]

                for recording_event in recordings:
                    yield SSEEvent(
                        type="newEvent",
                        data=recording_event).to_sse()

                if recordings:
                    last_recording_time = recordings[-1].start_time

                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info("Client disconnected from SSE stream.")
            raise

    @app.get("/api/stream", response_model=CameraStatus | LogEntry | RecordingEvent)
    async def stream(request: Request, user=Depends(require_user)):
        return EventSourceResponse(
            event_generator(request)
        )

    if system_config:
        app.mount(
            "/recordings",
            AuthStaticFiles(
                directory=system_config.recordings_directory,
                check_dir=True),
            name="recordings")
        app.mount(
            "/logs",
            AuthStaticFiles(
                directory=system_config.logs_directory,
                check_dir=True),
            name="logs")
        app.mount(
            "/",
            AuthStaticFiles(
                directory="pynvr/frontend_dist",
                html=True),
            name="frontend")

    @app.get("/{path:path}")
    async def spa_fallback(path: str, user=Depends(require_user)):
        return FileResponse("pynvr/frontend_dist/index.html")

    return app
