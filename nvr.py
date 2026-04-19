import os
import glob
import threading
import time
import subprocess
import threading
import queue
from datetime import datetime
from logging import getLogger
import cv2
import torch
import numpy as np

from ffmpeg import FFmpeg

import constants
from context import Context
from logger import log_event
from camera import Camera
from model import Model

logger = getLogger("nvr")

def _is_night_time(frame, brightness_threshold=50):
    # Convert to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate average brightness (V channel)
    mean_brightness = np.mean(hsv[:,:,2])
    
    # If brightness is low, it's likely night time
    return mean_brightness < brightness_threshold

def _keep_overlapping_any(boxes, ref_boxes):
    """
    boxes: YOLO r.boxes.xyxy  -> (N, 4)
    ref_boxes: cv2.boundingRect -> (M, 4) in (x, y, x1, y1)
    """

    # 1. Ensure correct shapes
    boxes = boxes.view(-1, 4)
    ref_boxes = ref_boxes.view(-1, 4)

    # 3. Compute pairwise intersection
    x1 = torch.maximum(boxes[:, None, 0], ref_boxes[None, :, 0])
    y1 = torch.maximum(boxes[:, None, 1], ref_boxes[None, :, 1])
    x2 = torch.minimum(boxes[:, None, 2], ref_boxes[None, :, 2])
    y2 = torch.minimum(boxes[:, None, 3], ref_boxes[None, :, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)

    overlap = (inter_w * inter_h) > 0  # (N, M)

    # 4. Keep if overlaps ANY ROI
    return overlap.any(dim=1)

# =========================
# NVR ENGINE
# =========================
class NVR:
    def __init__(self, ctx: Context, model: Model):
        self.ctx = ctx
        self.model = model
        self.debug = self.ctx.debug
        self.width = ctx.downsize_resolution[0]
        self.height = ctx.downsize_resolution[1]
        self.motion_threshold = self.ctx.motion_threshold
        self.confidence_threshold = self.ctx.confidence_threshold
        self.selected_classes = self.model.class_to_index(ctx.classes)

        self.recordings_dir = ctx.directory
        self.segments_dir = os.path.join(self.recordings_dir, "segments")
        self.images_dir = os.path.join(self.recordings_dir, "images")
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        self.cameras = {}
        for name, cfg in ctx.camera_config.items():
            self.cameras[name] = Camera(name=name,
                                        url=cfg['url'],
                                        enabled=cfg['enabled'],
                                        recordings_dir=os.path.join(self.recordings_dir, name),
                                        segments_dir=os.path.join(self.segments_dir, name),
                                        images_dir=os.path.join(self.images_dir, name),
                                        )

    def start(self):
        for camera in self.cameras.values():
            if camera.enabled:
                os.makedirs(camera.recordings_dir, exist_ok=True)
                os.makedirs(camera.segments_dir, exist_ok=True)
                os.makedirs(camera.images_dir, exist_ok=True)
                log_event(message=f"starting recorder", level="info", camera=camera)
                camera.process = self._start_segment_recorder(camera=camera)
                threading.Thread(target=self._frame_reader, args=(camera,), daemon=True).start()
                threading.Thread(target=self._process_frames,args=(camera,), daemon=True).start()
        threading.Thread(target=self._cleanup_segments,daemon=True).start()

    def restart(self, camera):
        if camera.enabled and camera.process is not None:
            ret = camera.process.poll()
            log_event(message=f"restarting recorder with ret {ret}", level="info", camera=camera)
            camera.process.terminate()

            try:
                camera.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                camera.process.kill()
            camera.process.stdout.close()
            camera.process.stdin.close()
            camera.process = self._start_segment_recorder(camera=camera)

    def _start_segment_recorder(self, camera: Camera):

        filespec = os.path.join(camera.segments_dir, "%Y%m%d_%H%M%S.ts")
        log_file = open(f"{camera.name}_ffmpeg.log", "w")
        ffmpeg_cmd = [
            "ffmpeg",

            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-use_wallclock_as_timestamps", "1",
            "-i", camera.url,

            # Split and reduce scale for raw only for OpenCV
            "-filter_complex",
            f"[0:v]scale={self.width}:{self.height},format=bgr24[raw]",

            # ---- TS segments (NO RE-ENCODE) ----
            "-map", "0:v",
            "-c", "copy",
            "-f", "segment",
            "-segment_time", "1",
            "-reset_timestamps", "1",
            "-strftime", "1",
            "-segment_format", "mpegts",
            filespec,

            # ---- Raw frames (OpenCV) ----
            "-map", "[raw]",
            "-f", "rawvideo",
            "pipe:1"
        ]

        process =  subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_file,
            bufsize=10**8
        )

        return process

    def _cleanup_segments(self):
        threading.current_thread().name = "cleanup_segments"
            
        while True:
            try:
                for camera in self.cameras.values():
                    if camera.enabled:
                        path = os.path.join(camera.segments_dir, "*.ts")
                        files = sorted(glob.glob(path))
                        if len(files) > constants.BUFFER_SECONDS:
                            for f in files[:-constants.BUFFER_SECONDS]:
                                try: os.remove(f)
                                except: pass
                time.sleep(1)
            except Exception as e:
                log_event(message=f"exception in cleanup_segments {e}", level="error")

    def _get_segments(self, camera: Camera, n: int):
        files = sorted(glob.glob(os.path.join(camera.segments_dir, "*.ts")))
        return files[-n:]

    def _merge_segments(self, files, output):
        list_file = output + ".txt"
        with open(list_file,"w") as f:
            for x in files:
                f.write(f"file '{os.path.abspath(x)}'\n")

        (
            FFmpeg()
            .option("y")
            .input(list_file, f="concat", safe=0)
            .output(
                output,
                c="libx264",
                pix_fmt="yuv420p",
                movflags="+faststart",
                preset="veryfast",
                crf=23
            )
            .execute()
        )

        os.remove(list_file)

    def _frame_reader(self, camera: Camera):
        threading.current_thread().name = f"{camera.name} _frame_reader"

        frame_size = self.width * self.height * 3

        while camera.running:
            raw = self._read_exact(camera.process.stdout, frame_size)

            if raw is None:
                log_event(message="reader failed, restarting stream", level="warn", camera=camera)
                self.restart(camera)
                continue

            frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))

            # FPS calculation
            now = time.perf_counter()
            if camera.last_frame_time > 0:
                dt = now - camera.last_frame_time

                # filter pipeline artifacts (VERY IMPORTANT)
                if 0.01 < dt < 1.0:
                    inst_fps = 1.0 / dt
                    camera.fps.update(inst_fps)

            camera.last_frame_time = now

            # latest-frame-wins
            if camera.frame_queue.full():
                camera.frame_queue.get_nowait()
            camera.frame_queue.put(frame)

    def _read_exact(self, pipe, size):
        buf = b""
        while len(buf) < size:
            chunk = pipe.read(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _process_frames(self, camera: Camera):
        threading.current_thread().name = f"{camera.name} _process_frames"

        first_frame = True
        prev_gray = None
        motion_frames = 0
        no_motion_frames = 0
        recording = False
        prev_time = time.time()
        is_night = 0

        while camera.running:
            # get latest frame (non-blocking)
            try:
                frame = camera.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if first_frame:
                log_event(message=f"first frame read from stream", level="info", camera=camera)
                first_frame = False

            now = time.time()

            if now - camera.last_night_time_check > constants.PERIODIC_CHECK_INTERVAL:
                is_night = 1 if _is_night_time(frame, constants.NIGHT_TIME_THRESHOLD) else 0
                camera.last_night_time_check = time.time()
                if now - prev_time > 10.0:
                    log_event("stopped reading frames", level="info", camera=camera)

            # motion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=camera.gray_buf)
            gray = cv2.GaussianBlur(gray, (21,21), 0, dst=camera.gray_buf)
            camera.motion_boxes_list.clear()
            overlay = None

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray, dst=camera.diff_buf)
                _, thresh = cv2.threshold(diff,25,255,cv2.THRESH_BINARY, dst=camera.thresh_buf)
                score = cv2.countNonZero(thresh)

                if score > self.motion_threshold[is_night]:
                    # Motion is detected, get contours of moving objects
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        x1, y1, w, h = cv2.boundingRect(contour)
                        x2 = x1 + w
                        y2 = y1 + h
                        if area < self.motion_threshold[is_night] / 10:  # filter small noise
                            # Ignore small motion areas
                            if self.debug:
                                if overlay is None:
                                    overlay = frame.copy()
                                cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 1)
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
                                #log_event(message=f"ignoring motion contour rect ({x1}, {y1}), ({x2}, {y2}) with area {area}")
                        else:
                            camera.motion_boxes_list.append([x1, y1, x2, y2])
                            if self.debug:
                                if overlay is None:
                                    overlay = frame.copy()
                                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 1)
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                                #log_event(message=f"motion contour rect ({x1}, {y1}), ({x2}, {y2}) with area {area}")
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                tag = "_".join( camera.active_objects_set) if  camera.active_objects_set else "motion"
                                image_filename = os.path.join(camera.images_dir, f"{timestamp}_{tag}.jpg")
                                cv2.imwrite(image_filename, overlay)
                                log_event(message=f"contour image written to {image_filename}", level="debug", camera=camera, file_path=image_filename)

            prev_gray = gray

            # YOLO
            result = None
            camera.classes_in_frame_set.clear()

            # if there is large enough motion boxes, run YOLO and see if objects we care about overlap
            # with the motion boxes (either the object is moving, or something is moving across the object)
            if camera.motion_boxes_list and (time.time() - camera.last_yolo_time > 0.2):
                camera.last_yolo_time = time.time()
                result = self.model.model.predict(frame, conf=self.confidence_threshold, classes=self.selected_classes if self.selected_classes else None, verbose=False)[0]
                boxes = result.boxes.xyxy.reshape(-1, 4)
                ref_motion_boxes_list = torch.as_tensor(camera.motion_boxes_list, dtype=boxes.dtype, device=boxes.device)
                keep = _keep_overlapping_any(boxes, ref_motion_boxes_list)
                boxes = result.boxes[keep]

                # store the name/class of object we saw in the frame that coincides with movement
                for box in boxes:
                    camera.classes_in_frame_set.add(self.model.model.names[int(box.cls)])

            # counters, we need motion (or no motion) for a number of consecutive frames to care
            if camera.motion_boxes_list:
                motion_frames += 1
                no_motion_frames = 0
            else:
                no_motion_frames += 1

            # start
            if motion_frames >= constants.MOTION_DETECT_FRAME_COUNT and not recording:
                valid_objects = len(camera.classes_in_frame_set) > 0
                if (not constants.REQUIRE_OBJECT_FOR_RECORDING or valid_objects):
                    if now - camera.last_event_time > constants.EVENT_COOLDOWN:
                        recording = True
                        camera.active_segments_list = self._get_segments(camera, constants.PRE_RECORD_SEGMENTS)
                        camera.active_objects_set = set(camera.classes_in_frame_set)
                        camera.last_event_time = now
                        log_event(message=f"recording start {", ".join(camera.active_objects_set) if camera.active_objects_set else ""}", level="info", camera=camera)

            # update active segments and objects
            if recording:
                camera.active_segments_list += self._get_segments(camera,1)
                camera.active_objects_set.update(camera.classes_in_frame_set)

            # stop recoding when there has been no motion for some time
            if recording and no_motion_frames >= constants.NO_MOTION_DETECT_FRAME_COUNT:
                recording = False
                segments = list(dict.fromkeys(camera.active_segments_list))

                # if we have segments, merge them into an mp4 file with timestamp and tags
                if segments:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tag = "_".join(camera.active_objects_set) if camera.active_objects_set else "motion"

                    recording_filename = os.path.join(camera.recordings_dir, f"{timestamp}_{tag}.mp4")
                    self._merge_segments(segments, recording_filename)

                    recorded_cap = cv2.VideoCapture(recording_filename)
                    frame_count = recorded_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if frame_count < constants.NO_MOTION_DETECT_FRAME_COUNT + 20 and os.path.isfile(recording_filename):
                        os.remove(recording_filename)
                        log_event(message=f"recording auto-deleted {os.path.basename(recording_filename)} with {frame_count} frames", level="info", camera=camera, file_path=recording_filename)
                    else:
                        log_event(message=f"recording available {os.path.basename(recording_filename)}", level="record", camera=camera, file_path=recording_filename)

                camera.active_segments_list.clear()
                camera.classes_in_frame_set.clear()
                camera.active_objects_set.clear()
                motion_frames = 0
                no_motion_frames = 0

            # render YOLO plots on the frame if there was a result
            img = result.plot() if result else frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if overlay is not None:
                img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

            if not self.cameras[camera.name].hd:
                img = cv2.resize(img, constants.RENDER_SIZE)

            prev_time = time.time()

            status = self.make_status(recording)
            
            camera.latest_frame = img
            camera.status = f"{status} {"| Night " if is_night else ""}| FPS {int(camera.fps.value())}" + (f" | {",".join(camera.active_objects_set)}" if len(camera.active_objects_set) > 0 else "")
            
    def make_status(self, recording: bool):
        idx = int(time.time() * 4) % 4

        red_cycle = ["🔴", "🔴", "🟠", "🟠"]
        green_cycle = ["🟢", "🟢", "⚪", "⚪"]

        pulse = red_cycle[idx] if recording else green_cycle[idx]

        return f"{pulse}{' REC' if recording else ' LIVE'}"