from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
import glob
import json
from logging import getLogger
import os
import queue
import subprocess
from threading import Thread, Event, current_thread
import time
from math import sqrt

import cv2
from gradio.monitoring_dashboard import data
from tomlkit import value
import torch
import numpy as np

from camera import Camera
import constants
from context import Context
from logger import log_event
from model import Model
from motion_profile import MotionProfile

logger = getLogger("nvr")

# =========================
# NVR ENGINE
# =========================
class NVR:
    def __init__(self, ctx: Context):
        self._ctx = ctx
        self._width: int = ctx.resolution[0]
        self._height: int = ctx.resolution[1]
        self._max_pixels = self._width * self._height
        
        self.model = Model(ctx)
        self.stop_event: Event = Event()
        self.debug: bool = self._ctx.debug
        self.debug_files: bool = self._ctx.debug_files
        self.motion_threshold: float = self._ctx.motion_threshold
        self.confidence_threshold: float = self._ctx.confidence_threshold
        self.selected_classes: list[int] = self.model.class_to_index(ctx.classes)

        self._recordings_dir: str = ctx.directory
        self._segments_dir: str = os.path.join(self._recordings_dir, "segments")
        self._images_dir: str = os.path.join(self._recordings_dir, "images")
        os.makedirs(self._recordings_dir, exist_ok=True)
        os.makedirs(self._segments_dir, exist_ok=True)
        os.makedirs(self._images_dir, exist_ok=True)
        self.cameras: dict[str, Camera] = {}
        for name, cfg in ctx.camera_config.items():
            self.cameras[name] = Camera(name=name,
                                        url=cfg['url'],
                                        enabled=cfg['enabled'],
                                        recordings_dir=os.path.join(self._recordings_dir, name),
                                        segments_dir=os.path.join(self._segments_dir, name),
                                        images_dir=os.path.join(self._images_dir, name),
                                        model=Model(ctx)
                                        )

    def start(self):
        """
        Start the NVR processes. Threads created are:
        1 ffmpeg reader thread for each camera, writing to segment files and stdout
        1 ffmpeg frame reader thread for each camera reading from stdout and writing frames to a queue
        1 frame processor thread to read frames from the queue and do image processing
        """
        if not self.stop_event.is_set():
            for camera in self.cameras.values():
                if camera.enabled:
                    os.makedirs(camera.recordings_dir, exist_ok=True)
                    os.makedirs(camera.segments_dir, exist_ok=True)
                    os.makedirs(camera.images_dir, exist_ok=True)
                    self._start_camera(camera=camera)
                    Thread(target=self._frame_reader, args=(camera,), daemon=True).start()
                    Thread(target=self._process_frames,args=(camera,), daemon=True).start()
            Thread(target=self._cleanup_segments,daemon=True).start()
            Thread(target=self._watch_cameras,daemon=True).start()

    def stop(self):
        """
        Stop the NVR
        """
        for camera in self.cameras.values():
            if camera.enabled and camera.process is not None:
                self._stop_camera(camera)

    def _restart_camera(self, camera):
        """
        Stop and start the camera unless we are shutting down
        """
        if not self.stop_event.is_set():
            log_event(message="restarting camera", level="warn", camera=camera)
            self._stop_camera(camera)
            self._start_camera(camera)

    def _stop_camera(self, camera):
        """
        Stops the background ffmpeg process for the camera, closes pipes and resets the camera
        """
        if camera.enabled and camera.process is not None:
            ret = camera.process.poll()
            log_event(message=f"stopping camera with ret {ret}", level="info", camera=camera)
            camera.process.terminate()

            try:
                camera.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                camera.process.kill()
            camera.process.stdout.close()
            camera.first_frame = True

    def _start_camera(self, camera: Camera):
        """
        Starts ffmpeg as a subprocess reading from the camera RTSP stream. The stream is split
        in two writing simultaneously to segment files and stdout. No re-encoding happens to the
        segment files. The frames written to stdout are resized for image processing by cv2. 
        """
        if not self.stop_event.is_set():
            log_event(message=f"starting recorder", level="info", camera=camera)
            filespec = os.path.join(camera.segments_dir, "%Y%m%d_%H%M%S.ts")
            ffmpeg_cmd = [
                "ffmpeg",

                "-rtsp_transport", "tcp",           # Forces RTSP over TCP instead of UDP
                "-fflags", "nobuffer+genpts",       # Disables internal buffering, generates PTS
                "-flags", "low_delay",              # Tells decoder/demuxer to minimize delay (Reduces frame reordering buffers)
                "-i", camera.url,                   # RTSP stream from camera
                "-hide_banner",
                "-loglevel", "error",               # ONLY errors (no frame spam)
                "-nostats",
                
                "-filter_complex",                  # Split and reduce scale for raw only for OpenCV
                f"[0:v]scale={self._width}:{self._height},format=bgr24[raw]", # re-scale and raw BGR pixel format (OpenCV native)

                # ---- TS segments (NO RE-ENCODE) ----
                "-map", "0:v",                      # original stream, unaltered
                "-c", "copy",                       # No re-encoding (copy stream)
                "-f", "segment",                    # enable segment muxer
                "-segment_time", "1",               # target segment length 1 second
                "-reset_timestamps", "0",           # don't reset timestamps
                "-strftime", "1",                   # enable timestamp based filenames
                "-segment_format", "mpegts",        # force mpeg-ts container
                filespec,

                # ---- Raw frames (OpenCV) ----
                "-map", "[raw]",                    # selects filtered (scaled + BGR) stream
                "-f", "rawvideo",                   # outputs raw uncompressed frames
                "pipe:1"                            # sends raw bytes to stdout
            ]
            process =  subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0
            )
            camera.process = process

            return process

    def _watch_cameras(self):
        """ check each ffmpeg process every 5 seconds and restart if necessary """
        while not self.stop_event.is_set():
            time.sleep(5)
            for camera in self.cameras.values():
                if camera.process and camera.process.poll() is not None:
                    log_event("ffmpeg died, restarting", "error", camera=camera)
                    self._restart_camera(camera)

    def _cleanup_segments(self):
        """
        Thread that periodically deletes old segment files for all cameras
        """
        current_thread().name = "cleanup_segments"
            
        while True and not self.stop_event.is_set():
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
        """
        get the list of segment file for this camera for the duration it was recording
        """
        files = sorted(glob.glob(os.path.join(camera.segments_dir, "*.ts")))
        return files[-n:]


    def _merge_segments_async(self, camera: Camera, segments: list[str], tags: defaultdict[set], output: str):
        """
        Runs ffmpeg merge in a separate thread. When the process finishes,
        the log the event and delete the listing file.
        """
        def timestamp_to_epoch(ts: str) -> int:
            """
            Convert 'YYYYMMDD_HHMMSS' → epoch seconds.

            Example:
                '20260428_143210' -> 1777386730
            """
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            return dt.timestamp()

        def worker():
            list_file = output + ".txt"
            with open(list_file,"w") as f:
                for x in segments:
                    try:
                        if os.stat(x).st_size > 0:
                            f.write(f"file '{os.path.abspath(x)}'\n")
                    except FileNotFoundError as e:
                        pass
            jsondata_file = output + ".json"
            tags_str = self._tags_to_str(tags)
            if self.debug:
                log_event(message=f"merging {len(segments)} segments {tags_str} to {output}", level="debug", camera=camera, file_path=output)
            
            # Convert to a standard dict and sets to lists
            serializable_tags = {k: list(v) for k, v in tags.items()}

            with open(jsondata_file, "w") as f:
                json_data = {
                    "camera": camera.name,
                    "segments": segments,
                    "tags": serializable_tags,
                    "output": output,
                    "start_time": timestamp_to_epoch(os.path.basename(segments[0]).split(".")[0]),
                    "end_time": timestamp_to_epoch(os.path.basename(segments[-1]).split(".")[0]),
                    "metadata": jsondata_file,
                }
                f.write(json.dumps(json_data, indent=4))
                
            flat_name = "".join(output.partition(camera.name)[1:]).replace(os.sep, '_')
            log_file = open(f"{os.path.splitext(flat_name)[0]}_merge.log", "w")
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-fflags", "+genpts",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-preset", "veryfast",
                "-crf", "23",
                "-vsync", "cfr",
                "-r", "20",
                "-video_track_timescale", "90000",
                output
            ]
            try:
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=log_file
                )

                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    # You can log or handle errors here if needed
                    pass

            finally:
                # This runs when the thread finishes (success or failure)
                log_file.close()
                os.remove(list_file)
                self._merge_complete(camera, tags, output)

        thread = Thread(target=worker, daemon=True)
        thread.start()
        return thread

    def _get_motion_threshold(self, is_night: bool):
        if is_night:
            return MotionProfile(
                name="night",
                motion_threshold=int(self.motion_threshold * 1.5),     # 50% higher
                min_solidity=0.70,                                     # stricter
                min_area_ratio=0.003,                                  # larger min area
                yolo_conf=min(0.6, self.confidence_threshold + 0.15),  # stricter YOLO
                motion_confidence_min=0.50,                            # require stronger motion
            )
        return MotionProfile(
        name="day",
        motion_threshold=self.motion_threshold,
        min_solidity=0.50,
        min_area_ratio=0.002,
        yolo_conf=self.confidence_threshold,
        motion_confidence_min=0.35,
    )

    def _tags_to_str(self, tags: defaultdict[set]):
        if not tags:
            return ""

        parts = []
        for obj, colors in tags.items():
            object_str = obj
            color_str = "/".join(colors)
            parts.append(f"{object_str}({color_str})")
        return ", ".join(parts)

    def _merge_complete(self, camera: Camera, tags: defaultdict[set], output: str):
        """
        logs the merge completion event and deletes recording if too short
        """
        cap = cv2.VideoCapture(output)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_seconds = frame_count / fps
        formatted_duration = str(timedelta(seconds=int(duration_seconds)))
        tags_str = self._tags_to_str(tags)
        if frame_count < constants.NO_MOTION_DETECT_FRAME_COUNT + 20 and os.path.isfile(output):
            os.remove(output)
            log_event(message=f"recording auto-deleted {os.path.basename(output)} with {frame_count} frames", level="info", camera=camera, file_path=output)
        else:
            log_event(message=f"recording available {formatted_duration} {tags_str} {os.path.basename(output)}", level="record", camera=camera, file_path=output)

    def load_events(self):
        grouped = defaultdict(list)
        for camera in self.cameras.values():
            if camera.enabled:
                events = []
                for f in glob.glob(f"{camera.recordings_dir}/*.json"):
                    with open(f) as fp:
                        events.append(json.load(fp))
                events.sort(key=lambda x: x["start_time"])
                grouped[camera.name] = events
        return grouped

    def _frame_reader(self, camera: Camera):
        """
        Thread to ead frames from the ffmpeg stdout stream and puts the frame on the camera queue.
        The queue length is 1, so if the queue is full that frame on the queue is dropped and
        replaced with the new frame. This means we drop frames to keep up. This is only for
        image processing, frames written to segments are not dropped
        """
        current_thread().name = f"{camera.name} _frame_reader"

        frame_size = self._width * self._height * 3

        while not self.stop_event.is_set():
            raw = self._read_exact(camera.process.stdout, frame_size)

            if raw is None:
                log_event(message="reader failed", level="warn", camera=camera)
                self._restart_camera(camera)
                continue

            frame = np.frombuffer(raw, np.uint8).reshape((self._height, self._width, 3))

            # FPS calculation
            now = time.perf_counter()
            if camera.last_frame_time > 0:
                dt = now - camera.last_frame_time

                # filter pipeline artifacts
                if 0.02 < dt < 0.2:
                    inst_fps = 1.0 / dt
                    camera.dt.update(dt)
                    camera.fps.update(1.0 / camera.dt.value())

            camera.last_frame_time = now

            # latest-frame-wins
            if camera.frame_queue.full():
                camera.frame_queue.get_nowait()
                camera.total_drops += 1
            camera.frame_queue.put(frame)
            camera.total_frames += 1
            camera.drop_rate = camera.total_drops / camera.total_frames


    def _read_exact(self, pipe, size):
        """
        reads bytes from the pipe until the buffer size is reached
        """
        buf = b""
        while len(buf) < size:
            chunk = pipe.read(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def draw_text(self, frame, text, position, font, font_scale, color, thickness, bg_color):
        """
        Utility function to draw text on a frame
        """
        if not text:
            return
        x, y = position
        text_size, _ = cv2.getTextSize(text, font, 0.7, thickness)
        text_w, text_h = text_size
        # rectangle: (x, y) top-left, (x + text_w, y + text_h) bottom-right
        # text: (x, y + text_h) bottom-left corner
        cv2.rectangle(frame, (x, y-2), (x + text_w + 2, y + text_h + 6), bg_color, -1)
        cv2.putText(frame, text, (x+1, y + text_h), font, font_scale, color, thickness)
        
    def _process_frames(self, camera: Camera):
        """
        Thread to process frames from the camera queue. Processing is as follows:
        1. get the frame from the queue (latest frame processing, some frames are dropped)
        2. convert to grayscale for image processing (faster than color)
        3. blur the gray (better for motion detection)
        4. calculate the difference between this gray frame and the previous one (for motion detection)
        5. calculate a theshold image based on the difference and score (count) the white pixels
        6. if the score is above threshold, compute the motion contours and rectangles from the threshold image
        7. draw contours and rectangles on a copy of the image. Red for movement that is too small, green for movement that we care about
        8. if we have movement we care about, run YOLO and check if movement and detected objects intersect
        9. if movement and objects intersect, start recording if we have seen motion for a number of frames, get a list of pre-record segments
        10. keep recording while there is motion, add to the segment list
        11. stop recording after motion is not detected for a number of frames
        12. get the list of segment files that correlate to the recording period
        13. merge the segment files in to a video file, asynchronously
        14. if there were YOLO results and movement, write the objects on to the image
        15. if there was motion, merge the image and overlay
        16. store the image and status in the camera object, the GUI will read this image and status at whatever rate it wants
        """
        current_thread().name = f"{camera.name} _process_frames"

        motion_frames = 0
        no_motion_frames = 0
        recording = False
        prev_time = time.time()
        is_night = 0

        while not self.stop_event.is_set():
            # get latest frame (non-blocking)
            try:
                frame = camera.frame_queue.get(timeout=0.5)
                frame_bgr = frame.copy() # we will draw on this, so copy it to avoid modifying the original frame in place which is used for motion detection and can cause weird artifacts if we draw on it directly
            except queue.Empty:
                continue

            if camera.first_frame:
                log_event(message=f"reading from stream", level="info", camera=camera)
                camera.first_frame = False

            now = time.time()

            # motion
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY, dst=camera.gray_buf)
            gray = cv2.GaussianBlur(gray, (21,21), 0, dst=camera.gray_buf)

            # initialize background model
            if camera.background_buf is None:
                camera.background_buf = gray.astype("float32")
                continue

            # periodic night/day check
            if now - camera.last_night_time_check > constants.PERIODIC_CHECK_INTERVAL:
                profile = self._get_motion_threshold(self._is_night_time(frame_bgr, constants.NIGHT_TIME_THRESHOLD))
                camera.last_night_time_check = time.time()
                if now - prev_time > 10.0:
                    log_event("stopped reading frames", level="info", camera=camera)

            # update background model
            cv2.accumulateWeighted(gray, dst=camera.background_buf, alpha=0.12 if is_night else 0.02)

            # convert background to uint8 for diff
            bg_frame = cv2.convertScaleAbs(camera.background_buf)

            # --- MOTION DIFF ---
            diff = cv2.absdiff(bg_frame, gray, dst=camera.diff_buf)

            # --- NOISE-ADAPTIVE LOW-INTENSITY FILTERING ---
            noise = np.std(diff)
            cutoff = max(8, min(20, noise * 1.5))

            _, diff_mask = cv2.threshold(diff, cutoff, 255, cv2.THRESH_BINARY)
            diff_filtered = cv2.bitwise_and(diff, diff_mask)

            # --- BLUR TO REDUCE HIGH-FREQUENCY NOISE ---
            diff_blur = cv2.GaussianBlur(diff_filtered, (7, 7), 0, dst=camera.diff_blur_buf)

            # --- OTSU THRESHOLD ON CLEANED DIFF ---
            _, thresh = cv2.threshold(
                diff_blur, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                dst=camera.thresh_buf
            )

            score = cv2.countNonZero(thresh)

            krs, kcs, dsrs, dscs, dars, dacs = [], [], [], [], [], []
            camera.motion_boxes_list.clear()
            if score > profile.motion_threshold:
                krs, kcs, dsrs, dscs, dars, dacs = self._find_motion_boxes(thresh, profile.motion_threshold, profile.min_solidity, profile.min_area_ratio)                
                camera.motion_boxes_list.extend(krs)

            # counters, we need motion (or no motion) for a number of consecutive frames to care
            if camera.motion_boxes_list:
                motion_frames += 1
                no_motion_frames = 0
            else:
                no_motion_frames += 1

            # Normalize score to [0, 1]
            pixel_score = min(score / (self._max_pixels * profile.motion_threshold), 1.0)  # profile.motion_threshold% of frame = full confidence

            # Box score: larger boxes = higher confidence
            box_areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in camera.motion_boxes_list]
            box_score = min(sum(box_areas) / (self._max_pixels * 0.10), 1.0) if box_areas else 0

            # Persistence score: more consecutive frames = higher confidence
            persist_score = min(motion_frames / constants.MOTION_DETECT_FRAME_COUNT, 1.0)

            # Final motion confidence
            motion_confidence = (pixel_score * 0.5) + (box_score * 0.3) + (persist_score * 0.2)
            camera.motion_confidence = motion_confidence

            if not camera.motion_boxes_list and score < profile.motion_threshold:
                camera.motion_confidence = 0.0

            self.draw_text(frame_bgr, camera.status_text, (0, 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (0, 255, 0), 2, (32, 32, 32))
            self.draw_text(frame_bgr, camera.objects_text, (0, 27),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, (32, 32, 32))
            
            if camera.debug:
                # --- BUILD 4-PANEL DEBUG COMPOSITE ---

                # 1. Original frame
                panel_orig = frame_bgr.copy()
                cv2.putText(panel_orig, "Original Frame", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 2. Background model (convert float32 → uint8)
                panel_bg = cv2.convertScaleAbs(camera.background_buf)
                panel_bg = cv2.cvtColor(panel_bg, cv2.COLOR_GRAY2BGR)
                cv2.putText(panel_bg, "Background Model", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 3. Diff (or diff_filtered)
                panel_diff = cv2.cvtColor(diff_filtered, cv2.COLOR_GRAY2BGR)
                cv2.putText(panel_diff, "Diff (Filtered)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 4. Threshold image (with motion boxes)
                panel_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                cv2.putText(panel_thresh, "Threshold + Motion Boxes", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # --- DRAW MOTION BOXES ON THRESH PANEL ---
                for (x1, y1, x2, y2) in krs:
                    cv2.rectangle(panel_thresh, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for (x1, y1, x2, y2) in dsrs:
                    cv2.rectangle(panel_thresh, (x1, y1), (x2, y2), (0, 165, 255), 2)

                for (x1, y1, x2, y2) in dars:
                    cv2.rectangle(panel_thresh, (x1, y1), (x2, y2), (0, 0, 255), 2)

                vpos = 55
                cv2.putText(panel_thresh, f"score={score}", (10, vpos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                vpos += 30
                cv2.putText(panel_thresh, f"conf={camera.motion_confidence:.2f}", (10, vpos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                vpos += 30
                cv2.putText(panel_thresh, f"conf={pixel_score:.2f}", (10, vpos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                vpos += 30
                cv2.putText(panel_thresh, f"box={box_score:.2f}", (10, vpos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                vpos += 30
                cv2.putText(panel_thresh, f"persist={persist_score:.2f}", (10, vpos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                vpos += 30
                cv2.putText(panel_thresh, f"box count={len(camera.motion_boxes_list)}", (10, vpos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                vpos += 30
                cv2.putText(panel_thresh, f"objs={self._tags_to_str(camera.active_objects_dict)}", (10, vpos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # --- RESIZE PANELS ---
                h, w = frame_bgr.shape[:2]
                half_w = w // 2
                half_h = h // 2

                p1 = cv2.resize(panel_orig, (half_w, half_h))
                p2 = cv2.resize(panel_bg, (half_w, half_h))
                p3 = cv2.resize(panel_diff, (half_w, half_h))
                p4 = cv2.resize(panel_thresh, (half_w, half_h))

                # --- STACK INTO 4-PANEL COMPOSITE ---
                top = np.hstack((p1, p2))
                bottom = np.hstack((p3, p4))
                composite = np.vstack((top, bottom))

                camera.debug_motion_image = composite

            # YOLO
            result = None
            camera.classes_in_frame_dict.clear()

            # Only run YOLO if we have meaningful motion
            # if there is large enough motion boxes, run YOLO and see if objects we care about overlap
            # with the motion boxes (either the object is moving, or something is moving across the object)
            if camera.debug or camera.motion_boxes_list: # and (time.time() - camera.last_yolo_time > 0.2)):
                camera.last_yolo_time = time.time()

                result = camera.model.model.predict(
                    frame_bgr,
                    conf=profile.yolo_conf,
                    classes=self.selected_classes if self.selected_classes else None,
                    verbose=False,
                    imgsz=512,
                )[0]

                # YOLO boxes (N,4)
                boxes_xyxy = result.boxes.xyxy.reshape(-1, 4)

                # Convert motion boxes to tensor
                if camera.motion_boxes_list:
                    motion_tensor = torch.as_tensor(
                        camera.motion_boxes_list,
                        dtype=boxes_xyxy.dtype,
                        device=boxes_xyxy.device
                    )

                    # Filter YOLO boxes by overlap with motion
                    keep_mask = self._keep_overlapping_any(boxes_xyxy, motion_tensor)
                    moving_boxes = result.boxes[keep_mask]
                else:
                    moving_boxes = []

                # Pick the YOLO box with the strongest overlap
                moving_box = self.get_most_moving_yolo_box(moving_boxes, camera.motion_boxes_list)

                if moving_box is not None:
                    # Extract ROI
                    roi = self.yolo_box_to_roi(frame_bgr, moving_box)

                    # Detect color
                    color = self._detect_object_color(roi)

                    # Class name
                    class_name = self.model.model.names[int(moving_box.cls)]

                    # Store in camera state
                    camera.classes_in_frame_dict[class_name].add(color)

                    if camera.debug and self.debug:
                        log_event(
                            message=f"Detected moving {color} {class_name} score {score} confidence {camera.motion_confidence:.2f} moving_box {moving_box} boxes_xyxy {boxes_xyxy} motion_boxes {camera.motion_boxes_list}",
                            level="debug",
                            camera=camera
                        )

            # start
            if motion_frames >= constants.MOTION_DETECT_FRAME_COUNT and not recording:
                valid_objects = len(camera.classes_in_frame_dict) > 0
                if valid_objects:
                    if now - camera.last_event_time > constants.EVENT_COOLDOWN:
                        recording = True
                        camera.active_segments_list = self._get_segments(camera, constants.PRE_RECORD_SEGMENTS)
                        camera.active_objects_dict = deepcopy(camera.classes_in_frame_dict)
                        camera.last_event_time = now
                        log_event(message=f"recording start {self._tags_to_str(camera.active_objects_dict)}", level="info", camera=camera)

            # update active segments and objects
            if recording:
                camera.active_segments_list += self._get_segments(camera,1)
                for item, colors in camera.classes_in_frame_dict.items():
                    camera.active_objects_dict[item].update(colors)

            # stop recoding when there has been no motion for some time
            if recording and no_motion_frames >= (constants.NO_MOTION_DETECT_FRAME_COUNT if not camera.debug else constants.NO_MOTION_DETECT_FRAME_COUNT/10):
                recording = False
                segments = list(dict.fromkeys(camera.active_segments_list))

                # if we have segments, merge them into an mp4 file with timestamp and tags
                if segments:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tags = deepcopy(camera.active_objects_dict)
                    recording_filename = os.path.join(camera.recordings_dir, f"{timestamp}.mp4")
                    self._merge_segments_async(camera, segments, tags, recording_filename)

                camera.active_segments_list.clear()
                camera.classes_in_frame_dict.clear()
                camera.active_objects_dict.clear()
                motion_frames = 0
                no_motion_frames = 0

            # render YOLO plots on the frame if there was a result
            if result:
                img_bgr = result.plot(pil=False) # pil=False returns BGR
            else:
                img_bgr = frame_bgr

            if camera.debug and camera.debug_motion_image is not None:
                #img_bgr = cv2.addWeighted(img_bgr, 0.5, camera.debug_motion_image, 0.5, 0)
                img_bgr = camera.debug_motion_image

            camera.latest_frame = img_bgr

            prev_time = time.time()

            parts = [self._make_status(recording)]
            if is_night:
                parts.append("Night")

            parts.append(f"FPS {int(camera.fps.value())}:{camera.drop_rate:.2f}")
            camera.objects_text = self._tags_to_str(camera.active_objects_dict)
            camera.status_text = " | ".join(parts)            


    def _make_status(self, recording: bool):
        """
        creates a string that represents the status (red/green for recording/live)
        """
        idx = int(time.time() * 4) % 4

        record_cycle = ["*", "*", " ", " "]

        pulse = record_cycle[idx] if recording else ""

        return f"{pulse}{'REC' if recording else 'LIVE'}"
    

    def _find_motion_boxes(self, thresh, pixel_threshold, min_solidity=0.5, min_area_ratio=0.002):
        """
        Find motion boxes using contour analysis with solidity filtering.
        Returns:
            kept_rects, kept_contours,
            small_rects, small_contours,
            angular_rects, angular_contours
        """

        h, w = thresh.shape[:2]
        frame_area = w * h
        min_area = frame_area * min_area_ratio  # e.g., 0.2% of frame

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        kept_rects = []
        kept_contours = []

        small_rects = []
        small_contours = []

        angular_rects = []
        angular_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1:
                continue

            # --- MINIMUM AREA FILTER ---
            if area < min_area:
                x, y, w0, h0 = cv2.boundingRect(cnt)
                small_rects.append((x, y, x + w0, y + h0))
                small_contours.append(cnt)
                continue

            # --- SOLIDITY FILTER ---
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            solidity = float(area) / hull_area

            if solidity < min_solidity:
                # This is leaf/branch/shadow motion
                x, y, w0, h0 = cv2.boundingRect(cnt)
                angular_rects.append((x, y, x + w0, y + h0))
                angular_contours.append(cnt)
                continue

            # --- ASPECT RATIO FILTER (optional but useful) ---
            x, y, w0, h0 = cv2.boundingRect(cnt)
            aspect = max(w0, h0) / max(1, min(w0, h0))

            if aspect > 6.0:
                # Very long skinny shapes = noise (grass, shadows)
                angular_rects.append((x, y, x + w0, y + h0))
                angular_contours.append(cnt)
                continue

            # --- ACCEPTED MOTION BOX ---
            kept_rects.append((x, y, x + w0, y + h0))
            kept_contours.append(cnt)

        return (
            kept_rects, kept_contours,
            small_rects, small_contours,
            angular_rects, angular_contours
        )


    def _is_night_time(self, frame, brightness_threshold=50):
        """
        determines if we are looking at a night time image.
        Converts the frame to HSV and computes the mean value of intensity channel
        it's night time if below the threshold, else it's day time
        """
        # Convert to HSV (Hue, Saturation, Intensity)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate average brightness (V channel - Intensity)
        mean_brightness = np.mean(hsv[:,:,2])
        
        # If brightness is low, it's likely night time
        return mean_brightness < brightness_threshold
    
    def get_most_moving_yolo_box(self, yolo_boxes, motion_boxes):
        best_box = None
        best_overlap = 0.0

        for box in yolo_boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_area = max(1, (x2 - x1) * (y2 - y1))

            for mx1, my1, mx2, my2 in motion_boxes:
                ix1 = max(x1, mx1)
                iy1 = max(y1, my1)
                ix2 = min(x2, mx2)
                iy2 = min(y2, my2)

                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                overlap_ratio = inter_area / box_area

                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_box = box

        return best_box


    def _keep_overlapping_any(self, boxes, ref_boxes):
        """
        compute the pairwise intersection of the motion boxes and object boxes.
        Return the filter so caller can weed out YOLO objects to draw/not draw
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

    def yolo_box_to_roi(self, frame_bgr, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Clamp to image bounds
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        roi = frame_bgr[y1:y2, x1:x2].copy()
        return roi


    def _detect_object_color(self, roi_bgr, k=2):
        if roi_bgr is None or roi_bgr.size == 0:
            return "unknown"

        # Smooth noise
        roi = cv2.GaussianBlur(roi_bgr, (5, 5), 0)

        # Convert to LAB (OpenCV LAB)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Convert OpenCV LAB → true LAB
        lab[:, :, 1] -= 128.0   # a channel shift
        lab[:, :, 2] -= 128.0   # b channel shift

        # Flatten for k-means
        pixels = lab.reshape((-1, 3))

        # K-means clustering
        _, labels, centers = cv2.kmeans(
            pixels, k, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
            3,
            cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(labels.flatten())
        sorted_idx = np.argsort(-counts)

        total = len(pixels)

        for idx in sorted_idx:
            # Ignore tiny clusters (noise, highlights)
            if counts[idx] < 0.05 * total:
                continue

            lab_color = centers[idx]
            color_name = self._classify_color_lab(lab_color)

            if color_name != "unknown":
                return color_name

        return "unknown"


    def _classify_color_lab(self, lab_color):
    # LAB-based color classifier
        L, a, b = lab_color
        chroma = sqrt(a*a + b*b)

        # Neutral detection
        if L < 30:
            return "black"
        if chroma < 10:
            return "white" if L > 200 else "gray"

        # Metallic detection
        if 60 < L < 95 and chroma < 25:
            return "silver"
        if 60 < L < 95 and 25 <= chroma < 45 and b > 20:
            return "gold"

        # Earth tone detection
        if 30 < L < 70 and 10 < chroma < 40:
            if b > 25:
                return "tan"
            if 10 < b <= 25:
                return "beige"
            if b <= 10:
                return "brown"

        # Standard color classification
        best = None
        best_dist = 1e9

        for name, ref in constants.REF_COLORS.items():
            dist = np.linalg.norm(lab_color - ref)
            if dist < best_dist:
                best_dist = dist
                best = name

        return best
