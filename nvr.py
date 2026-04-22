from datetime import datetime, timedelta
import glob
from logging import getLogger
import os
import queue
import subprocess
import threading
import time

import cv2
from ffmpeg import FFmpeg
import torch
import numpy as np

from camera import Camera
import constants
from context import Context
from logger import log_event
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
        self.stop_event = threading.Event()
        self.debug = self.ctx.debug
        self.debug_files = self.ctx.debug_files
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
                                        model=Model(ctx)
                                        )

    def start(self):
        if not self.stop_event.is_set():
            for camera in self.cameras.values():
                if camera.enabled:
                    os.makedirs(camera.recordings_dir, exist_ok=True)
                    os.makedirs(camera.segments_dir, exist_ok=True)
                    os.makedirs(camera.images_dir, exist_ok=True)
                    self._start_camera(camera=camera)
                    threading.Thread(target=self._frame_reader, args=(camera,), daemon=True).start()
                    threading.Thread(target=self._process_frames,args=(camera,), daemon=True).start()
            threading.Thread(target=self._cleanup_segments,daemon=True).start()

    def stop(self):
        for camera in self.cameras.values():
            if camera.enabled and camera.process is not None:
                self._stop_camera(camera)

    def _restart_camera(self, camera):
        if not self.stop_event.is_set():
            log_event(message="restarting camera", level="warn", camera=camera)
            self._stop_camera(camera)
            self._start_camera(camera)

    def _stop_camera(self, camera):
        if camera.enabled and camera.process is not None:
            ret = camera.process.poll()
            log_event(message=f"stopping camera with ret {ret}", level="info", camera=camera)
            camera.process.terminate()

            try:
                camera.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                camera.process.kill()
            camera.process.stdout.close()
            camera.process.stdin.close()

    def _start_camera(self, camera: Camera):

        if not self.stop_event.is_set():
            log_event(message=f"starting recorder", level="info", camera=camera)
            filespec = os.path.join(camera.segments_dir, "%Y%m%d_%H%M%S.ts")
            log_file = open(f"{camera.name}_ffmpeg.log", "w")
            ffmpeg_cmd = [
                "ffmpeg",

                "-rtsp_transport", "tcp",           # Forces RTSP over TCP instead of UDP
                "-fflags", "nobuffer+genpts",       # Disables internal buffering, generates PTS
                "-flags", "low_delay",              # Tells decoder/demuxer to minimize delay (Reduces frame reordering buffers)
                #"-use_wallclock_as_timestamps", "1",# Uses system clock instead of stream timestamps (RTSP streams often have missing timestamps)
                "-i", camera.url,                   # RTSP stream from camera

                
                "-filter_complex",                  # Split and reduce scale for raw only for OpenCV
                f"[0:v]scale={self.width}:{self.height},format=bgr24[raw]", # re-scale and raw BGR pixel format (OpenCV native)

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
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_file,
                bufsize=10**8
            )
            camera.process = process
            return process

    def _cleanup_segments(self):
        threading.current_thread().name = "cleanup_segments"
            
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
        files = sorted(glob.glob(os.path.join(camera.segments_dir, "*.ts")))
        return files[-n:]

    def _merge_segments(self, files, output):
        list_file = output + ".txt"
        with open(list_file,"w") as f:
            for x in files:
                try:
                    if os.stat(x).st_size > 0:
                        f.write(f"file '{os.path.abspath(x)}'\n")
                except FileNotFoundError as e:
                    pass

        try:
            (
                FFmpeg()
                .option("y")
                .option("fflags", "+genpts")          # regenerate timestamps
                .input(list_file, f="concat", safe=0)
                .output(
                    output,
                    c="libx264",                    # Re-encodes video using H.264 codec
                    pix_fmt="yuv420p",              # Forces pixel format to 4:2:0 (required for iOS Safari, Android browsers, HTML5 <video>)
                    movflags="+faststart",          # Moves MP4 metadata (moov atom) to the beginning
                    preset="veryfast",              # Controls encoding speed vs compression efficiency (ultrafast → superfast → veryfast → faster → fast → medium → slow)
                    crf=23,                         # Constant Rate Factor (quality control: 18 = visually lossless, 23 = default (balanced), 28+ = lower quality)
                    vsync="cfr",                    # constant frame pacing
                    r=20,                           # normalize FPS
                    video_track_timescale=90000     # smoother playback on mobile (Sets MP4 timebase resolution, 90000 Standard MPEG clock (used in TS, RTP, etc.)
                )
                .execute()
            )
        except Exception as e:
            log_event(f"ffmpeg merge {list_file} failed {e}", level="error")

        os.remove(list_file)

    def _frame_reader(self, camera: Camera):
        threading.current_thread().name = f"{camera.name} _frame_reader"

        frame_size = self.width * self.height * 3

        while not self.stop_event.is_set():
            raw = self._read_exact(camera.process.stdout, frame_size)

            if raw is None:
                log_event(message="reader failed", level="warn", camera=camera)
                self._restart_camera(camera)
                continue

            frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))

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

        while not self.stop_event.is_set():
            # get latest frame (non-blocking)
            try:
                frame = camera.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if first_frame:
                log_event(message=f"first frame read from stream", level="info", camera=camera)
                first_frame = False

            now = time.time()

            # motion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=camera.gray_buf)
            gray = cv2.GaussianBlur(gray, (21,21), 0, dst=camera.gray_buf)

            if prev_gray is None:
                prev_gray = gray
                continue

            if now - camera.last_night_time_check > constants.PERIODIC_CHECK_INTERVAL:
                is_night = 1 if _is_night_time(frame, constants.NIGHT_TIME_THRESHOLD) else 0
                camera.last_night_time_check = time.time()
                if now - prev_time > 10.0:
                    log_event("stopped reading frames", level="info", camera=camera)


            # look for motion in this frame, compared to previous
            diff = cv2.absdiff(prev_gray, gray, dst=camera.diff_buf)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY, dst=camera.thresh_buf)
            score = cv2.countNonZero(thresh)

            krs, kcs, dsrs, dscs, dars, dacs = [], [], [], [], [], []
            camera.motion_boxes_list.clear()
            if score > self.motion_threshold[is_night]:
                krs, kcs, dsrs, dscs, dars, dacs = self._find_motion_boxes(thresh, self.motion_threshold[is_night], 0.1, 0.25)                
                camera.motion_boxes_list.extend(krs)

            overlay = None
            if self.debug and any([krs, kcs, dsrs, dscs, dars, dacs]):
                # draw on a copy of the image
                motion = "captured" if len(camera.motion_boxes_list) else "discarded"
                overlay = frame.copy()
                for x1, y1, x2, y2 in krs:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2) # green BGR
                for x1, y1, x2, y2 in dsrs:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 96, 255), 2) # orange
                for x1, y1, x2, y2 in dars:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2) # red
                for c in kcs:
                    cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2) # green
                for c in dscs:
                    cv2.drawContours(overlay, [c], -1, (0, 96, 255), 2) # orange
                for c in dacs:
                    cv2.drawContours(overlay, [c], -1, (0, 0, 255), 2) # red
                if self.debug_files:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tag = "_".join( camera.active_objects_set) if  camera.active_objects_set else "motion"
                    image_filename = os.path.join(camera.images_dir, f"{timestamp}_{score}_{motion}_{tag}.jpg")                
                    cv2.imwrite(image_filename, overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    log_event(message=f"contour image written to {image_filename}", level="debug", camera=camera, file_path=image_filename)

            prev_gray = gray

            # YOLO
            result = None
            camera.classes_in_frame_set.clear()

            # if there is large enough motion boxes, run YOLO and see if objects we care about overlap
            # with the motion boxes (either the object is moving, or something is moving across the object)
            if self.debug or camera.motion_boxes_list: # and (time.time() - camera.last_yolo_time > 0.2)):
                camera.last_yolo_time = time.time()
                result = camera.model.model.predict(frame, conf=self.confidence_threshold, classes=self.selected_classes if self.selected_classes else None, verbose=False)[0]
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
                if valid_objects:
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
            if recording and no_motion_frames >= (constants.NO_MOTION_DETECT_FRAME_COUNT if not self.debug else constants.NO_MOTION_DETECT_FRAME_COUNT/10):
                recording = False
                segments = list(dict.fromkeys(camera.active_segments_list))

                # if we have segments, merge them into an mp4 file with timestamp and tags
                if segments:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tag = "_".join(camera.active_objects_set) if camera.active_objects_set else "motion"

                    recording_filename = os.path.join(camera.recordings_dir, f"{timestamp}_{tag}.mp4")
                    self._merge_segments(segments, recording_filename)

                    cap = cv2.VideoCapture(recording_filename)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration_seconds = frame_count / fps
                    formatted_duration = str(timedelta(seconds=int(duration_seconds)))
                    if frame_count < constants.NO_MOTION_DETECT_FRAME_COUNT + 20 and os.path.isfile(recording_filename):
                        os.remove(recording_filename)
                        log_event(message=f"recording auto-deleted {os.path.basename(recording_filename)} with {frame_count} frames", level="info", camera=camera, file_path=recording_filename)
                    else:
                        log_event(message=f"recording available {formatted_duration} {os.path.basename(recording_filename)}", level="record", camera=camera, file_path=recording_filename)

                camera.active_segments_list.clear()
                camera.classes_in_frame_set.clear()
                camera.active_objects_set.clear()
                motion_frames = 0
                no_motion_frames = 0

            # render YOLO plots on the frame if there was a result
            img = result.plot() if result else frame
            if overlay is not None:
                img = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if not self.cameras[camera.name].hd:
                img = cv2.resize(img, constants.RENDER_SIZE)

            prev_time = time.time()

            parts = [self.make_status(recording)]
            if is_night:
                parts.append("Night")

            parts.append(f"FPS {int(camera.fps.value())}:{camera.drop_rate:.2f}")
            if camera.active_objects_set:
                parts.append(",".join(camera.active_objects_set))

            camera.latest_frame = img
            camera.status = " | ".join(parts)            


    def make_status(self, recording: bool):
        idx = int(time.time() * 4) % 4

        red_cycle = ["🔴", "🔴", "⚪", "⚪"]
        green_cycle = ["🟢", "🟢", "⚪", "⚪"]

        pulse = red_cycle[idx] if recording else green_cycle[idx]

        return f"{pulse}{' REC' if recording else ' LIVE'}"
    
    def _find_motion_boxes(self, thresh: tuple, motion_threshold, motion_factor: float, area_factor: float):

        keep_rects = []
        keep_contours = []
        discard_small_rects = []
        discard_angular_rects = []
        discard_small_contours = []
        discard_angular_contours = []
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            x1, y1, w, h = cv2.boundingRect(contour)
            x2 = x1 + w
            y2 = y1 + h
            rect = [x1, y1, x2, y2]
            bounding_rect_area = w * h
            if contour_area < motion_threshold * motion_factor:  # filter small noise
                discard_small_rects.append(rect)
                discard_small_contours.append(contour)
                #if self.debug:
                #    log_event(message=f"ignoring motion contour rect ({x1}, {y1}), ({x2}, {y2}) with area {contour_area}")
            elif contour_area < bounding_rect_area * area_factor: # filter smaller than bounding rect (angular)
                discard_angular_rects.append(rect)
                discard_angular_contours.append(contour)
                #if self.debug:
                #    log_event(message=f"ignoring angular motion ({x1}, {y1}), ({x2}, {y2}) with area {contour_area} rect area {bounding_rect_area}")
            else:
                keep_rects.append(rect)
                keep_contours.append(contour)
        
        return keep_rects, keep_contours, discard_small_rects, discard_small_contours, discard_angular_rects, discard_angular_contours