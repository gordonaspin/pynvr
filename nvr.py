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

def _keep_overlapping_any(boxes, ref_boxes):
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

import cv2
import numpy as np

def detect_object_color(roi_bgr, k=3):
    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown"

    h, w = roi_bgr.shape[:2]

    # -----------------------------
    # 1. Crop center (reduce background)
    # -----------------------------
    pad = 0.2
    roi = roi_bgr[
        int(h*pad):int(h*(1-pad)),
        int(w*pad):int(w*(1-pad))
    ]

    if roi.size == 0:
        return "unknown"

    # -----------------------------
    # 2. Resize for speed
    # -----------------------------
    roi = cv2.resize(roi, (64, 64))

    # -----------------------------
    # 3. Convert to LAB (better color distance)
    # -----------------------------
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)

    # -----------------------------
    # 4. K-means clustering
    # -----------------------------
    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        5,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Count cluster sizes
    counts = np.bincount(labels.flatten())

    # Sort clusters by dominance
    sorted_idx = np.argsort(-counts)

    for idx in sorted_idx:
        color_lab = centers[idx]

        # Convert LAB → BGR → HSV for classification
        color_bgr = cv2.cvtColor(
            np.uint8([[color_lab]]),
            cv2.COLOR_LAB2BGR
        )[0][0]

        color_name = classify_color(color_bgr)

        if color_name != "unknown":
            return color_name

    return "unknown"

def classify_color(bgr):
    b, g, r = bgr
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    # -----------------------------
    # Handle neutrals first
    # -----------------------------
    if v < 50:
        return "black"
    if s < 40:
        if v > 200:
            return "white"
        return "gray"

    # -----------------------------
    # Hue-based classification
    # -----------------------------
    if h < 10 or h >= 170:
        return "red"
    elif h < 25:
        return "orange"
    elif h < 35:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 125:
        return "blue"
    elif h < 155:
        return "purple"
    else:
        return "pink"

def get_dominant_color_name(roi_bgr):
    if roi_bgr.size == 0:
        return "unknown"

    # Convert to HSV
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Flatten pixels
    pixels = hsv.reshape(-1, 3)

    # Filter out low saturation (removes white/gray/black)
    pixels = pixels[pixels[:,1] > 40]

    if len(pixels) == 0:
        return "gray"

    # Take hue channel
    hues = pixels[:,0]

    # Compute histogram
    hist = np.bincount(hues, minlength=180)

    dominant_hue = np.argmax(hist)

    # Map hue to color
    return hue_to_color_name(dominant_hue)
    
def hue_to_color_name(h):
    if h < 10 or h >= 170:
        return "red"
    elif h < 25:
        return "orange"
    elif h < 35:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 125:
        return "blue"
    elif h < 160:
        return "purple"
    else:
        return "pink"

# =========================
# NVR ENGINE
# =========================
class NVR:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.model = Model(ctx)
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
                    threading.Thread(target=self._frame_reader, args=(camera,), daemon=True).start()
                    threading.Thread(target=self._process_frames,args=(camera,), daemon=True).start()
            threading.Thread(target=self._cleanup_segments,daemon=True).start()
            threading.Thread(target=self._watch_cameras,daemon=True).start()

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
        """
        get the list of segment file for this camera for the duration it was recording
        """
        files = sorted(glob.glob(os.path.join(camera.segments_dir, "*.ts")))
        return files[-n:]


    def _merge_segments_async(self, camera: Camera, files: list[str], output: str):
        """
        Runs ffmpeg merge in a separate thread. When the process finishes,
        the log the event and delete the listing file.
        """

        def worker():
            list_file = output + ".txt"
            with open(list_file,"w") as f:
                for x in files:
                    try:
                        if os.stat(x).st_size > 0:
                            f.write(f"file '{os.path.abspath(x)}'\n")
                    except FileNotFoundError as e:
                        pass
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
                self._merge_complete(camera, output)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread


    def _merge_complete(self, camera: Camera, output: str):
        """
        logs the merge completion event and deletes recording if too short
        """
        cap = cv2.VideoCapture(output)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_seconds = frame_count / fps
        formatted_duration = str(timedelta(seconds=int(duration_seconds)))
        if frame_count < constants.NO_MOTION_DETECT_FRAME_COUNT + 20 and os.path.isfile(output):
            os.remove(output)
            log_event(message=f"recording auto-deleted {os.path.basename(output)} with {frame_count} frames", level="info", camera=camera, file_path=output)
        else:
            log_event(message=f"recording available {formatted_duration} {os.path.basename(output)}", level="record", camera=camera, file_path=output)


    def _frame_reader(self, camera: Camera):
        """
        Thread to ead frames from the ffmpeg stdout stream and puts the frame on the camera queue.
        The queue length is 1, so if the queue is full that frame on the queue is dropped and
        replaced with the new frame. This means we drop frames to keep up. This is only for
        image processing, frames written to segments are not dropped
        """
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
        threading.current_thread().name = f"{camera.name} _process_frames"

        prev_gray = None
        motion_frames = 0
        no_motion_frames = 0
        recording = False
        prev_time = time.time()
        is_night = 0

        while not self.stop_event.is_set():
            # get latest frame (non-blocking)
            try:
                frame_bgr = camera.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if camera.first_frame:
                log_event(message=f"reading from stream", level="info", camera=camera)
                camera.first_frame = False

            now = time.time()

            # motion
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY, dst=camera.gray_buf)
            gray = cv2.GaussianBlur(gray, (21,21), 0, dst=camera.gray_buf)

            if prev_gray is None:
                prev_gray = gray
                continue

            if now - camera.last_night_time_check > constants.PERIODIC_CHECK_INTERVAL:
                is_night = 1 if _is_night_time(frame_bgr, constants.NIGHT_TIME_THRESHOLD) else 0
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

            overlay_bgr = None
            if self.debug and any([krs, kcs, dsrs, dscs, dars, dacs]):
                # draw on a copy of the image
                motion = "captured" if len(camera.motion_boxes_list) else "discarded"
                overlay_bgr = frame_bgr.copy()
                for x1, y1, x2, y2 in krs:
                    cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2) # green BGR
                for x1, y1, x2, y2 in dsrs:
                    cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 96, 255), 2) # orange
                for x1, y1, x2, y2 in dars:
                    cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2) # red
                for c in kcs:
                    cv2.drawContours(overlay_bgr, [c], -1, (0, 255, 0), 2) # green
                for c in dscs:
                    cv2.drawContours(overlay_bgr, [c], -1, (0, 96, 255), 2) # orange
                for c in dacs:
                    cv2.drawContours(overlay_bgr, [c], -1, (0, 0, 255), 2) # red
                if self.debug_files:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tag = "_".join( camera.active_objects_set) if  camera.active_objects_set else "motion"
                    image_filename = os.path.join(camera.images_dir, f"{timestamp}_{score}_{motion}_{tag}.jpg")                
                    cv2.imwrite(image_filename, overlay_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    log_event(message=f"contour image written to {image_filename}", level="debug", camera=camera, file_path=image_filename)

            prev_gray = gray

            # YOLO
            result = None
            camera.classes_in_frame_set.clear()

            # if there is large enough motion boxes, run YOLO and see if objects we care about overlap
            # with the motion boxes (either the object is moving, or something is moving across the object)
            if self.debug or camera.motion_boxes_list: # and (time.time() - camera.last_yolo_time > 0.2)):
                camera.last_yolo_time = time.time()
                result = camera.model.model.predict(frame_bgr, conf=self.confidence_threshold, classes=self.selected_classes if self.selected_classes else None, verbose=False)[0]
                boxes = result.boxes.xyxy.reshape(-1, 4)
                ref_motion_boxes_list = torch.as_tensor(camera.motion_boxes_list, dtype=boxes.dtype, device=boxes.device)
                keep = _keep_overlapping_any(boxes, ref_motion_boxes_list)
                boxes = result.boxes[keep]

                # store the color and name/class of object we saw in the frame that coincides with movement
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = frame_bgr[y1:y2, x1:x2]
                    color = detect_object_color(roi)
                    class_name = self.model.model.names[int(box.cls)]
                    camera.classes_in_frame_set.add(f"{color}-{class_name}")

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
                        tag = ", ".join(camera.active_objects_set) if camera.active_objects_set else ""
                        log_event(message=f"recording start {tag}", level="info", camera=camera)

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
                    #self._merge_segments(segments, recording_filename)
                    self._merge_segments_async(camera, segments, recording_filename)

                camera.active_segments_list.clear()
                camera.classes_in_frame_set.clear()
                camera.active_objects_set.clear()
                motion_frames = 0
                no_motion_frames = 0

            # render YOLO plots on the frame if there was a result
            if result:
                img_bgr = result.plot(pil=False) # pil=False returns BGR
            else:
                img_bgr = frame_bgr

            if overlay_bgr is not None:
                img_bgr = cv2.addWeighted(img_bgr, 0.5, overlay_bgr, 0.5, 0)

            if not self.cameras[camera.name].hd:
                img_bgr = cv2.resize(img_bgr, constants.RENDER_SIZE)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            prev_time = time.time()

            parts = [self.make_status(recording)]
            if is_night:
                parts.append("Night")

            parts.append(f"FPS {int(camera.fps.value())}:{camera.drop_rate:.2f}")
            if camera.active_objects_set:
                parts.append(",".join(camera.active_objects_set))

            camera.latest_frame = img_rgb
            camera.status = " | ".join(parts)            


    def make_status(self, recording: bool):
        """
        creates a string that represents the status (red/green for recording/live)
        """
        idx = int(time.time() * 4) % 4

        red_cycle = ["🔴", "🔴", "⚪", "⚪"]
        green_cycle = ["🟢", "🟢", "⚪", "⚪"]

        pulse = red_cycle[idx] if recording else green_cycle[idx]

        return f"{pulse}{' REC' if recording else ' LIVE'}"
    
    def _find_motion_boxes(self, thresh: tuple, motion_threshold, motion_factor: float, area_factor: float):
        """
        using the threshold image, find contours and its bounding rectangle
        if the contour area is below the motion score by the motion_factor, add to the discard list
        if the contou area is smaller than its bounding rectangle by the area_factor, add to the discard list
        else add the contour and rectangle to the keep list
        return the lists so we can draw them on the overlay
        This does not determine what we consider momtion, but helps identify what we ignore 
        """

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
    
