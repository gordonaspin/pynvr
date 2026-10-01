"""
Frame recorder classes for handling video recording from cameras.
"""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from threading import Event, current_thread
from typing import override
from urllib.parse import quote

import av
import cv2

from pynvr.camera.camera import Camera
from pynvr.config.config import RecorderConfig
from pynvr.file_cleaner import FileCleaner
from pynvr.utils import make_readable_ts, make_ts_string, tags_to_str, RollingAverage

logger = getLogger("pynvr.recorder")

class FrameRecorderFactory:
    """Factory class to create FrameRecorder instances based on the specified recorder type."""
    @staticmethod
    def create(
        camera: Camera,
        recorder_name: str,
        stop_event: Event,
        add_recording_callback: Callable,
        recorder_config: RecorderConfig
        ):
        """
        Create a FrameRecorder instance based on the specified recorder_name.
        """
        match recorder_name:
            case "OpenCVFrame":
                return OpenCVFrameRecorder(
                    camera=camera,
                    stop_event=stop_event,
                    add_recording_callback=add_recording_callback,
                    recorder_config=recorder_config)
            case "AVFFmpegFrame":
                return AVFFmpegFrameRecorder(
                    camera=camera,
                    stop_event=stop_event,
                    add_recording_callback=add_recording_callback,
                    recorder_config=recorder_config)
            case "FFmpegFrame":
                return FFmpegFrameRecorder(
                    camera=camera,
                    stop_event=stop_event,
                    add_recording_callback=add_recording_callback,
                    recorder_config=recorder_config)
            case "FFmpegSegment":
                return FFmpegSegmentRecorder(
                    camera=camera,
                    stop_event=stop_event,
                    add_recording_callback=add_recording_callback,
                    recorder_config=recorder_config)
            case _:
                logger.warning(
                    f"Unknown '{recorder_name}', defaulting to AVFFmpegFrameRecorderFactory")
                return AVFFmpegFrameRecorder(
                    camera=camera,
                    stop_event=stop_event,
                    add_recording_callback=add_recording_callback,
                    recorder_config=recorder_config)

class FrameRecorder:
    """Base class for frame recorders that handle recording video from a camera."""
    def __init__(
            self,
            camera: Camera,
            stop_event: Event,
            add_recording_callback: Callable,
            recorder_config: RecorderConfig
            ):
        self.camera: Camera = camera
        self.stop_event = stop_event
        self.add_recording_callback: Callable = add_recording_callback
        self.recorder_config: RecorderConfig = recorder_config
        self.max_pre_frames: int = int(20 * recorder_config.pre_duration)
        self.uuid: str = str(uuid.uuid4())
        # Rolling buffer (automatically discards frames past the time limit)
        self.rolling_buffer: deque = deque(maxlen=self.max_pre_frames)

        # Recording queue (unlimited size, used during active recording)
        self.record_queue: deque = deque()

        self.frame_count: int = 0
        self.window_start: float = time.time()
        # Thread management states
        self.event: threading.Event = threading.Event()
        self.lock: threading.Lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.duration_seconds: float = 0
        self.formatted_duration: str = "0:00:00"
        self.temporary_media_filename: Path = None
        self.temporary_log_filename: Path = None
        self.list_filename: Path = None

        self.final_fps: int = None
        self.final_tags: dict | None = None
        self.final_tags_str: str = None
        self.final_timestamp_tags_str: str = None
        self.final_start_time: float | None = None
        self.final_end_time: float | None = None
        self.final_media_filename: Path = None
        self.final_log_filename: Path = None
        self.final_metadata_filename: Path = None
        self.final_timestamp_name_tags: str = None
        self.final_motion: dict | None = None

        self.fps: RollingAverage = RollingAverage()

    def should_add_frame(self):
        """Determine if the current frame should be added to the recording queue."""
        return True

    def add_frame(self, frame):
        """Call this inside your main loop for every single incoming frame."""
        now = time.time()
        self.frame_count += 1
        if now - self.window_start >= 1.0:
            self.fps.update(self.frame_count / (now - self.window_start))
            self.frame_count = 0
            self.window_start = now

        with self.lock:
            # Always make a copy to prevent OpenCV reference overwrites
            copied_frame = frame.copy()

            # Keep updating the rolling pre-buffer history
            # Convert to microseconds for better precision in ffmpeg timestamps
            now_ns = time.monotonic_ns() // 1000
            self.rolling_buffer.append((now_ns, copied_frame))

            # If actively recording, append subsequent frames to the stream
            if self.camera.recording_state.recording:
                self.record_queue.append((now_ns, copied_frame))


    def can_start(self):
        """Check if there are enough frames in the rolling buffer to start recording."""
        with self.lock:
            if len(self.rolling_buffer) == 0:
                return False
        return True

    def start_recording(self):
        """ gather metadata and setup for recording, then spawn the recording thread """
        if self.stop_event.is_set():
            return

        if not self.can_start():
            return

        start: float = self.camera.recording_state.recording_start_time
        self.final_start_time: float = start - self.recorder_config.pre_duration
        timestamp_str: str = make_ts_string(self.final_start_time)
        self.final_fps: int = self.fps.as_int()
        self.final_tags: dict = deepcopy(self.camera.motion.active_objects_dict)
        self.final_motion: dict = self.camera.motion.to_dict()
        self.final_tags_str: str = tags_to_str(self.final_tags)
        self.final_timestamp_tags_str: str = timestamp_str + \
            "_" + self.camera.camera_info.name + "_" + self.final_tags_str
        self.final_media_filename = Path(
            self.camera.camera_info.recordings_dir,
            self.final_timestamp_tags_str + ".mp4")
        self.final_metadata_filename = Path(
            self.camera.camera_info.metadata_dir,
            self.final_timestamp_tags_str + ".json")
        self.final_log_filename = Path(
            self.camera.camera_info.logs_dir,
            self.final_timestamp_tags_str + ".log")

        #pylint: disable=consider-using-with
        self.temporary_media_filename: Path  = Path(tempfile.NamedTemporaryFile(
            "w+b",
            dir=self.camera.camera_info.recordings_dir,
            suffix=".mp4",
            delete=False).name)

        self.temporary_log_filename: Path = self.temporary_media_filename.with_suffix(".log")

        self.seed_buffer()
        self.start_thread()

    def seed_buffer(self):
        """Seed the recording queue with frames from the rolling buffer."""
        fps = self.fps.as_int() if self.fps.as_int() > 0 else 20
        frames = fps * self.recorder_config.pre_duration
        self.record_queue = deque(list(self.rolling_buffer)[-frames:])

    def start_thread(self):
        """Start the background thread that will write frames to disk."""
        self.thread = threading.Thread(
            target=self._async_writer_worker,
            daemon=False,  # Not daemon because we want to guarantee it finishes writing
        )
        self.thread.start()
        logger.debug("FrameRecorder started")

    def stop_recording(self):
        """Signals the recording to stop. Main thread can keep running."""
        self.event.set()

        self.final_end_time = time.time()

        # We join the thread to guarantee the file is written and safe on disk
        if self.thread:
            self.thread.join()

        self._finalize_files()

    def _get_additional_metadata(self):
        return {}

    def _finalize_files(self):
        self._finalize_media_file()
        self._finalize_log_file()

        cap = cv2.VideoCapture(self.final_media_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = int(self.fps.as_int())

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        self.duration_seconds = 0.0
        if frame_count and fps:
            self.duration_seconds = frame_count / fps

        self.formatted_duration = str(timedelta(seconds=int(self.duration_seconds)))

        metadata = self._create_metadata()

        pre = self.recorder_config.pre_duration
        post = self.recorder_config.post_duration
        if self.duration_seconds < (pre + post) * 5 / 6:
            logger.info(f"auto-deleted {self.duration_seconds:.2f}s {self.final_media_filename}")
            os.remove(self.final_media_filename)
        else:
            with open(self.final_metadata_filename, "w", encoding="utf-8") as f:
                json.dump(metadata, f, default=lambda o: o.__dict__, indent=4)

            if self.duration_seconds <= 0.0:
                logger.error(f"recording broken {self.formatted_duration}")

            else:
                self.report_complete()
                self.add_recording_callback(self.final_metadata_filename)

    def report_complete(self):
        """Override this method in subclasses to report that the recording is complete."""
        logger.info(
            self.name + " " +
            self.camera.camera_info.name +
            " recording available "
            + self.formatted_duration,
            extra = {
                "recording": True,
                "file_path": self.final_metadata_filename
            })

    def _finalize_media_file(self):
        shutil.move(self.temporary_media_filename, self.final_media_filename)

    def _finalize_log_file(self):
        shutil.move(self.temporary_log_filename, self.final_log_filename)

    def _create_metadata(self):
        # Convert to a standard dict and sets to lists
        serializable_tags = {k: list(v) for k, v in self.final_tags.items()}

        json_data = {
            "camera": self.camera.camera_info.name,
            "fps": self.final_fps,
            "tags": serializable_tags,
            "media_filename": quote(str(self.final_media_filename)),
            "log_filename": quote(str(self.final_log_filename)),
            "start_time": self.final_start_time,
            "end_time": self.final_end_time,
            "duration": self.duration_seconds,
            "duration_fmt": self.formatted_duration,
            "start_fmt": make_readable_ts(self.final_start_time),
            "end_fmt": make_readable_ts(self.final_end_time),
            "metadata_filename": quote(str(self.final_metadata_filename)),
            "recorder_type": self.name,
            "motion": self.final_motion
        }
        return json_data | self._get_additional_metadata()


class OpenCVFrameRecorder(FrameRecorder):
    """A recorder that uses OpenCV to encode frames into a video file."""
    def __init__(
            self,
            camera: Camera,
            stop_event: Event,
            add_recording_callback: Callable,
            recorder_config: RecorderConfig
            ):
        self.name = "OpenCVFrame"
        super().__init__(
            camera=camera,
            stop_event=stop_event,
            add_recording_callback=add_recording_callback,
            recorder_config=recorder_config)

    def _async_writer_worker(self):
        """Background thread that continuously drains the queue and writes to disk."""
        current_thread().name = f" {self.camera.camera_info.name} {self.name}Recorder"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            self.temporary_media_filename,
            fourcc,
            self.fps.as_int(),
            (self.camera.resolution.width, self.camera.resolution.height)
            )

        try:
            while not self.stop_event.is_set():
                frame = None
                with self.lock:
                    # Check if there are frames ready to be written
                    if len(self.record_queue) > 0:
                        _, frame = self.record_queue.popleft()
                    # If queue is empty AND not recording, we are completely done
                    elif self.event.is_set() and len(self.record_queue) == 0:
                        break

                if frame is not None:
                    # High CPU overhead compression happens here safely on the background thread
                    video_writer.write(frame)
                else:
                    # Queue is momentarily empty, yield CPU execution to the main thread
                    time.sleep(0.01)
        finally:
            video_writer.release()

    @override
    def _finalize_media_file(self):
        """Moves the moov atom to the front of an MP4 file for browser streaming."""
        # Define the FFmpeg command as a list of strings
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            self.temporary_media_filename,
            "-c:v",
            "libx264",  # Explicitly convert the mp4v video to H.264
            "-c:a",
            "copy",  # Copy audio if present without changes
            "-movflags",
            "+faststart",  # Move metadata to the front for streaming
            self.final_media_filename,
        ]

        try:
            #pylint: disable=consider-using-with
            log_file = open(self.temporary_log_filename, "w", encoding="utf-8")
            # Launch the process safely using a context manager
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=log_file, text=True
            ) as process:
                # Wait for completion and capture outputs
                process.communicate()
                if process.returncode != 0:
                    logger.debug("ffmpeg Error in OpenCVFrameRecorder")

                os.remove(self.temporary_media_filename)

        except FileNotFoundError:
            logger.error("ffmpeg is not installed or not in your system PATH.")

        finally:
            log_file.close()

    @override
    def start_recording(self):
        if self.stop_event.is_set():
            return

        logger.debug(f"{self.camera.camera_info.name} cv2 FrameRecorder started")
        super().start_recording()  # This will set up the filename and log_filename


class AVFFmpegFrameRecorder(FrameRecorder):
    """A recorder that uses PyAV (FFmpeg bindings) to encode frames into a video file."""
    def __init__(
            self,
            camera: Camera,
            stop_event: Event,
            add_recording_callback: Callable,
            recorder_config: RecorderConfig
            ):
        self.name = "AVFFmpegFrame"
        super().__init__(
            camera=camera,
            stop_event=stop_event,
            add_recording_callback=add_recording_callback,
            recorder_config=recorder_config
        )

    def _async_writer_worker(self):
        camera_name = self.camera.camera_info.name
        current_thread().name = f" {camera_name} {self.name}Recorder"

        try:
            #pylint: disable=consider-using-with
            log_file = open(self.temporary_log_filename, "w", encoding="utf-8")
            log_file.write(
                f"{self.name}Recorder started {camera_name} at {make_readable_ts}\n")
            log_file.write(f"writing to {self.temporary_media_filename}\n")
            output = av.open(
                str(self.temporary_media_filename),
                mode="w",
                format="mp4",
                options={
                    "movflags": "faststart",
                }
            )
            stream = output.add_stream("libx264", rate=self.fps.as_int())
            log_file.write("libx264 stream added\n")
            stream.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
            }
            stream.width = self.camera.resolution.width
            stream.height = self.camera.resolution.height
            stream.pix_fmt = "yuv420p"
            stream.codec_context.max_b_frames = 0
            log_file.write(
                f"stream width: {stream.width} height: {stream.height} pix_fmt: yuv420p\n")
            frame_number = 0
            while not self.stop_event.is_set():
                frame = None
                with self.lock:
                    if len(self.record_queue) > 0:
                        _, frame = self.record_queue.popleft()
                    elif self.event.is_set() and len(self.record_queue) == 0:
                        break

                if frame is None:
                    time.sleep(0.01)
                    continue

                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                frame_number += 1
                packet_number = 0
                log_file.write(f"encoding frame to stream: {frame_number}\n")
                for packet in stream.encode(video_frame):
                    packet_number += 1
                    log_file.write(
                        f"sending frame {frame_number}, packet {packet_number} to muxer\n"
                        )
                    output.mux(packet)

            log_file.write("flushing stream\n")
            packet_number = 0
            for packet in stream.encode():
                log_file.write(
                    f"flushing frame {frame_number}, packet {packet_number} to muxer\n"
                )
                output.mux(packet)

            output.close()
            log_file.close()

        except Exception as e:
            logger.error(f"AVError in AVFFmpegFrameRecorder: {e}")
            traceback.print_exc()


    @override
    def start_recording(self):
        if self.stop_event.is_set():
            return

        logger.debug(f"{self.camera.camera_info.name} {self.name} FrameRecorder started")
        super().start_recording()  # This will set up the filename and log_filename


class FFmpegFrameRecorder(FrameRecorder):
    """A recorder that uses FFmpeg to encode frames into a video file."""
    def __init__(
            self,
            camera: Camera,
            stop_event: Event,
            add_recording_callback: Callable,
            recorder_config: RecorderConfig
            ):
        self.name = "FFmpegFrame"
        super().__init__(
            camera=camera,
            stop_event=stop_event,
            add_recording_callback=add_recording_callback,
            recorder_config=recorder_config
        )

    def _async_writer_worker(self):
        """Background thread that continuously drains the queue and writes to disk."""
        current_thread().name = f" {self.camera.camera_info.name} {self.name}Recorder"

        command = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",       # Matches OpenCV format
            "-s", f"{self.camera.resolution.width}x{self.camera.resolution.height}",
            "-r", f"{self.fps.as_int()}",            # Framerate
            "-i", "-",                 # Input from Python pipe

            # --- Compression & Compatibility Settings ---
            "-c:v", "libx264",         # H.264 codec (universal browser support)
            "-pix_fmt", "yuv420p",     # Crucial for browser playback
            "-crf", "23",              # Balanced quality vs file size (lower = higher quality)
            "-preset", "medium",       # Balanced encoding speed vs file size

            # --- Browser Optimization ---
            "-movflags", "+faststart", # Puts index at front for instant browser playback
            self.temporary_media_filename
        ]

        try:
            #pylint: disable=consider-using-with
            log_file = open(self.temporary_log_filename, "w", encoding="utf-8")

            # Run the process
            #pylint: disable=consider-using-with
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=log_file,
                )

            first_pts_μs = None
            frame_number = 0
            delta_μs = 0
            prev_pts_μs = 0
            while not self.stop_event.is_set():
                frame = None
                with self.lock:
                    if len(self.record_queue) > 0:
                        timestamp_μs, frame = self.record_queue.popleft()
                    elif self.event.is_set() and len(self.record_queue) == 0:
                        break

                if frame is None:
                    time.sleep(0.01)
                    continue

                if first_pts_μs is None:
                    first_pts_μs = timestamp_μs

                timestamp_μs -= first_pts_μs
                delta_μs = timestamp_μs - prev_pts_μs
                prev_pts_μs = timestamp_μs

                sleep_time = (delta_μs / 1_000_000) * 0.9
                if sleep_time > 0:
                    time.sleep(sleep_time)
                process.stdin.write(frame.tobytes())
                frame_number += 1

        except Exception as e:
            logger.error(f"Error in FFmpegFrameRecorder: {e}")

        finally:
            log_file.close()
            process.stdin.close()
            process.wait()

    @override
    def start_recording(self):
        if self.stop_event.is_set():
            return

        logger.debug(f"{self.camera.camera_info.name} {self.name}Recorder started")
        super().start_recording()  # This will set up the filename and log_filename


class FFmpegSegmentRecorder(FrameRecorder):
    """
    A recorder that captures pre-existing FFmpeg segments
    from disk and merges them into a single MP4.
    """
    def __init__(
            self,
            camera: Camera,
            stop_event: Event,
            add_recording_callback: Callable,
            recorder_config: RecorderConfig
            ):
        self.name = "FFmpegSegment"
        super().__init__(
            camera=camera,
            stop_event=stop_event,
            add_recording_callback=add_recording_callback,
            recorder_config=recorder_config
        )
        self.segments: list[str] = []

    @override
    def should_add_frame(self):
        # Segment recorder doesn't use individual frames
        # it uses ffmpeg segments directly from disk
        return False

    @override
    def can_start(self):
        # Segment recorder can always "start" (it just arms the worker)
        return True

    @override
    def seed_buffer(self):
        pass

    @override
    def start_recording(self):
        """
        For segment recording, 'start' just means:
        - start the merge worker thread (it will wait on self.event)
        """
        if self.stop_event.is_set():
            return

        camera_name = self.camera.camera_info.name
        recorder_name = self.name
        logger.debug(f"{camera_name} {recorder_name}Recorder started")
        super().start_recording()  # This will set up the filename and log_filename

    @override
    def stop_recording(self):
        """
        Capture everything needed to finalize this recording,
        without ever reading camera state again later.
        """
        current_thread().name = f"{self.camera.camera_info.name} {self.name}Recorder"

        self.final_end_time = time.time()

        # Resolve segments for this event window
        self.segments = self._get_segments(
            start_time=self.final_start_time,
            end_time=self.final_end_time,
        )

        camera_name = self.camera.camera_info.name
        recorder_name = self.name
        logger.debug(
            f"{camera_name} {recorder_name} captured {len(self.segments)} segments for recording")
        super().stop_recording()  # This will signal the worker to start merging


    def _async_writer_worker(self):
        """
        Runs ffmpeg merge in a separate thread.
        When the process finishes, write metadata and finalize files.
        """
        # Wait until stop_recording() has populated segments + snapshots
        self.event.wait()
        self.event.clear()

        self.list_filename: Path = self.final_media_filename.with_suffix(".list")

        if not self.segments:
            # Nothing to merge → nothing to record
            return

        # Protect these segments from cleanup while we merge
        FileCleaner.do_not_delete_set.update(self.segments)

        # Wait until the last segment in the window is fully written
        last_segment = self.segments[-1]
        self._wait_for_last_segment_to_finish(last_segment)

        # Build concat list file
        with open(self.list_filename, "w", encoding="utf-8") as f:
            for segment_file in self.segments:
                try:
                    stat_entry = os.stat(segment_file)
                    if stat_entry.st_size > 0:
                        f.write(f"file '{os.path.abspath(segment_file)}'\n")
                except FileNotFoundError:
                    continue

        # Run ffmpeg concat to merge segments into a single MP4
        # with the moov atom at the front for streaming

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-fflags", "+genpts",
            "-f", "concat",
            "-safe", "0",
            "-i", self.list_filename,
            "-c", "copy",
            "-movflags", "+faststart",

            self.temporary_media_filename,
        ]

        try:
            #pylint: disable=consider-using-with
            log_file = open(self.temporary_log_filename, "w", encoding="utf-8")
            logger.debug(f"{self.camera.camera_info.name} merging {len(self.segments)} segments")
            #pylint: disable=consider-using-with
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=log_file,
            )
            process.communicate()
        finally:
            log_file.close()
            FileCleaner.do_not_delete_set -= set(self.segments)

    @override
    def _get_additional_metadata(self):
        # Use the segments captured for this recording
        return {
            "segments": self.segments,
            "list_filename": self.list_filename,
            }

    def _get_segments(self, start_time: float, end_time: float) -> list[str]:
        """
        Return all .ts segments whose timestamp falls within [start_time, end_time].
        """
        selected: list[tuple[str, float]] = []
        for f in os.scandir(self.camera.camera_info.segments_dir):
            if f.name.endswith(".ts"):
                try:
                    stat_entry = f.stat()
                    if start_time <= stat_entry.st_mtime <= end_time:
                        selected.append(
                            (
                                os.path.join(
                                    self.camera.camera_info.segments_dir,
                                    f.name),
                                stat_entry.st_mtime
                            )
                        )
                except Exception:
                    pass
        selected.sort(key=lambda x: x[1])
        return [f[0] for f in selected]

    def _wait_for_last_segment_to_finish(
        self,
        last_segment: str,
        timeout: float = 5.0,
        stable_time: float = 3.0,
    ):
        """
        Wait until the last segment *within the event window* is fully written.
        """
        deadline = time.time() + timeout
        last_size = -1
        stable_since = None

        while time.time() < deadline:
            try:
                size = os.path.getsize(last_segment)
            except FileNotFoundError:
                time.sleep(0.05)
                continue

            if size != last_size:
                last_size = size
                stable_since = time.time()
            else:
                if stable_since is not None and time.time() - stable_since >= stable_time:
                    return

            time.sleep(0.05)

        # Timeout → proceed anyway
        return
