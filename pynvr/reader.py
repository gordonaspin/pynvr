"""
Frame reader for handling video stream processing.
"""
import os
import subprocess
import select
import time
from logging import getLogger
from pathlib import Path
from queue import Queue, Empty
from threading import Event, Thread, current_thread
from typing import override

import cv2
import numpy as np
from numpy.typing import NDArray

from pynvr.camera.camera import Camera
from pynvr.config.config import ModelConfig, Resolution
from pynvr.utils import RollingAverage

logger = getLogger("pynvr.reader")

class Reader:
    """ Reader ABC """
    def __init__(self):
        raise NotImplementedError()

    def start(self):
        """ start """
        raise NotImplementedError()

    def stop(self):
        """ stop """
        raise NotImplementedError()

    def get_frame(self):
        """ get frame """
        raise NotImplementedError()


class FrameReader(Reader):
    """
    Frame reader that handles video stream processing, including reading frames,
    detecting corrupted frames, and managing FFmpeg subprocesses.
    """
    def __init__(
        self,
        camera: Camera,
        model_config: ModelConfig,
        produce_segments: bool,
        stop_event: Event,
    ):
        self.camera: Camera = camera
        self.resolution: Resolution = model_config.resolution
        self.produce_segments = produce_segments
        self.stop_event: Event = stop_event

        self.process: subprocess.Popen | None = None
        self.yolo_pipe: None | object = None  # file object for YOLO pipe
        self.yolo_read_fd: int | None = None
        self.yolo_write_fd: int | None = None
        self.log_filename: Path | None = None
        self.log_file: object | None = None

        self.thread: Thread | None = None
        self.frame_queue: Queue[NDArray[np.uint8]] = Queue(maxsize=1)

        self.total_frames: int = 0
        self.total_drops: int = 0
        self.last_frame_time: float = 0.0
        self.fps: RollingAverage = RollingAverage()
        self.dt: RollingAverage = RollingAverage()

        # For corruption detection
        self._prev_mean: float | None = None
        self._prev_frame: NDArray[np.uint8] | None = None
        self._diff_buf_row: NDArray[np.uint8] | None = None

    @override
    def start(self):
        """
        Start autonomous reader thread.
        """
        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """
        Autonomous lifecycle:
        - open stream
        - read frames
        - on stall/error → stop and reopen
        - loop until stop_event is set
        """
        current_thread().name = f"{self.camera.camera_info.name}FrameReader"

        while not self.stop_event.is_set():
            try:
                self._open_stream()
            except Exception as e:
                logger.error(f"{self.camera.camera_info.name} failed to open stream: {e}")
                self._cleanup_process()
                time.sleep(30.0)
                continue

            self._frame_reader_loop()
            self._cleanup_process()

            if not self.stop_event.is_set():
                time.sleep(30.0)

        logger.info(f"{self.camera.camera_info.name} FrameReader main loop exiting")

    #pylint: disable=too-many-branches
    def _cleanup_process(self):
        """
        Internal cleanup: terminate FFmpeg, close pipes/logs, reset state.
        """
        # Process and stdout
        if self.process is not None:
            ret = self.process.poll()
            logger.info(f"{self.camera.camera_info.name} stopping FrameReader with ret {ret}")
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass

            if self.process.stdout is not None:
                try:
                    self.process.stdout.close()
                except Exception:
                    pass

        # YOLO file object
        if self.yolo_pipe is not None:
            try:
                self.yolo_pipe.close()
            except Exception:
                pass

        # Raw YOLO FDs (in case fdopen/close never happened)
        if self.yolo_read_fd is not None:
            try:
                os.close(self.yolo_read_fd)
            except Exception:
                pass

        if self.yolo_write_fd is not None:
            try:
                os.close(self.yolo_write_fd)
            except Exception:
                pass

        # Log file
        if self.log_file is not None:
            try:
                self.log_file.close()
            except Exception:
                pass

        self.process = None
        self.yolo_pipe = None
        self.yolo_read_fd = None
        self.yolo_write_fd = None
        self.log_file = None

    @override
    def stop(self):
        """
        Public stop: signal thread and clean up process.
        """
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._cleanup_process()

    @override
    def get_frame(self) -> NDArray[np.uint8] | None:
        """
        Retrieve the latest frame from the camera queue.
        """
        try:
            frame: NDArray[np.uint8] = self.frame_queue.get(timeout=0.5)
            return frame.copy()
        except Empty:
            return None

    def _open_stream(self):
        if self.stop_event.is_set():
            return

        need_yolo_pipe = self._needs_yolo_pipe()

        filespec = (
            Path(self.camera.camera_info.segments_dir, "%Y%m%d_%H%M%S.ts")
            if self.produce_segments
            else None
        )

        self.yolo_read_fd = None
        self.yolo_write_fd = None

        if need_yolo_pipe:
            self.yolo_read_fd, self.yolo_write_fd = self._make_yolo_pipe()

        ffmpeg_cmd = self.build_ffmpeg_cmd(
            url=self.camera.camera_info.url,
            filespec=filespec,
            segment_mode=self.produce_segments,
            yolo_fd=self.yolo_write_fd,
        )

        self.log_filename = Path(
            self.camera.camera_info.logs_dir,
            f"{self.camera.camera_info.name}_ffmpeg.log",
        )
        #pylint: disable=consider-using-with
        self.log_file = open(self.log_filename, "a", encoding="utf-8", buffering=1)
        for item in ffmpeg_cmd:
            if item.startswith("-"):
                self.log_file.write("\n")
            self.log_file.write(f"{item} ")
        self.log_file.write(
            f"\n--- Starting FFmpeg for {self.camera.camera_info.name} "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        )
        self.log_file.flush()

        try:
            if need_yolo_pipe:
                logger.info(f"{self.camera.camera_info.name} starting dual-pipe FrameReader")

                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=self.log_file,
                    pass_fds=(self.yolo_write_fd,),
                    bufsize=0,
                )

                # Parent does not write to YOLO pipe
                if self.yolo_write_fd is not None:
                    os.close(self.yolo_write_fd)
                    self.yolo_write_fd = None

                # Wrap read-end
                self.yolo_pipe = os.fdopen(self.yolo_read_fd, "rb", buffering=0)
            else:
                logger.info(f"{self.camera.camera_info.name} starting single-pipe FrameReader")

                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=self.log_file,
                    bufsize=0,
                )

                self.yolo_pipe = None
        except Exception:
            # On failure, ensure raw FDs are closed
            if self.yolo_read_fd is not None:
                try:
                    os.close(self.yolo_read_fd)
                except Exception:
                    pass
                self.yolo_read_fd = None
            if self.yolo_write_fd is not None:
                try:
                    os.close(self.yolo_write_fd)
                except Exception:
                    pass
                self.yolo_write_fd = None
            raise

        self.process = process

    #pylint: disable=too-many-statements
    def _frame_reader_loop(self):
        """
        Reads frames from FFmpeg until stall/EOF/stop_event.
        """
        if self.process is None or self.process.stdout is None:
            return

        fail_count = 0

        while (
            not self.stop_event.is_set()
            and self.process is not None
            and self.process.stdout is not None
            and not self.process.stdout.closed
        ):
            # 1. Full-res frame
            raw_full = self._read_exact(
                self.process.stdout.fileno(),
                self.camera.buffers.read_view_full,
                self.camera.buffers.full_frame_size,
            )

            if raw_full is None:
                fail_count += 1

                if self.process.poll() is not None:
                    logger.warning(
                        self.camera.camera_info.name +
                        " ffmpeg exited, breaking reader loop")
                    break

                if fail_count >= 10:
                    logger.warning(
                        self.camera.camera_info.name +
                        " full-res reader stalled, restarting after repeated timeouts")
                    break

                continue

            fail_count = 0

            self.camera.buffers.full_frame_bytes[:] = raw_full
            full_frame = self.camera.buffers.full_frame

            if self._looks_corrupted(full_frame):
                continue

            # 2. YOLO frame
            if (
                self.yolo_pipe is not None
                and self.camera.buffers.read_view_yolo is not None
            ):
                raw_yolo = self._read_exact(
                    self.yolo_pipe.fileno(),
                    self.camera.buffers.read_view_yolo,
                    self.camera.buffers.yolo_frame_size,
                )

                if raw_yolo is None:
                    fail_count += 1

                    if self.process.poll() is not None:
                        logger.warning(
                            self.camera.camera_info.name +
                            " ffmpeg exited, breaking YOLO reader loop"
                        )
                        break

                    if fail_count >= 10:
                        logger.warning(
                            self.camera.camera_info.name +
                            " yolo reader stalled, restarting after repeated timeouts"
                        )
                        break

                    continue

                    # if we continue, we skip this frame entirely
                fail_count = 0

                self.camera.buffers.yolo_frame_bytes[:] = raw_yolo
                self.camera.buffers.yolo_frame = self.camera.buffers.yolo_frame
            else:
                self.camera.buffers.yolo_frame = full_frame

            # 3. FPS
            now = time.perf_counter()
            if self.last_frame_time > 0:
                dt = now - self.last_frame_time
                if 0.02 < dt < 0.2:
                    self.dt.update(dt)
                    self.fps.update(1.0 / self.dt.value())
            self.last_frame_time = now

            # 4. Queue (latest-frame-wins)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.total_drops += 1
                except Exception:
                    pass

            self.frame_queue.put(full_frame)
            self.total_frames += 1

        logger.info(f"{self.camera.camera_info.name} FrameReader loop exiting")

    def _read_exact(self, fd, view, size, timeout=2.0):
        """
        Reads exactly `size` bytes into `view`.
        Returns view[:size] on success, None on timeout/EOF.
        """
        total = 0
        deadline = time.time() + timeout

        while total < size:
            if time.time() > deadline:
                return None

            r, _, _ = select.select([fd], [], [], timeout)
            if not r:
                return None

            try:
                chunk = os.read(fd, size - total)
            except OSError:
                return None

            if not chunk:
                return None

            n = len(chunk)
            view[total : total + n] = chunk
            total += n

        return view[:size]

    def _looks_corrupted(self, frame: NDArray[np.uint8]) -> bool:
        """
        Lightweight corruption detector to skip obviously bad frames.
        """
        camera_name = self.camera.camera_info.name
        mean = float(frame.mean())

        # All-black or all-white frames are suspicious
        if mean < 1.0 or mean > 254.0:
            logger.warning(
                f"{camera_name} mean={mean:.2f} - Detected corrupted frame, dropping"
            )
            return True

        # Exposure jump / I-frame jump
        if self._prev_mean is not None and abs(mean - self._prev_mean) > 80:
            logger.warning(
                f"{camera_name} exposure jump > 80 - Detected exposure jump, dropping"
            )
            self._prev_mean = mean
            self._prev_frame = frame.copy()
            return True

        if self._prev_frame is not None:
            diff = cv2.absdiff(frame, self._prev_frame)
            if diff.mean() > 120.0:
                logger.warning(
                    f"{camera_name} diff.mean={diff.mean():.2f}" +
                    f" diff.mean={diff.mean():.2f}" +
                    " - Detected corrupted frame, dropping"
                )
                self._prev_mean = mean
                self._prev_frame = frame.copy()
                return True

            if self._diff_buf_row is None:
                self._diff_buf_row = np.zeros((frame.shape[1], 3), dtype=np.uint8)

            cv2.absdiff(frame[0], self._prev_frame[0], dst=self._diff_buf_row)
            continuity_diff = self._diff_buf_row.mean()
            if continuity_diff > 150:
                logger.warning(
                    f"{camera_name} continuity_diff.mean={continuity_diff:.2f}" +
                    " - Detected corrupted frame, dropping"
                )
                self._prev_mean = mean
                self._prev_frame = frame.copy()
                return True

        self._prev_mean = mean
        self._prev_frame = frame.copy()
        return False

    #pylint: disable=too-many-statements
    def build_ffmpeg_cmd(
        self,
        url: str,
        filespec: Path | None,
        segment_mode: bool,
        yolo_fd: int | None,
    ) -> list[str]:
        """
        Build the FFmpeg command for reading the camera stream, optionally producing
        segments and a YOLO pipe.
        """

        actual_w = self.camera.resolution.width
        actual_h = self.camera.resolution.height
        yolo_w = self.resolution.width
        yolo_h = self.resolution.height

        need_yolo_pipe = not (actual_w == yolo_w and actual_h == yolo_h)

        if need_yolo_pipe and yolo_fd is None:
            raise ValueError("yolo_fd must be provided when YOLO pipe is required")

        full_block = actual_w * actual_h * 3
        yolo_block = yolo_w * yolo_h * 3

        base: list[str] = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-max_delay", "0",
            "-fflags", "nobuffer+genpts+discardcorrupt",
            "-flags", "low_delay",
            "-avioflags", "direct",
            "-i", url,
            "-hide_banner",
            "-loglevel", "error",
            "-nostats",
        ]

        filters: list[str] = []

        if need_yolo_pipe:
            filters.append("[0:v]split[full_in][yolo_in]")
            filters.append("[full_in]format=bgr24[full]")

            if actual_w == yolo_w and actual_h == yolo_h:
                filters.append("[yolo_in]format=bgr24[yolo]")
            else:
                filters.append(
                    "[yolo_in]"
                    f"scale={yolo_w}:-1:force_original_aspect_ratio=decrease,"
                    f"pad={yolo_w}:{yolo_h}:({yolo_w}-iw)/2:({yolo_h}-ih)/2:"
                    "color=#727272,"
                    "format=bgr24[yolo]"
                )
        else:
            filters.append("[0:v]format=bgr24[full]")

        filter_graph = ";".join(filters)

        cmd: list[str] = base + [
            "-filter_complex", filter_graph,
            "-map", "[full]",
            "-f", "rawvideo",
            "-blocksize", str(full_block),
            "pipe:1",
        ]

        if filespec is not None and segment_mode:
            cmd += [
                "-map", "0:v",
                "-c:v", "copy",
                "-f", "segment",
                "-segment_time", "1",
                "-reset_timestamps", "0",
                "-strftime", "1",
                "-segment_format", "mpegts",
                filespec,
            ]

        if need_yolo_pipe:
            cmd += [
                "-map", "[yolo]",
                "-f", "rawvideo",
                "-blocksize", str(yolo_block),
                f"pipe:{yolo_fd}",
            ]

        return cmd

    def _needs_yolo_pipe(self) -> bool:
        return not (
            self.camera.resolution.width == self.resolution.width
            and self.camera.resolution.height == self.resolution.height
        )

    def _make_yolo_pipe(self) -> tuple[int, int]:
        read_fd, write_fd = os.pipe()
        logger.debug(
            f"{self.camera.camera_info.name} created YOLO pipe "
            f"read fd={read_fd}, write fd={write_fd}"
        )
        return read_fd, write_fd
