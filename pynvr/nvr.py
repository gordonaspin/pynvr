"""
NVR is the controlling coordinator of readers and processors
"""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from logging import getLogger
from pathlib import Path
from threading import Event

from pynvr.camera.camera import Camera
from pynvr.config.config import SystemConfig, CameraConfig, Resolution
from pynvr.constants import TS_FILE_RING_SECONDS
from pynvr.file_cleaner import FileCleaner
from pynvr.processor import FrameProcessor
from pynvr.reader import Reader, FrameReader
from pynvr.recorder import FrameRecorderFactory
from pynvr.thread_safe import ThreadSafeList
from pynvr.utils import get_camera_resolution
from pynvr.api.types import RecordingEvent

logger = getLogger("pynvr")

# =========================
# NVR ENGINE
# =========================
class NVR:
    """ Class representing the control center for the NVR """
    def __init__(self, system_config: SystemConfig):

        self.stop_event: Event = Event()

        self.cameras: dict[str, Camera] = {}
        self.frame_readers: dict[str, Reader] = {}
        self.processors: dict[str, FrameProcessor] = {}
        self.recordings: ThreadSafeList = ThreadSafeList()

        camera_resolutions = self.get_camera_resolutions(camera_configs=system_config.cameras)

        for name, camera_config in system_config.cameras.items():
            if not camera_config.enabled:
                logger.info(f"{name} camera is disabled")
                continue
            camera_resolution = camera_resolutions[name]
            if not camera_resolution.is_valid():
                camera_resolution = Resolution(
                    width=camera_config.resolution.width,
                    height=camera_config.resolution.height)
                logger.warning(
                    f"{name} could not get resolution, " +
                    "falling back to configured resolution " +
                    f"{camera_resolution.width}x{camera_resolution.height}")

            camera = self.cameras[name] = Camera(
                name=name,
                camera_config=camera_config,
                camera_resolution=camera_resolution,
                model_resolution=system_config.model.resolution,
                logs_dir=system_config.logs_directory,
                recordings_dir=system_config.recordings_directory,
            )

            reader = self.frame_readers[name] = FrameReader(
                camera=camera,
                model_config=system_config.model,
                produce_segments=camera_config.recorder == "FFmpegSegment",
                stop_event=self.stop_event)

            self.processors[name] = FrameProcessor(
                processor_config=system_config.processor,
                camera=camera,
                reader=reader,
                recorder=FrameRecorderFactory.create(
                    camera=camera,
                    recorder_name=camera_config.recorder,
                    stop_event=self.stop_event,
                    add_recording_callback=self.add_recording,
                    recorder_config=system_config.processor.recorder,
                ),
                model_config=system_config.model,
                stop_event=self.stop_event,
                )
        FileCleaner.stop_event = self.stop_event
        FileCleaner.add(
            system_config.recordings_directory,
            "*.mp4",
            system_config.keep_recordings_timedelta,
            timedelta(minutes=5))
        FileCleaner.add(
            system_config.recordings_directory,
            "*.jpg",
            system_config.keep_recordings_timedelta,
            timedelta(minutes=5))
        FileCleaner.add(
            system_config.recordings_directory,
            "*.json",
            system_config.keep_recordings_timedelta,
            timedelta(minutes=5))
        FileCleaner.add(
            system_config.recordings_directory,
            "*.log",
            system_config.keep_logs_timedelta,
            timedelta(minutes=5))
        FileCleaner.add(
            system_config.recordings_directory,
            "*.ts",
            timedelta(seconds=TS_FILE_RING_SECONDS),
            timedelta(seconds=5))
        FileCleaner.add(
            system_config.recordings_directory,
            "*.list",
            timedelta(seconds=TS_FILE_RING_SECONDS),
            timedelta(seconds=5))

        FileCleaner.add(
            system_config.logs_directory,
            "*.log",
            system_config.keep_logs_timedelta,
            timedelta(minutes=5))

    def start(self):
        """
        Start the NVR processes. Threads created are:
        1 ffmpeg reader thread for each camera, writing to segment files and stdout
        1 ffmpeg frame reader thread for each camera reading
            from stdout and writing frames to a queue
        1 frame processor thread to read frames from the queue and do image processing
        """
        self._load_events()

        if not self.stop_event.is_set():
            for camera in self.cameras.values():
                if camera.camera_info.enabled:
                    self.frame_readers[camera.camera_info.name].start()
                    self.processors[camera.camera_info.name].start()


    def stop(self):
        """
        Stop the NVR
        """
        logger.info("stopping NVR processors")
        for processor in self.processors.values():
            processor.stop()
        logger.info("stopping NVR readers")
        for reader in self.frame_readers.values():
            reader.stop()


    def threads(self):
        """ return array of threads owned by the NVR """
        threads = []
        for camera in self.cameras.values():
            name = camera.camera_info.name
            if self.frame_readers[name].thread is not None:
                threads.append(self.frame_readers[name].thread)
            if self.processors[name].thread is not None:
                threads.append(self.processors[name].thread)
            if self.processors[name].recorder.thread is not None:
                threads.append(self.processors[name].recorder.thread)
        if FileCleaner.thread is not None:
            threads.append(FileCleaner.thread)

        return threads

    def add_recording(self, metadata_file: Path):
        """ called by recorders to add a new recording """
        self.recordings.append(self._load_event(metadata_file=metadata_file))

    def _load_event(self, metadata_file: Path) -> RecordingEvent:
        recording_event = None

        with open(metadata_file, "r", encoding="utf-8") as fp:
            try:
                event = json.load(fp)
                recording_event = RecordingEvent(
                    camera=event["camera"],
                    tags=event["tags"],
                    media_filename=event["media_filename"],
                    start_time=event["start_time"],
                    end_time=event["end_time"],
                    start_fmt=event["start_fmt"],
                    end_fmt=event["end_fmt"],
                    metadata_filename=event["metadata_filename"],
                    recorder_type=event["recorder_type"])

            except json.JSONDecodeError:
                logger.warning(f"invalid JSON in file {metadata_file}, deleting the file")
                os.remove(metadata_file)
        return recording_event

    def _load_events(self):
        events = []

        start = time.time()
        for camera in self.cameras.values():
            if camera.camera_info.enabled:
                for f in glob.glob(f"{camera.camera_info.metadata_dir}/*.json"):
                    try:
                        event = self._load_event(Path(f))
                        if event:
                            events.append(event)

                    except FileNotFoundError:
                        pass # it's possible a clean-up job whacked the file

        # Sort globally by start_time
        events.sort(key=lambda e: e.start_time)
        self.recordings.extend(events)

        logger.debug(f"loaded {len(events)} events in {(time.time() - start):.2f} seconds")


    def get_camera_resolutions(
            self,
            camera_configs: dict[str, CameraConfig]
            ) -> dict[str, Resolution]:
        """ return dictionary of camera resolutions """
        results = {}

        def task(name, url):
            w, h = get_camera_resolution(url)
            return name, Resolution(width=w, height=h)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(task, name, camera_config.url)
                for name, camera_config in camera_configs.items() if camera_config.enabled
            ]

            for f in as_completed(futures):
                name, res = f.result()
                logger.info(f"{name} camera resolution detected as {res.width}x{res.height}")
                results[name] = res

        return results
