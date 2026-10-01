"""
Camera representation and wiring for ByteTrack-only motion detection.
- Owns FrameBuffers (full + YOLO frame)
- Owns MotionDetector (ByteTrack + velocity-based motion)
- Owns RecordingState
- Tracks night/day state
- Holds latest frames for UI/debug
"""
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import numpy as np
from numpy.typing import NDArray

from pynvr.api.types import ConfigValue
from pynvr.camera.frame_buffers import FrameBuffers
from pynvr.camera.motion_detector import MotionDetector
from pynvr.config.config import CameraConfig, Resolution

@dataclass
class RecordingState:
    """
    Represents the recording state of a camera.
    """
    recording: bool = False
    recording_start_time: float = 0.0
    should_record: bool = False
    should_continue: bool = False

    # Used only by shadow filters, not recording logic
    white_ratio: float = 0.0


class CameraInfo:
    """
    Represents the configuration for a camera.
    """
    def __init__(self, camera_config: CameraConfig,
                 name: str,
                 logs_dir: Path,
                 recordings_dir: Path):

        self.resolution: Resolution = camera_config.resolution
        self.yolo_confidence: ConfigValue = ConfigValue.from_config(
            default=camera_config.yolo_confidence,
            model_cls=CameraConfig,
            model_field_name="yolo_confidence"
        )
        self.name: str = name
        self.max_pixels: int = self.resolution.width * self.resolution.height
        self.enabled: bool = camera_config.enabled
        self.debug: bool = camera_config.debug
        self.url: str = camera_config.url
        self.render_annotations: str = camera_config.render_annotations

        # Directories
        self.logs_dir: Path = logs_dir
        self.recordings_dir: Path = Path(recordings_dir, name)
        self.segments_dir: Path = Path(recordings_dir, "segments", name)
        self.images_dir: Path = Path(recordings_dir, "images", name)
        self.metadata_dir: Path = Path(recordings_dir, "metadata", name)
        self.plates_dir: Path = Path(recordings_dir, "plates", name)

        # Ensure dirs exist
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.plates_dir.mkdir(parents=True, exist_ok=True)


class Camera:
    """
    Camera wiring for ByteTrack-only motion:

    - Owns FrameBuffers (full + YOLO frame)
    - Owns MotionDetector (ByteTrack + velocity-based motion)
    - Owns RecordingState
    - Tracks night/day state
    - Holds latest frames for UI/debug
    """

    def __init__(
        self,
        name: str,
        camera_config: CameraConfig,
        camera_resolution: Resolution,
        model_resolution: Resolution,
        logs_dir: Path,
        recordings_dir: Path,
    ):
        self.resolution: Resolution = camera_resolution
        #self.width = camera_resolution.width
        #self.height = camera_resolution.height
        self.start_time = time.time()

        # Per-camera config (paths, resolution, flags)
        self.camera_info: CameraInfo = CameraInfo(camera_config, name, logs_dir, recordings_dir)

        # Frame buffers: full-res + optional YOLO-res
        self.buffers: FrameBuffers = FrameBuffers(camera_resolution, model_resolution)

        # ByteTrack-only motion detector
        # Expects camera-specific config with:
        #   track_thresh, match_thresh, track_buffer,
        #   minimum_track_speed, yolo_confidence
        self.motion: MotionDetector = MotionDetector(camera_config, name)

        # Recording state machine
        self.recording_state: RecordingState = RecordingState()

        # Debug flag
        self.debug: bool = camera_config.debug

        # Latest-frame-wins buffers for UI/debug
        self.latest_frame: NDArray[np.uint8] | None = None
        self.yolo_frame: NDArray[np.uint8] | None = None
        self.debug_motion_image: NDArray[np.uint8] | None = None

        # Night/day state (used for color detection, UI, optional YOLO tweaks)
        self.is_night: bool = False

        # Optional: event queue / control messages
        self.events: "Queue[dict]" = Queue()

    # ----------------------------------------------------------------------
    # Convenience hooks for FrameProcessor (optional)
    # ----------------------------------------------------------------------
    def update_latest_frame(self, frame_bgr: NDArray[np.uint8]):
        """Update the latest frame for UI/debug purposes."""
        self.latest_frame = frame_bgr

    def update_yolo_frame(self, frame_bgr: NDArray[np.uint8]):
        """Update the YOLO frame for UI/debug purposes."""
        self.yolo_frame = frame_bgr

    def update_debug_motion_image(self, frame_bgr: NDArray[np.uint8]):
        """Update the debug motion image for UI/debug purposes."""
        self.debug_motion_image = frame_bgr
