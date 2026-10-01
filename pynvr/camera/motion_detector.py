"""
MotionDetector uses ByteTrack to track YOLO detections and compute per-track velocity.
It maintains a has_moving_object state based on minimum_track_speed_pxpf and linger time.
"""
import time
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from threading import Lock

import numpy as np

from pynvr.byte_track.byte_tracker import BYTETracker
from pynvr.api.types import ConfigValue
from pynvr.config.config import CameraConfig

logger = getLogger("pynvr")

@dataclass
class MotionTrack:
    """
    Represents a single tracked object with its properties.
    """
    track_id: int
    cls: int
    conf: float
    tlbr: tuple[int, int, int, int]
    speed: float
    relative_speed: float


class MotionDetector:
    """
    ByteTrack-only motion detector.
    Responsibilities:
    - Run BYTETracker on YOLO detections
    - Compute per-track velocity
    - Decide has_moving_object based on minimum_track_speed_pxpf and linger time
    - Maintain classes_in_frame_dict and active_objects_dict
    - Maintain last_motion_time
    - Expose active_tracks for debug UI
    """

    def __init__(self, camera_config: CameraConfig, name: str):
        self.name = name
        self.track_threshold: ConfigValue = ConfigValue.from_config(
            default=camera_config.yolo_confidence,
            model_cls=CameraConfig,
            model_field_name="track_threshold"
        )

        self.match_threshold: ConfigValue = ConfigValue.from_config(
            default=camera_config.match_threshold,
            model_cls=CameraConfig,
            model_field_name="match_threshold"
        )

        self.track_buffer: ConfigValue = ConfigValue.from_config(
            default=camera_config.track_buffer,
            model_cls=CameraConfig,
            model_field_name="track_buffer"
        )

        self.minimum_relative_motion: ConfigValue = ConfigValue.from_config(
            default=camera_config.minimum_relative_motion,
            model_cls=CameraConfig,
            model_field_name="minimum_relative_motion"
        )

        self.lock: Lock = Lock()

        # --- ByteTrack configuration ---
        self.tracker = self.create_tracker()

        # --- Motion state ---
        self.has_moving_object: bool = False
        self.last_motion_time: float = time.time()

        # YOLO class + color metadata
        self.classes_in_frame_dict = defaultdict(set)
        self.active_objects_dict = defaultdict(set)

        # Track history for velocity computation
        self.last_positions: dict[int, tuple[float, float]] = {}

        # Active ByteTrack objects for debug UI
        self.active_tracks: list[MotionTrack] = []
        self.moving_track_ids: set[int] = set()
        self.smoothed_speeds = {}  # tid → smoothed relative speed

    def create_tracker(self):
        """Recreate the BYTETracker with updated config values."""
        with self.lock:
            tracker = BYTETracker(
                track_thresh=self.track_threshold.value,
                match_thresh=self.match_threshold.value,
                track_buffer=self.track_buffer.value,
            )
        self.tracker = tracker
        return tracker

    # ----------------------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------------------
    def update(self, yolo_dets: np.ndarray, now: float, is_night: bool):
        """
        Update motion state using YOLO detections and ByteTrack.
        yolo_dets: Nx6 array [x1, y1, x2, y2, score, cls]
        """

        # Run ByteTrack
        with self.lock:
            online_targets = self.tracker.update(yolo_dets)

        moving = False
        active_tracks = []
        self.moving_track_ids.clear()

        # --- Night-aware parameters ---
        # 1. Motion threshold (higher at night to reduce jitter false positives)
        base_threshold = self.minimum_relative_motion.value
        threshold = base_threshold * (1.6 if is_night else 1.0)

        # 2. Smoothing factor (heavier smoothing at night)
        alpha = 0.25 if not is_night else 0.15

        # 3. Track persistence (longer at night)
        linger_time = 3.0 if not is_night else 6.0

        for t in online_targets:
            tid = t.track_id
            x1, y1, x2, y2 = t.tlbr
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            size = max(w, h)

            speed = 0.0
            relative_speed = 0.0
            smooth = 0.0

            if tid in self.last_positions:
                px, py = self.last_positions[tid]
                dx = cx - px
                dy = cy - py

                speed = float((dx * dx + dy * dy) ** 0.5)
                relative_speed = float(speed / size if size > 0 else 0)

                prev = self.smoothed_speeds.get(tid, 0.0)
                smooth = float(alpha * relative_speed + (1 - alpha) * prev)
                self.smoothed_speeds[tid] = smooth

                # Night-aware motion threshold
                if smooth > threshold:
                    moving = True
                    self.moving_track_ids.add(tid)

            self.last_positions[tid] = (cx, cy)

            active_tracks.append(MotionTrack(
                track_id=tid,
                cls=int(t.cls),
                conf=float(t.score),
                tlbr=(int(x1), int(y1), int(x2), int(y2)),
                speed=speed,
                relative_speed=smooth
            ))

        self.active_tracks = active_tracks

        # Night-aware persistence
        if moving:
            self.last_motion_time = now
            self.has_moving_object = True
        else:
            self.has_moving_object = (now - self.last_motion_time) < linger_time


    # ----------------------------------------------------------------------
    # CLASS + COLOR METADATA
    # ----------------------------------------------------------------------
    def clear_frame_classes(self):
        """Call this at the start of each frame before adding YOLO class/color info."""
        self.classes_in_frame_dict.clear()

    def add_class_color(self, class_name: str, color: str):
        """Called by FrameProcessor after color detection."""
        self.classes_in_frame_dict[class_name].add(color)

    def finalize_active_objects(self):
        """
        Merge classes_in_frame_dict into active_objects_dict while recording.
        Called by FrameProcessor during recording.
        """
        for cls, colors in self.classes_in_frame_dict.items():
            self.active_objects_dict[cls].update(colors)

    def reset_active_objects(self):
        """Called when recording stops."""
        self.active_objects_dict.clear()

    def get_active_objects(self):
        """Return a dict of active objects for the current recording."""
        return {k: list(v) for k, v in self.active_objects_dict.items()}

    # ----------------------------------------------------------------------
    # DEBUG / INSPECTION
    # ----------------------------------------------------------------------
    def to_dict(self):
        """Minimal state for debug UI or API."""
        return {
            "has_moving_object": self.has_moving_object,
            "last_motion_time": self.last_motion_time,
            "active_tracks": [
                vars(t)
                for t in self.active_tracks
            ],
            "classes_in_frame": {k: list(v) for k, v in self.classes_in_frame_dict.items()},
            "active_objects": {k: list(v) for k, v in self.active_objects_dict.items()},
        }
