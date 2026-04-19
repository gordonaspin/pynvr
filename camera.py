import dataclasses
from subprocess import Popen
import numpy as np
from queue import Queue

from collections import deque

class RollingAverage:
    def __init__(self, window_size=100):
        self.window = deque(maxlen=window_size)
        self.sum = 0.0

    def update(self, value):
        # If full, remove oldest from sum
        if len(self.window) == self.window.maxlen:
            self.sum -= self.window[0]

        self.window.append(value)
        self.sum += value

        return self.sum / len(self.window)
    
    def value(self):
        if not self.window:
            return 0.0
        return self.sum / len(self.window)
    
@dataclasses.dataclass
class Camera:
    name: str
    url: str
    enabled: bool
    recordings_dir: str
    segments_dir: str
    images_dir: str

    # stream state
    process: Popen = None
    running = True

    # latest-frame-wins buffer
    latest_frame: np.ndarray = None
    frame_queue: Queue = dataclasses.field(default_factory=lambda: Queue(maxsize=1))

    # buffers for cv2 frames
    gray_buf = None
    diff_buf = None
    thresh_buf = None

    # FPS tracking (IMPORTANT FIX: per-instance)
    fps: RollingAverage = dataclasses.field(default_factory=lambda: RollingAverage(100))
    last_frame_time: float = 0.0

    # UI / metadata
    hd: bool = False
    status: str = "Not streaming"

    # logic state
    last_event_time: float = 0.0
    last_night_time_check: float = 0.0
    last_yolo_time: float = 0.0

    # motion detection
    motion_boxes_list: list = dataclasses.field(default_factory=list)
    classes_in_frame_set: set = dataclasses.field(default_factory=set)
    active_objects_set: set = dataclasses.field(default_factory=set)
    active_segments_list: list = dataclasses.field(default_factory=list)
    