import dataclasses
from subprocess import Popen
import numpy as np

@dataclasses.dataclass
class Camera:
    name: str
    url: str
    enabled: bool
    recordings_dir: str
    segments_dir: str
    images_dir: str
    frame: np.ndarray = None
    process: Popen = None
    hd: bool = False
    last_event_time: float = 0.0
    last_night_time_check: float = 0.0
    last_status_update: float = 0.0
    active_objects: dict = dataclasses.field(default_factory=dict)
    active_segments: dict = dataclasses.field(default_factory=dict)
    status: str = "Not streaming"