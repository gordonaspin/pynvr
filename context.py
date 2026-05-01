""" Context manager for the application. """
import dataclasses

@dataclasses.dataclass
class Context:
    """Context manager for the application."""
    directory: str
    username: str
    password: str
    gui_username: str
    gui_password: str
    camera_config: dict
    bind_address: str
    motion_threshold: float
    confidence_threshold: float
    motion_detect_frame_count: int
    resolution: list[int, int]
    model: str
    classes: list[str]
    debug: bool
    debug_files: bool = False
