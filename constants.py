"""Constants"""
import os
from enum import Enum, auto
# For retrying connection after timeouts and errors

# =========================
# SETTINGS
# =========================
MAX_LOG_LINES = 1000
YOLO_SIZE = (608, 416)
RENDER_SIZE = (304, 208)

MOTION_DETECT_FRAME_COUNT = 40
NO_MOTION_DETECT_FRAME_COUNT = 5*20
PRE_RECORD_SEGMENTS = 2

CONFIDENCE_THRESHOLD_MIN = 0.1
CONFIDENCE_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD_MAX = 0.9

MOTION_THRESHOLD_MIN = [50, 50]
MOTION_THRESHOLD = [200, 4000]
MOTION_THRESHOLD_MAX = [10000,10000]

BUFFER_SECONDS = 60
EVENT_COOLDOWN = 1
REQUIRE_OBJECT_FOR_RECORDING = True

RECORDINGS_DIR = "recordings"
SEGMENTS_DIR = os.path.join(RECORDINGS_DIR, "segments")
IMAGES_DIR = os.path.join(RECORDINGS_DIR, "images")

class ExitCode(Enum):
    """
    ExitCode definitions
    """
    EXIT_NORMAL: int = 0
    EXIT_FAILED_ALREADY_RUNNING: int = auto()
    EXIT_FAILED_CLICK_EXCEPTION: int = auto()
    EXIT_FAILED_CLICK_USAGE: int = auto()
    EXIT_FAILED_NOT_A_DIRECTORY: int = auto()
    EXIT_FAILED_MISSING_COMMAND: int = auto()
    EXIT_FAILED_LOGIN: int = auto()
    EXIT_FAILED_CLOUD_API: int = auto()
    EXIT_FAILED_2FA_REQUIRED: int = auto()
    EXIT_FAILED_SEND_2SA_CODE: int = auto()
    EXIT_FAILED_VERIFY_2SA_CODE: int = auto()
    EXIT_FAILED_VERIFY_2FA_CODE: int = auto()
