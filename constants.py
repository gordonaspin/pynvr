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

MOTION_FRAME_COUNT = 2
TAIL_FRAME_COUNT = 5*20
PRE_RECORD = 2

CONFIDENCE_THRESHOLD = 0.3
MOTION_THRESHOLD = 200

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
