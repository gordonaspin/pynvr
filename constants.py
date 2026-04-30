"""Constants"""
from enum import Enum, auto
# For retrying connection after timeouts and errors

# =========================
# SETTINGS
# =========================
MAX_LOG_LINES = 1000
RENDER_SIZE = (304, 208)

MOTION_DETECT_FRAME_COUNT = 2*20
NO_MOTION_DETECT_FRAME_COUNT = 5*20
PRE_RECORD_SEGMENTS = 5

CONFIDENCE_THRESHOLD_MIN = 0.1
CONFIDENCE_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD_MAX = 0.9

# change of pixels [day, night]
MOTION_THRESHOLD_MIN = [100, 200]
MOTION_THRESHOLD = [500, 2000]
MOTION_THRESHOLD_MAX = [2500, 5000]

BUFFER_SECONDS = 120
EVENT_COOLDOWN = 1

PERIODIC_CHECK_INTERVAL = 300 # seconds
NIGHT_TIME_THRESHOLD = 100

STATUS_UPDATE_INTERVAL = 0.5 # seconds

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
