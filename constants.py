"""Constants"""
from enum import Enum, auto
import numpy as np # For retrying connection after timeouts and errors

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

# % change of pixels
MOTION_THRESHOLD_MIN = 0.1
MOTION_THRESHOLD = 5.0
MOTION_THRESHOLD_MAX = 25.0

BUFFER_SECONDS = 120
EVENT_COOLDOWN = 1

PERIODIC_CHECK_INTERVAL = 300 # seconds
NIGHT_TIME_THRESHOLD = 100

STATUS_UPDATE_INTERVAL = 0.5 # seconds

# -----------------------------------------
# Reference LAB colors (approximate swatches)
# -----------------------------------------
REF_COLORS = {
    # Standard colors
    "red":     np.array([53,  80,  67]),
    "orange":  np.array([65,  45,  70]),
    "yellow":  np.array([97, -21,  94]),
    "green":   np.array([87, -86,  83]),
    "cyan":    np.array([91, -48, -14]),
    "blue":    np.array([32,  79, -108]),
    "purple":  np.array([60,  98, -60]),
    "pink":    np.array([75,  25,  -5]),

    # Earth tones
    "brown":   np.array([37,  14,  18]),
    "beige":   np.array([80,   0,  20]),
    "tan":     np.array([70,   5,  30]),

    # Metallics
    "gold":    np.array([75,   5,  65]),
    "silver":  np.array([80,   0,   0]),
}

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
