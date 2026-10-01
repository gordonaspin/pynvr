""" utility classes """
import subprocess
import time
from collections import defaultdict, deque
from datetime import datetime
from math import sqrt

import cv2
import numpy as np

class RollingAverage:
    """ Class to compute rolling average """
    def __init__(self, window_size=20):
        self.window = deque(maxlen=window_size)
        self.sum = 0.0

    def update(self, value):
        """ update """
        # If full, remove oldest from sum
        if len(self.window) == self.window.maxlen:
            self.sum -= self.window[0]

        self.window.append(value)
        self.sum += value

        return self.sum / len(self.window)

    def value(self):
        """ get value """
        if not self.window:
            return 0.0
        return self.sum / len(self.window)

    def as_int(self):
        """ get integer value """
        return int(self.value())


def make_ts_string(epoch=None):
    """ return YYYYMMDD_HHMMSS """
    if epoch is None:
        epoch = time.time()
    return datetime.fromtimestamp(epoch).strftime("%Y%m%d_%H%M%S")

def make_ts_string_precise(epoch=None):
    """ return YYYYMMDD_HHMMSS_mmm """
    if epoch is None:
        epoch = time.time()
    return datetime.fromtimestamp(epoch).strftime("%Y%m%d_%H%M%S_%f")

def make_readable_ts(epoch=None):
    """ return YYYY/MM/DD HH:MM:SS """
    if epoch is None:
        epoch = time.time()
    return datetime.fromtimestamp(epoch).strftime("%Y/%m/%d %H:%M:%S")

def make_readable_hms(epoch=None):
    """ return HH:MM:SS """
    if epoch is None:
        epoch = time.time()
    return datetime.fromtimestamp(epoch).strftime("%H:%M:%S")

def tags_to_str(tags: defaultdict[set]):
    """ Flattens tags to string """
    if not tags:
        return ""

    parts = []
    for obj, colors in tags.items():
        object_str = obj
        color_str = "-".join(colors)
        parts.append(f"{object_str}-{color_str}")
    return "_".join(parts)

def get_camera_resolution(url: str):
    """ run ffprobe to get camera stream resolution """
    ffprobe_cmd = [
        "timeout 5s ffprobe",
        "-v error",
        "-rtsp_transport tcp",
        "-analyzeduration 0",
        "-probesize 32",
        "-select_streams v:0",
        "-show_entries stream=width,height",
        "-of csv=p=0",
        f"'{url}'"
    ]
    ffprobe_cmd_str = " ".join(ffprobe_cmd)
    try:
        output = subprocess.check_output(ffprobe_cmd_str, shell=True).decode().strip()
        width, height = map(int, output.split(","))
        return width, height
    except Exception:
        return 0, 0

def detect_object_color(roi_bgr, is_night: bool):
    """
    Adaptive color classifier.
    Uses day classifier (LAB + kmeans) or night classifier (brightness-based)
    depending on lighting conditions.
    """

    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown"

    if is_night:
        return detect_object_color_night(roi_bgr)

    return detect_object_color_day(roi_bgr)


def detect_object_color_day(roi_bgr, k=2):
    """ detect color of object in region """
    if roi_bgr is None or roi_bgr.size == 0:
        return "unknown"

    # Smooth noise
    roi = cv2.GaussianBlur(roi_bgr, (5, 5), 0)

    # Convert to true LAB
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0

    # Normalize LAB for k-means stability
    lab_norm = np.empty_like(lab)
    lab_norm[:, :, 0] = lab[:, :, 0] / 100.0      # L 0–100
    lab_norm[:, :, 1] = lab[:, :, 1] / 128.0      # a -128–128
    lab_norm[:, :, 2] = lab[:, :, 2] / 128.0      # b -128–128

    pixels = lab_norm.reshape((-1, 3))

    # K-means clustering
    _, labels, centers = cv2.kmeans(
        pixels.astype(np.float32), k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        3,
        cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(labels.flatten())
    total = counts.sum()

    # Sort clusters by size
    idxs = np.argsort(-counts)

    for idx in idxs:
        if counts[idx] < 0.05 * total:
            continue

        # Convert normalized center back to LAB
        lab = centers[idx][0] * 100.0
        a = centers[idx][1] * 128.0
        b = centers[idx][2] * 128.0
        lab_color = np.array([lab, a, b], dtype=np.float32)

        # Remove specular highlights
        if lab > 95 and abs(a) < 5 and abs(b) < 5:
            continue

        # Merge low-chroma colors
        chroma = np.sqrt(a*a + b*b)
        if chroma < 8:
            return "gray"

        name = classify_color_lab(lab_color)
        if name != "unknown":
            return name

    return "unknown"


def detect_object_color_night(roi_bgr):
    """ detect color of region at night """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))

    # IR reflection → white
    b, g, r = cv2.split(roi_bgr.astype(np.float32))
    chroma = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))
    if mean > 170 and chroma < 10:
        return "white"

    if mean < 40:
        return "black"

    if mean < 120:
        return "gray"

    return "white"

#pylint: disable=too-many-return-statements
#pylint: disable=too-many-branches
def classify_color_lab(lab_color):
    """ classify the LAB color """
    # -----------------------------------------
    # Reference LAB colors (approximate swatches)
    # -----------------------------------------
    ref_colors = {
        # --- Primary automotive colors ---
        "red":      np.array([53,   80,   67]),
        "blue":     np.array([32,   79, -108]),
        "green":    np.array([87,  -86,   83]),
        "yellow":   np.array([97,  -21,   94]),
        "orange":   np.array([65,   45,   70]),
        "purple":   np.array([60,   98,  -60]),
        "pink":     np.array([75,   25,   -5]),
        "cyan":     np.array([91,  -48,  -14]),

        # --- Earth tones ---
        "brown":    np.array([40,   15,   20]),
        "beige":    np.array([78,    0,   18]),
        "tan":      np.array([70,    5,   30]),

        # --- Metallics ---
        "silver":   np.array([82,    0,    0]),
        "gold":     np.array([75,    5,   55]),

        # --- Additional useful colors ---
        "lime":     np.array([90,  -70,   80]),
        "teal":     np.array([60,  -40,  -10]),
        "navy":     np.array([20,   10,  -40]),
    }

    lab, a, b = lab_color
    chroma = sqrt(a*a + b*b)

    # -----------------------------------------
    # Neutral colors
    # -----------------------------------------
    if lab < 35:
        return "black"

    if chroma < 10:
        if lab > 75:
            return "white"
        if lab > 40:
            return "light_gray"
        return "gray"

    # -----------------------------------------
    # Blue safeguard (fixes blue→brown)
    # -----------------------------------------
    # Dark + negative b = blue/navy
    if lab < 55 and b < -5:
        return "blue"

    # -----------------------------------------
    # Metallics
    # -----------------------------------------
    # Silver: mid‑L, low chroma
    if 55 < lab < 110 and chroma < 22:
        return "silver"

    # Gold: warm b channel + moderate chroma
    if 55 < lab < 110 and 22 <= chroma < 45 and b > 15:
        return "gold"

    # -----------------------------------------
    # Earth tones (corrected boundaries)
    # -----------------------------------------
    if 35 < lab < 80 and 12 < chroma < 40:
        # Brown must have positive b (fixes blue→brown)
        if 0 <= b <= 10:
            return "brown"
        if 10 < b <= 25:
            return "beige"
        if b > 25:
            return "tan"

    # -----------------------------------------
    # Standard colors (LAB distance)
    # -----------------------------------------
    best = None
    best_dist = 1e9

    for name, ref in ref_colors.items():
        dist = np.linalg.norm(lab_color - ref)
        if dist < best_dist:
            best_dist = dist
            best = name

    return best


def tlbr_to_tlwh(box):
    """
    Convert [x1,y1,x2,y2] → [x,y,w,h].
    """
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)


def tlwh_to_xyah(tlwh):
    """
    Convert [x,y,w,h] → [cx,cy,area,ratio].
    """
    x, y, w, h = tlwh
    cx = x + w / 2
    cy = y + h / 2
    area = w * h
    ratio = w / max(1e-6, h)
    return np.array([cx, cy, area, ratio], dtype=np.float32)
