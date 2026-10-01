""" Frame buffers for handling different resolution frames. """
import numpy as np
from numpy.typing import NDArray

from pynvr.config.config import Resolution

class FrameBuffers:
    """
    ByteTrack-only frame buffers.

    Responsibilities:
    - Hold full-resolution frame for recording, UI, and color detection
    - Hold YOLO-resolution frame when using a dual-pipe (different model resolution)
    - Provide byte-level views for fast pipe reads

    All legacy motion-diff buffers (background, diff, Sobel, thresh, etc.)
    have been removed.
    """

    def __init__(self, camera_resolution: Resolution, model_resolution: Resolution):

        # ------------------------------------------------------------------
        # FULL-RES FRAME BUFFER
        # ------------------------------------------------------------------
        # Used for:
        # - recording
        # - debug UI overlays
        # - color detection
        # - night/day detection (via on-the-fly grayscale)
        self.full_frame: NDArray[np.uint8] = np.empty(
            (camera_resolution.height, camera_resolution.width, 3), dtype=np.uint8
        )
        self.full_frame_bytes = memoryview(self.full_frame).cast("B")
        self.full_frame_size = camera_resolution.height * camera_resolution.width * 3

        # Read buffer for full-res pipe
        self.read_buf_full = bytearray(self.full_frame_size)
        self.read_view_full = memoryview(self.read_buf_full)

        # ------------------------------------------------------------------
        # YOLO FRAME BUFFER (DUAL-PIPE ONLY)
        # ------------------------------------------------------------------
        # If the camera resolution differs from the model resolution,
        # we maintain a separate buffer for the YOLO input frame.

        if (camera_resolution.width != model_resolution.width or
            camera_resolution.height != model_resolution.height):
            self.yolo_frame: NDArray[np.uint8] = np.empty(
                (model_resolution.height, model_resolution.width, 3), dtype=np.uint8
            )
            self.yolo_frame_bytes = memoryview(self.yolo_frame).cast("B")
            self.yolo_frame_size = model_resolution.height * model_resolution.width * 3

            self.read_buf_yolo = bytearray(self.yolo_frame_size)
            self.read_view_yolo = memoryview(self.read_buf_yolo)
        else:
            # Single-pipe: YOLO runs directly on full_frame
            self.yolo_frame = None
            self.yolo_frame_bytes = None
            self.yolo_frame_size = None
            self.read_buf_yolo = None
            self.read_view_yolo = None
