import asyncio
import fractions
import time
from typing import List
from logging import getLogger

logger = getLogger("nvr")

import cv2
import numpy as np
from aiortc.mediastreams import VideoStreamTrack
from av import VideoFrame

class CameraTrack(VideoStreamTrack):
    """
    WebRTC track that streams a single camera's latest_frame.
    """
    kind = "video"

    def __init__(self, camera):
        super().__init__()
        self._camera = camera

    async def recv(self) -> VideoFrame:
        frame = self._camera.latest_frame

        if frame is None:
            # Provide a fallback frame so SDP negotiation succeeds
            #logger.info(f"CameraTrack.recv for {self._camera.name} - frame is None, providing placeholder")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = time.time_ns()
        video_frame.time_base = fractions.Fraction(1, 1_000_000_000)
        return video_frame

class MosaicTrack(VideoStreamTrack):
    """
    WebRTC track that streams a high-quality mosaic of multiple cameras.
    """
    kind = "video"

    def __init__(self, cameras: List[object], cols: int = 5):
        super().__init__()
        self._cameras = cameras
        self._cols = cols

        # 4K width, height computed dynamically to preserve 4:3 tiles
        self.MOSAIC_W = 3840

    async def recv(self) -> VideoFrame:
        # Collect frames
        frames = []
        for cam in self._cameras:
            frame = cam.latest_frame
            if frame is None:
                frame = np.zeros((480, 704, 3), dtype=np.uint8)
            frames.append(frame)

        total = len(frames)
        rows = int(np.ceil(total / self._cols))

        # Camera aspect ratio (704x480)
        CAM_ASPECT = 704 / 480

        # Tile width fixed by mosaic width
        TILE_W = self.MOSAIC_W // self._cols

        # Tile height computed to preserve 4:3
        TILE_H = int(TILE_W / CAM_ASPECT)

        # Mosaic height computed from tile height
        self.MOSAIC_H = TILE_H * rows

        # Prepare mosaic canvas
        mosaic = np.zeros((self.MOSAIC_H, self.MOSAIC_W, 3), dtype=np.uint8)

        for idx, frame in enumerate(frames):
            src_h, src_w, _ = frame.shape

            # Compute tile grid position
            row = idx // self._cols
            col = idx % self._cols

            # Resize while preserving aspect ratio (no cropping)
            resized = cv2.resize(
                frame,
                (TILE_W, TILE_H),
                interpolation=cv2.INTER_AREA if (src_w > TILE_W or src_h > TILE_H)
                else cv2.INTER_CUBIC
            )

            # Place tile
            y0 = row * TILE_H
            x0 = col * TILE_W
            mosaic[y0:y0+TILE_H, x0:x0+TILE_W] = resized

        # Convert to VideoFrame
        video_frame = VideoFrame.from_ndarray(mosaic, format="bgr24")
        video_frame.pts = time.time_ns()
        video_frame.time_base = fractions.Fraction(1, 1_000_000_000)

        return video_frame

