"""
WebRTC tracks for streaming camera frames and mosaics.
"""
import fractions
import time
from logging import getLogger
from typing import List

import cv2
import numpy as np
from aiortc.mediastreams import VideoStreamTrack
from av import VideoFrame

from pynvr.config.config import MosaicConfig
from pynvr.camera.camera import Camera

logger = getLogger("pynvr.webrtc")

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

    def __init__(self, cameras: List[Camera], mosaic_config: MosaicConfig):
        super().__init__()
        self._cameras = [camera for camera in cameras if camera.camera_info.enabled]
        self.rows = mosaic_config.rows
        self.cols = mosaic_config.columns
        #self._max_cols = max_cols

        # 4K width, height computed dynamically to preserve 4:3 tiles
        self.mosaic_w = mosaic_config.resolution.width
        self.mosaic_h = 0  # computed dynamically based on tile height

    async def recv(self) -> VideoFrame:
        # Collect frames
        frames = []
        for camera in self._cameras:
            frame = camera.latest_frame
            if frame is None:
                frame = np.zeros((480, 704, 3), dtype=np.uint8)
            frames.append(frame)

        if not frames:
            # No enabled cameras → return a black frame
            black = np.zeros((480, 704, 3), dtype=np.uint8)
            vf = VideoFrame.from_ndarray(black, format="bgr24")
            vf.pts = time.time_ns()
            vf.time_base = fractions.Fraction(1, 1_000_000_000)
            return vf

        # Camera aspect ratio (704x480)
        cam_aspect = 704 / 480

        # Tile width fixed by mosaic width
        tile_w = self.mosaic_w // self.cols

        # Tile height computed to preserve 4:3
        tile_h = int(tile_w / cam_aspect)

        # Pad with black tiles if needed
        needed = self.rows * self.cols
        while len(frames) < needed:
            frames.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

        # Mosaic height computed from tile height
        self.mosaic_h = tile_h * self.rows

        # Prepare mosaic canvas
        mosaic = np.zeros((self.mosaic_h, self.mosaic_w, 3), dtype=np.uint8)

        for idx, frame in enumerate(frames):
            src_h, src_w, _ = frame.shape

            # Compute tile grid position
            row = idx // self.cols
            col = idx % self.cols

            # Resize while preserving aspect ratio (no cropping)
            resized_frame = cv2.resize(
                frame,
                (tile_w, tile_h),
                interpolation=cv2.INTER_AREA if (src_w > tile_w or src_h > tile_h)
                else cv2.INTER_CUBIC
            )

            # Place tile
            y0 = row * tile_h
            x0 = col * tile_w
            mosaic[y0:y0+tile_h, x0:x0+tile_w] = resized_frame

        # Convert to VideoFrame
        video_frame = VideoFrame.from_ndarray(mosaic, format="bgr24")
        video_frame.pts = time.time_ns()
        video_frame.time_base = fractions.Fraction(1, 1_000_000_000)

        return video_frame
