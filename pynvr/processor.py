"""
FrameProcessor handles the main processing loop for a camera.
It reads frames from the camera, runs YOLO inference, updates motion tracking,
manages recording state, and prepares the final output frame.
"""
import time
import platform
from datetime import datetime
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from threading import Thread, Event, current_thread
from typing import Tuple

import cv2
import torch
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO
from ultralytics.engine.results import Results

from pynvr.constants import StreamingState, KNUTH_MULTIPLIER

from pynvr.camera.camera import Camera
from pynvr.config.config import ProcessorConfig, ModelConfig
from pynvr.debug_panel import draw_debug_panels
from pynvr.recorder import FrameRecorder
from pynvr.reader import Reader
from pynvr.utils import (
    tags_to_str,
    detect_object_color,
)

logger = getLogger("pynvr.processor")

class FrameProcessor:
    """
    FrameProcessor handles the main processing loop for a camera.
    It reads frames from the camera, runs YOLO inference, updates motion tracking,
    manages recording state, and prepares the final output frame.
    """
    def __init__(
        self,
        processor_config: ProcessorConfig,
        camera: Camera,
        reader: Reader,
        recorder: FrameRecorder,
        model_config: ModelConfig,
        stop_event: Event,
    ):
        self.processor_config: ProcessorConfig = processor_config
        self.camera: Camera = camera
        self.reader: Reader = reader
        self.recorder: FrameRecorder = recorder
        self.model: YOLO = YOLO(model_config.name)
        self.selected_classes: list[int] = []
        self.classes: dict[str, bool] = model_config.classes
        self.set_selected_classes(self.classes)
        logger.info("CUDA is available: %s", torch.cuda.is_available())
        if torch.cuda.is_available() and processor_config.device != "cpu":
            logger.info(self.camera.camera_info.name +
                        " using CUDA device " +
                        torch.cuda.get_device_name(torch.cuda.current_device()) +
                        " for YOLO inference")
            self.device = -1
        elif platform.system() == "Darwin" and processor_config.device == "mps":
            logger.info(f"{self.camera.camera_info.name} using MPS for YOLO inference")
            self.device: str = processor_config.device
        else:
            logger.info(f"{self.camera.camera_info.name} using CPU for YOLO inference")
            self.device: str = "cpu"

        self.stop_event: Event = stop_event
        self.thread: Thread | None = None
        self.frame_count: int = 0
        self.streaming_state: StreamingState = StreamingState.STREAMING_INIT
        self.last_night_time_check: float = time.time()
        self.last_yolo_result = None
        self.last_dets = np.empty((0, 6), dtype=np.float32)

    def start(self):
        """
        Start the frame processing thread.
        """
        self.thread = Thread(target=self._process_frames, daemon=True)
        self.thread.start()

    def stop(self):
        """
        Stop the frame processing thread.
        """
        logger.info(f"{self.camera.camera_info.name} stopping FrameProcessor")
        if self.thread is not None:
            self.thread.join()

    def set_selected_classes(self, classes: dict[str, bool]):
        """
        Set the selected classes for YOLO inference.
        """
        classname_to_classindex: dict = {v: k for k, v in self.model.names.items()}
        self.selected_classes: list[int] = [
            classname_to_classindex[n] for n in classes if classes[n]
        ]

    def _process_frames(self):
        current_thread().name = f"{self.camera.camera_info.name}FrameProcessor"

        while not self.stop_event.is_set():
            # --- FRAME ACQUISITION ---
            frame_bgr = self.reader.get_frame()
            if frame_bgr is None:
                if self.streaming_state == StreamingState.STREAMING_INIT:
                    pass
                elif self.streaming_state == StreamingState.STREAMING_NORMAL:
                    self.streaming_state = StreamingState.STREAMING_STOPPED
                continue

            self.streaming_state = StreamingState.STREAMING_NORMAL

            yolo_frame = self.camera.buffers.yolo_frame
            if yolo_frame is None:
                # single-pipe: YOLO runs on full frame
                yolo_frame = frame_bgr

            full_h, full_w = frame_bgr.shape[:2]
            yolo_h, yolo_w = yolo_frame.shape[:2]

            now = time.time()
            self.frame_count += 1

            # --- ENVIRONMENT UPDATES ---
            self._update_night_day(frame_bgr, now)

            # --- YOLO PIPELINE (run every N frames) ---
            if self.frame_count % self.processor_config.detect_every_nth_frame == 0:
                # Run YOLO
                yolo_result = self._run_yolo(yolo_frame)

                # Scale boxes to full resolution
                self._scale_yolo_boxes_to_full_res(
                    yolo_result,
                    full_w, full_h,
                    yolo_w, yolo_h
                )

                # Build detections for ByteTrack
                dets = self._build_yolo_dets(yolo_result)

                # Save for reuse
                self.last_yolo_result = yolo_result
                self.last_dets = dets

            else:
                # No YOLO this frame — reuse last detections
                yolo_result = self.last_yolo_result
                dets = self.last_dets

            # --- BYTE TRACK UPDATE (always every frame) ---
            self.camera.motion.update(dets, now, self.camera.is_night)

            # --- Class + color metadata ---
            self._update_class_color_metadata(frame_bgr, yolo_result)

            # --- RECORDING LOGIC ---
            if (
                self.recorder.fps.as_int() > 0
                and self.frame_count
                > self.processor_config.recorder.startup_delay * self.recorder.fps.as_int()
            ):
                self._update_recording_state(now)

            # Build debug UI if camera.debug
            self._render_debug_ui(frame_bgr, yolo_result)

            # Finalize output (select debug > YOLO > raw)
            self._finalize_output(frame_bgr, yolo_result)

            # Recorder always gets latest_frame
            self.recorder.add_frame(self.camera.latest_frame)


    # ----------------------------------------------------------------------
    # Night/day detection (no gray_buf)
    # ----------------------------------------------------------------------
    def _update_night_day(self, frame_bgr: NDArray[np.uint8], now: float) -> None:
        if now - self.last_night_time_check <= self.processor_config.night_check_period:
            return

        was_night = self.camera.is_night
        self.camera.is_night = self._is_night_time(frame_bgr)
        if was_night != self.camera.is_night:
            logger.info(
                self.camera.camera_info.name +
                f" night/day change: is_night={self.camera.is_night}")
        self.last_night_time_check = now

    def _is_night_time(
        self,
        frame: NDArray[np.uint8],
        luma_threshold: float = 90.0,
        ir_chroma_threshold: float = 4.0,
    ) -> bool:
        if frame is None or frame.size == 0:
            return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_luma = float(gray.mean())

        b, g, r = cv2.split(frame.astype(np.float32))
        chroma = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))

        ir_mode_on = bool((avg_luma < luma_threshold) and (chroma < ir_chroma_threshold))
        is_night = bool((avg_luma < luma_threshold) or ir_mode_on)

        return is_night

    # ----------------------------------------------------------------------
    # YOLO inference
    # ----------------------------------------------------------------------
    def _run_yolo(self, yolo_frame: NDArray[np.uint8]) -> Results | None:
        result: Results = self.model.predict(
            yolo_frame,
            conf=self.camera.camera_info.yolo_confidence.value,
            classes=self.selected_classes if self.selected_classes else None,
            verbose=False,
            imgsz=max(yolo_frame.shape[0], yolo_frame.shape[1]),
            device=self.device,
        )[0]
        return result

    # ----------------------------------------------------------------------
    # Scale YOLO boxes to full resolution
    # ----------------------------------------------------------------------
    def _scale_yolo_boxes_to_full_res(
        self,
        result: Results | None,
        full_w: int,
        full_h: int,
        yolo_w: int,
        yolo_h: int,
    ) -> None:
        if result is None or result.boxes is None or len(result.boxes) == 0:
            return

        boxes = result.boxes
        data = boxes.data.clone()
        xyxy = data[:, :4]

        # --- FFmpeg letterbox math ---
        # 1. Compute scaled height (preserving aspect ratio)
        scaled_h = int(full_h * (yolo_w / full_w))  # e.g., 480 * (640/704) = 436
        scaled_w = yolo_w                           # always 640

        # 2. Compute padding applied by FFmpeg
        pad_x = (yolo_w - scaled_w) // 2            # always 0 for 704x480
        pad_y = (yolo_h - scaled_h) // 2            # e.g., (640 - 436) / 2 = 102

        # --- Remove padding ---
        xyxy[:, [0, 2]] -= pad_x
        xyxy[:, [1, 3]] -= pad_y

        # --- Scale back to full resolution ---
        scale_x = full_w / scaled_w
        scale_y = full_h / scaled_h

        scale = xyxy.new_tensor([scale_x, scale_y, scale_x, scale_y])
        xyxy = xyxy * scale

        # --- Clamp to full-res image ---
        xyxy[:, 0].clamp_(0, full_w)
        xyxy[:, 2].clamp_(0, full_w)
        xyxy[:, 1].clamp_(0, full_h)
        xyxy[:, 3].clamp_(0, full_h)

        data[:, :4] = xyxy
        boxes.data = data

    # ----------------------------------------------------------------------
    # Build detections array for ByteTrack
    # ----------------------------------------------------------------------
    def _build_yolo_dets(self, yolo_result: Results | None) -> np.ndarray:
        if yolo_result is None or yolo_result.boxes is None:
            return np.empty((0, 6), dtype=np.float32)

        dets = []
        for box in yolo_result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf)
            cls = int(box.cls)
            dets.append([x1, y1, x2, y2, score, cls])

        return np.array(dets, dtype=np.float32)

    # ----------------------------------------------------------------------
    # Recording state (ByteTrack-only)
    # ----------------------------------------------------------------------
    def _update_recording_state(self, now: float) -> None:
        motion = self.camera.motion
        rec = self.camera.recording_state

        rec.should_record = motion.has_moving_object

        rec.should_continue = (
            motion.has_moving_object
            or (now - motion.last_motion_time < self.processor_config.recorder.post_duration)
        )

        if not rec.recording and rec.should_record:
            rec.recording = True
            rec.recording_start_time = now
            motion.active_objects_dict = deepcopy(motion.classes_in_frame_dict)
            self.recorder.start_recording()
            logger.info(
                self.camera.camera_info.name +
                f" recording start {tags_to_str(motion.active_objects_dict)}"
            )

        if rec.recording:
            motion.finalize_active_objects()

        if rec.recording and not rec.should_continue:
            recorder = self.recorder
            recorder.stop_recording()
            self.recorder = type(recorder)(
                    camera=recorder.camera,
                    stop_event=recorder.stop_event,
                    add_recording_callback=recorder.add_recording_callback,
                    recorder_config=recorder.recorder_config
                    )

            rec.recording = False
            motion.reset_active_objects()

    # ----------------------------------------------------------------------
    # Class + color metadata
    # ----------------------------------------------------------------------
    def _update_class_color_metadata(
        self,
        frame_bgr: np.ndarray,
        yolo_result: Results | None,
    ):
        self.camera.motion.clear_frame_classes()

        if yolo_result is None or yolo_result.boxes is None:
            return

        names = self.model.names
        full_h, full_w = frame_bgr.shape[:2]

        # Get moving track IDs from MotionDetector
        moving_ids = self.camera.motion.moving_track_ids
        tracks = self.camera.motion.active_tracks

        # Build map: track_id → tlbr
        track_map = {t.track_id: t.tlbr for t in tracks if t.track_id in moving_ids}

        for box in yolo_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1 = max(0, min(full_w - 1, x1))
            y1 = max(0, min(full_h - 1, y1))
            x2 = max(0, min(full_w, x2))
            y2 = max(0, min(full_h, y2))

            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Match YOLO box to a moving track
            matched_tid = None
            for tid, (tx1, ty1, tx2, ty2) in track_map.items():
                ix1 = max(x1, tx1)
                iy1 = max(y1, ty1)
                ix2 = min(x2, tx2)
                iy2 = min(y2, ty2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                if inter > 0:
                    matched_tid = tid
                    break

            # Only add metadata if this YOLO box belongs to a moving track
            if matched_tid is None:
                continue

            cls_idx = int(box.cls)
            class_name = names[cls_idx]
            color = detect_object_color(roi, self.camera.is_night)
            self.camera.motion.add_class_color(class_name, color)


    # ----------------------------------------------------------------------
    # Debug UI
    # ----------------------------------------------------------------------
    def _render_debug_ui(
        self,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None,
    ) -> None:
        if not self.camera.camera_info.debug:
            return

        self.camera.debug_motion_image = draw_debug_panels(
            self.camera,
            self.model.names,
            self.frame_count,
            frame_bgr,
            yolo_result,
            [t.tlbr for t in self.camera.motion.active_tracks],
        )

        if self.camera.recording_state.recording:
            timestamp_str = datetime.fromtimestamp(time.time()).strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            filename: Path = Path(
                self.camera.camera_info.images_dir, f"{timestamp_str}.jpg"
            )
            cv2.imwrite(filename, self.camera.debug_motion_image)

    # ----------------------------------------------------------------------
    # Frame selection + final output
    # ----------------------------------------------------------------------
    def _select_frame(
        self,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None,
    ) -> Tuple[NDArray[np.uint8], bool]:
        if self.camera.camera_info.debug and self.camera.debug_motion_image is not None:
            return self.camera.debug_motion_image, True

        return self._apply_yolo_overlay(frame_bgr, yolo_result), False

    def _finalize_output(
        self,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None,
    ) -> None:
        final_frame, _ = self._select_frame(frame_bgr, yolo_result)
        self.camera.latest_frame = final_frame

    def _apply_yolo_overlay(
        self,
        frame_bgr: NDArray[np.uint8],
        yolo_result: Results | None,
    ) -> NDArray[np.uint8]:

        if yolo_result is None:
            return frame_bgr

        if self.camera.camera_info.render_annotations == "never":
            return frame_bgr

        # Only draw YOLO boxes if ByteTrack says something is moving
        if (self.camera.camera_info.render_annotations == "motion"
            and not self.camera.motion.has_moving_object):
            return frame_bgr

        return self.draw_yolo_boxes_fullres(
            frame_bgr,
            yolo_result,
            self.model.names,
        )

    #pylint: disable=too-many-statements
    def draw_yolo_boxes_fullres(
        self,
        frame,
        result,
        class_names=None,
        min_conf=0.25,
        box_thickness=2,
        font_scale=0.5,
    ):
        """
        Draw YOLO boxes + ByteTrack IDs on the full-resolution frame.
        Assumes YOLO boxes have already been scaled to full resolution.
        Only draws boxes for tracks that are currently moving.
        """

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return frame

        height, width = frame.shape[:2]

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        # Pull active tracks + moving track IDs from MotionDetector
        tracks = self.camera.motion.active_tracks
        moving_ids = self.camera.motion.moving_track_ids

        def yolo_color(cid: int):
            h = (cid * KNUTH_MULTIPLIER) & 0xFFFFFFFF
            r = (h >> 16) & 255
            g = (h >> 8) & 255
            b = h & 255
            return (int(b * 0.7 + 75), int(g * 0.7 + 75), int(r * 0.7 + 75))

        # Helper: find the track whose box overlaps the YOLO box
        def match_track_to_box(x1, y1, x2, y2):
            best_id = None
            best_iou = 0.0

            for t in tracks:
                tx1, ty1, tx2, ty2 = t.tlbr

                # Compute IoU
                ix1 = max(x1, tx1)
                iy1 = max(y1, ty1)
                ix2 = min(x2, tx2)
                iy2 = min(y2, ty2)

                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a1 = (x2 - x1) * (y2 - y1)
                a2 = (tx2 - tx1) * (ty2 - ty1)
                union = a1 + a2 - inter
                iou = inter / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_id = t.track_id

            return best_id

        overlay = frame.copy()

        # Draw each YOLO box ONLY if its track is moving
        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
            if conf < min_conf:
                continue

            x1 = int(max(0, min(width - 1, x1)))
            y1 = int(max(0, min(height - 1, y1)))
            x2 = int(max(0, min(width - 1, x2)))
            y2 = int(max(0, min(height - 1, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            # Match YOLO → ByteTrack
            track_id = match_track_to_box(x1, y1, x2, y2)

            # Skip if no track or track is not moving
            if track_id is None or track_id not in moving_ids:
                continue

            color = yolo_color(cid)

            # Draw box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, box_thickness)

            # Build label
            label = class_names[cid] if class_names else str(cid)
            label = f"{label} {conf:.2f}"
            track = next((x for x in tracks if x.track_id == track_id), None)
            if self.camera.debug:
                if track is not None:
                    label = f"i:{track_id} v:{track.relative_speed:.2f} {label} {conf:.2f}"

            # Draw label background
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            tx1, ty1 = x1, max(0, y1 - th - baseline - 2)
            tx2, ty2 = x1 + tw + 4, y1

            cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), color, -1)

            # Draw text
            cv2.putText(
                overlay,
                label,
                (tx1 + 2, ty2 - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame
