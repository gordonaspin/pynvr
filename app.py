import os
from pathlib import Path
import time
from datetime import datetime
import logging

import cv2
import torch
import gradio as gr
from ultralytics import YOLO

from logger import setup_logging, log_event, event_log
from nvr import get_segments, merge_segments, cleanup_segments, start_segment_recorders
from constants import (
    RECORDINGS_DIR,
    IMAGES_DIR,
    SEGMENTS_DIR,
    TAIL_FRAME_COUNT,
    MOTION_FRAME_COUNT,
    PRE_RECORD,
    REQUIRE_OBJECT_FOR_RECORDING,
    EVENT_COOLDOWN,
    YOLO_SIZE,
    MOTION_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    RENDER_SIZE,

)
from cameras import CAMERAS

logger = logging.getLogger("yolo-rtsp-security-cam")
setup_logging("logging-config.json")

start_segment_recorders(CAMERAS)

LOG_STREAM_DIV = """
    <div class="inner-log" style="
        height: 300px; 
        overflow-y: auto; 
        border: 1px solid #ccc; 
        padding: 5px; 
        font-family: monospace;
        font-size: small;
        background-color: #1e1e1e; 
        color: #ffffff;
        box-sizing: border-box;
    ">
    <div style="font-weight: bold; margin-bottom: 8px; font-size: small;">
        📜 Event Log
    </div>
    """

# =========================
# GLOBAL STATE
# =========================
model = YOLO("yolov8n.pt")

CLASSES_OF_INTEREST = ["person","car","bicycle","motorcycle","bus","truck","cat","dog"]
CLASS_TO_IDX = {v: k for k, v in model.names.items()}
selected_classes = [CLASS_TO_IDX[i] for i in CLASSES_OF_INTEREST]

hd_mode_states = {name: False for name, _ in CAMERAS}
active_events = {}
active_objects = {}
last_event_time = {}

# =========================
# UI HANDLERS
# =========================
def update_confidence_threshold(val):
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = val
    log_event(f"Confidence updated → {val}")

def update_motion_threshold(val):
    global MOTION_THRESHOLD
    MOTION_THRESHOLD = val
    log_event(f"Motion threshold → {val}")

def update_detection_classes(names):
    global selected_classes
    selected_classes = [CLASS_TO_IDX[n] for n in names] if names else []
    log_event(f"Classes → {names}")

def update_hd_mode(cam, val):
    hd_mode_states[cam] = val


def keep_overlapping_any(boxes, ref_boxes):
    """
    boxes: YOLO r.boxes.xyxy  -> (N, 4)
    ref_boxes: cv2.boundingRect -> (M, 4) in (x, y, x1, y1)
    """

    # 1. Ensure correct shapes
    boxes = boxes.view(-1, 4)
    ref_boxes = ref_boxes.view(-1, 4)

    # 3. Compute pairwise intersection
    x1 = torch.maximum(boxes[:, None, 0], ref_boxes[None, :, 0])
    y1 = torch.maximum(boxes[:, None, 1], ref_boxes[None, :, 1])
    x2 = torch.minimum(boxes[:, None, 2], ref_boxes[None, :, 2])
    y2 = torch.minimum(boxes[:, None, 3], ref_boxes[None, :, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)

    overlap = (inter_w * inter_h) > 0  # (N, M)

    # 4. Keep if overlaps ANY ROI
    return overlap.any(dim=1)

# =========================
# STREAM
# =========================
def make_stream_fn(cam_id, rtsp_url):
    def stream():
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cam_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cam_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        cam_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((cam_fourcc >> 8 * i) & 0xFF) for i in range(4)])

        if not cap.isOpened():
            log_event("Failed to open stream", "error", cam_id)
            raise gr.Error(f"Failed to open stream: {rtsp_url}")

        log_event(f"Stream opened with resolution {cam_res} {cam_fps:.1f}fps codec:{codec}", "info", cam_id)

        prev_gray = None
        motion_frames = 0
        no_motion_frames = 0
        recording = False
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                log_event("Stream read failed, attempting to reconnect...", "warn", cam_id)
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            small = cv2.resize(frame, YOLO_SIZE)

            # motion
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),(21,21),0)
            motion_boxes = []
            overlay = None

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
                score = cv2.countNonZero(thresh)

                if score > MOTION_THRESHOLD:
                    #log_event(f"Motion detected (score: {score})", "info", cam_id)                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    overlay = small.copy()
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        x1, y1, w, h = cv2.boundingRect(contour)
                        x2 = x1 + w
                        y2 = y1 + h
                        if area < MOTION_THRESHOLD / 10:  # filter small noise
                            cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 1)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
                            #log_event(f"ignoring motion contour rect ({x1}, {y1}), ({x2}, {y2}) with area {area}")
                        else:
                            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 1)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            #log_event(f"motion contour rect ({x1}, {y1}), ({x2}, {y2}) with area {area}")
                            motion_boxes.append([x1, y1, x2, y2])
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        tag = "_".join(classes_in_frame) if classes_in_frame else "motion"
                        img_dir = os.path.join(IMAGES_DIR, cam_id)
                        image_filename = os.path.join(img_dir, f"{timestamp}_{tag}.jpg")
                        cv2.imwrite(image_filename, overlay)

            prev_gray = gray

            # YOLO
            result = None
            classes_in_frame = set()

            if motion_boxes:
                result = model.predict(small, conf=CONFIDENCE_THRESHOLD, classes=selected_classes if selected_classes else None, verbose=False)[0]
                boxes = result.boxes.xyxy.reshape(-1, 4)
                ref_motion_boxes = torch.as_tensor(motion_boxes, dtype=boxes.dtype, device=boxes.device)
                keep = keep_overlapping_any(boxes, ref_motion_boxes)
                #log_event(f"motion_boxes {ref_motion_boxes.shape} {ref_motion_boxes}")
                #log_event(f"boxes {boxes.shape} {boxes}")
                #log_event(f"keep {keep.shape} {keep}")
                boxes = result.boxes[keep]

                for box in boxes:
                    classes_in_frame.add(model.names[int(box.cls)])

            # counters
            if motion_boxes:
                motion_frames += 1
                no_motion_frames = 0
            else:
                no_motion_frames += 1

            valid_objects = len(classes_in_frame) > 0
            #if valid_objects:
            #    log_event(f"Object classes detected: {', '.join(classes_in_frame)}", "info", cam_id)

            # start
            now = time.time()
            if motion_frames >= MOTION_FRAME_COUNT and not recording:
                if (not REQUIRE_OBJECT_FOR_RECORDING or valid_objects):
                    if now - last_event_time.get(cam_id,0) > EVENT_COOLDOWN:
                        recording = True
                        active_events[cam_id] = get_segments(cam_id, PRE_RECORD)
                        active_objects[cam_id] = set(classes_in_frame)
                        last_event_time[cam_id] = now
                        log_event(f"Recording start {list(classes_in_frame)}", "record", cam_id)

            # update
            if recording:
                active_events[cam_id] += get_segments(cam_id,1)
                active_objects[cam_id].update(classes_in_frame)

            # stop
            if recording and no_motion_frames >= TAIL_FRAME_COUNT:
                recording = False
                segments = list(dict.fromkeys(active_events.get(cam_id,[])))
                active_events.pop(cam_id, None)
                classes_in_frame = active_objects.pop(cam_id)

                if segments:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    tag = "_".join(classes_in_frame) if classes_in_frame else "motion"

                    cam_dir = os.path.join(RECORDINGS_DIR, cam_id)

                    recording_filename = os.path.join(cam_dir, f"{timestamp}_{tag}.mp4")
                    merge_segments(segments, recording_filename)

                    recorded_cap = cv2.VideoCapture(recording_filename)
                    frame_count = recorded_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if frame_count < TAIL_FRAME_COUNT + cam_fps and os.path.isfile(recording_filename):
                        os.remove(recording_filename)
                        log_event(f"Recording auto-deleted {os.path.basename(recording_filename)} with {frame_count} frames", "record", cam_id, file_path=recording_filename)
                    else:
                        log_event(f"Recording available {os.path.basename(recording_filename)}", "record", cam_id, file_path=recording_filename)

                motion_frames = 0
                no_motion_frames = 0

            # render
            img = result.plot() if result else small
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if overlay is not None:
                img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

            if not hd_mode_states[cam_id]:
                img = cv2.resize(img, RENDER_SIZE)

            fps = 1/(time.time()-prev_time)
            prev_time = time.time()

            status = "🔴 REC" if recording else "🟢 LIVE"
            yield img, f"{status} | {cam_res[0]}x{cam_res[1]} | FPS {int(fps)}" + (f" | {len(classes_in_frame)} objects" if len(classes_in_frame) > 0 else "")

    return stream

# =========================
# UI STREAMS
# =========================
# --- Event log streamer ---
def log_stream():
    while True:
        html_content = "".join(event_log)
        # JS snippet to scroll to bottom
        scroll_js = "<script>var el=document.currentScript.parentElement; el.scrollTop=el.scrollHeight;</script>"
        yield LOG_STREAM_DIV +html_content + "</div>" + scroll_js
        time.sleep(0.5)

def recordings_stream():
    while True:
        files=[]
        for r,_,f in os.walk(RECORDINGS_DIR):
            for x in f:
                if x.endswith(".mp4"):
                    files.append(os.path.join(r,x))

        files.sort(key=os.path.getmtime, reverse=True)

        html="""
        <div style="
            height: 200px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 5px;
            font-family: monospace;
            font-size: small;
            background-color: #1e1e1e;
            color: #ffffff;
            box-sizing: border-box;
        ">
            <div style="font-weight: bold; margin-bottom: 8px; font-size: medium;">
                🎥 Recordings
            </div>
        """
        for f in files:
            p = Path(f)
            html+=f'<a href="/gradio_api/file={f}" target="_blank" style="color: white;">{p.parent.name}/{p.name}</a><br>'
        html+="</div>"

        yield html
        time.sleep(2)



# =========================
# BUILD UI (RESTORED)
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## Portside Condominiums Security Cam Viewer")

    with gr.Accordion("Controls", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                confidence_threshold_slider = gr.Slider(0.1,0.9,value=CONFIDENCE_THRESHOLD,step=0.05,label="Confidence")
            with gr.Column(scale=1):
                motion_threshold_slider = gr.Slider(100,10000,value=MOTION_THRESHOLD,step=100,label="Motion")
            with gr.Column(scale=4):
                detection_classes = gr.CheckboxGroup(
                        choices=CLASSES_OF_INTEREST,
                        value=CLASSES_OF_INTEREST,
                        label="Objects"
                    )

        confidence_threshold_slider.change(update_confidence_threshold, confidence_threshold_slider)
        motion_threshold_slider.change(update_motion_threshold, motion_threshold_slider)
        detection_classes.change(update_detection_classes, detection_classes)

    outputs = []
    for i in range(0, len(CAMERAS), 5):
        with gr.Row():
            for name, url in CAMERAS[i:i+5]:
                with gr.Column():
                    annotated = gr.Image(label=f"{name}", streaming=True)
                    stats_box = gr.Textbox(
                        label=f"{name} Stats",
                        show_label=False,
                        interactive=False,
                        elem_classes="mono-textbox"
                    )
                    # Add HD Mode toggle button
                    hd_toggle = gr.Checkbox(label="HD Mode", value=False)
                    hd_toggle.change(fn=update_hd_mode, inputs=[gr.State(value=name), hd_toggle],  outputs=[])
                    outputs.append((annotated, stats_box, name, url))

    # recordings HTML
    recordings_box = gr.HTML(label="All Recordings")

    # Event log HTML
    log_box = gr.HTML(label="Event Log", value=LOG_STREAM_DIV+"</div>", elem_classes="scrollable-log")

    # --- LAUNCH STREAMS ---
    # Image streams
    for annotated, stats, name, url in outputs:
        demo.load(fn=make_stream_fn(name, url), inputs=None, outputs=[annotated, stats])
    # Recordings stream
    demo.load(fn=recordings_stream, inputs=None, outputs=recordings_box)
    # Event log stream
    demo.load(fn=log_stream, inputs=None, outputs=log_box)


if __name__ == "__main__":

    demo.launch(
        #server_name="0.0.0.0",
        theme=gr.themes.Soft(),
        allowed_paths=[RECORDINGS_DIR],
        css="""
.mono-textbox textarea {
    font-family: "Courier New", monospace !important;
    font-size: x-small !important;
}
""",
        )