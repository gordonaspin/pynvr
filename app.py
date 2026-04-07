import os
import cv2
import gradio as gr
from ultralytics import YOLO
import time
from datetime import datetime
import subprocess
import threading
from ffmpeg import FFmpeg

# --- Settings ---
MAX_LOG_LINES = 1000
YOLO_SIZE = (608, 416)
RENDER_SIZE = (int(YOLO_SIZE[0]/2), int(YOLO_SIZE[1]/2))  # for better visibility in UI
MOTION_FRAME_COUNT = 2
TAIL_FRAME_COUNT = 8
CONFIDENCE_THRESHOLD = 0.4
MOTION_THRESHOLD = 500

LOG_STREAM_DIV = """
    <div class="inner-log" style="
        height: 300px; 
        overflow-y: auto; 
        border: 1px solid #ccc; 
        padding: 5px; 
        font-family: monospace; 
        background-color: #1e1e1e; 
        color: #ffffff;
        box-sizing: border-box;
    ">
    <div style="font-weight: bold; margin-bottom: 8px; font-size: 16px;">
        📜 Event Log
    </div>
    """

RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

RTSP_ENTRIES = [
    ("Shed", f"rtsp://admin:Portside123!!!@shed.portsidecondominium.com:554/cam/realmonitor?channel=1&subtype=1"),
    ("B56Lot", f"rtsp://admin:Portside123!!!@shed.portsidecondominium.com:554/cam/realmonitor?channel=2&subtype=1"),
    ("B5Lot", f"rtsp://admin:Portside123!!!@shed.portsidecondominium.com:554/cam/realmonitor?channel=3&subtype=1"),
    ("Pool", f"rtsp://admin:Portside123!!!@pool.portsidecondominium.com:554/cam/realmonitor?channel=1&subtype=1"),
    ("PoolEntry", f"rtsp://admin:Portside123!!!@pool.portsidecondominium.com:554/cam/realmonitor?channel=2&subtype=1")
]
# Tracks live HD mode per camera
hd_mode_states = {name: False for name, _ in RTSP_ENTRIES}

# Load YOLO
model = YOLO("yolov8n.pt")
CLASSES_OF_INTEREST = ["person", "car", "bicycle", "motorcycle", "bus", "truck", "cat", "dog"]
CLASS_TO_IDX = {v: k for k, v in model.names.items()}
INTERESTED_CLASSES = [CLASS_TO_IDX[i] for i in CLASSES_OF_INTEREST]
selected_classes = INTERESTED_CLASSES.copy()

# Persistent storage
event_log = []
recorded_files = []  # store gr.File components for download

# --- Logging ---
def log_event(message, level="info", camera="", file_path=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    colors = {"info": "#00c853", "warn": "#ffd600", "error": "#ff5252", "record": "#ff1744"}
    color = colors.get(level, "#ffffff")

    prefix = f"{camera:<9}" if camera else ""

    # Make recording file clickable
    if file_path and os.path.isfile(file_path):
        filename = os.path.basename(file_path)
        link = f'<a href="/gradio_api/file={file_path}" target="_blank" style="color:#40c4ff;">{filename}</a>'
        message = f"{message} {link}"

    entry = f'<div style="color:{color}; font-family:monospace;">[{timestamp}] {level.upper():<8}: {prefix}{message}</div>'
    event_log.append(entry)

    if len(event_log) > MAX_LOG_LINES:
        event_log.pop(0)

# --- Confidence slider handler ---
def update_conf(val):
    global CONFIDENCE_THRESHOLD
    log_event(f"Confidence threshold updated from {CONFIDENCE_THRESHOLD} to {val}", "info")
    CONFIDENCE_THRESHOLD = val

# --- Motion factor slider handler ---
def update_motion_factor(val):
    global MOTION_THRESHOLD
    log_event(f"Motion factor updated from {MOTION_THRESHOLD} to {val}", "info")
    MOTION_THRESHOLD = val

# --- Class selection handler ---
def update_classes(selected_names):
    global selected_classes

    if not selected_names:
        selected_classes = []  # detect nothing
        log_event("No classes selected", "warn")
        return

    selected_classes = [CLASS_TO_IDX[name] for name in selected_names]
    log_event(f"Classes updated: {', '.join(selected_names)}", "info")

# --- HD Mode toggle handler ---
def update_hd_mode(camera_name, hd_value):
    global hd_mode_states
    hd_mode_states[camera_name] = hd_value
    log_event(f"HD Mode {'enabled' if hd_value else 'disabled'}", camera=camera_name)

# --- Event log streamer ---
def log_stream():
    while True:
        html_content = "".join(event_log)
        # JS snippet to scroll to bottom
        scroll_js = "<script>var el=document.currentScript.parentElement; el.scrollTop=el.scrollHeight;</script>"
        yield LOG_STREAM_DIV +html_content + scroll_js + "</div>"
        time.sleep(0.5)

# --- Recordings streamer ---
def recordings_stream():
    while True:
        files = []
        if os.path.exists(RECORDINGS_DIR):
            # Get full paths
            files = [
                os.path.join(RECORDINGS_DIR, f)
                for f in os.listdir(RECORDINGS_DIR)
                if f.endswith(".mp4")
            ]
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        file_links = ""
        for full_path in files:
            filename = os.path.basename(full_path)
            link = f'<a href="/gradio_api/file={full_path}" target="_blank" style="color:#40c4ff;">{filename}</a>'
            file_links += f"{link}<br>"

        yield f"""
        <div style="
            height: 200px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 5px;
            font-family: monospace;
            background-color: #1e1e1e;
            color: #ffffff;
            box-sizing: border-box;
        ">
            <div style="font-weight: bold; margin-bottom: 8px; font-size: 16px;">
                🎥 Recordings
            </div>
            {file_links}
        </div>
        """
        time.sleep(2)

# --- FFmpeg recording thread ---
def start_ffmpeg(ffmpeg_copy, cam_id):
    try:
        ffmpeg_copy.execute()
    except:
        log_event(f"Issue recording the stream. Trying again.", "error", cam_id)
        time.sleep(1)
        ffmpeg_copy.execute()

def start_ffmpeg_recording(rtsp_url, filename, cam_id, stop_event):
    """
    Runs FFmpeg in a thread and stops when stop_event is set.
    """
    ffmpeg_copy = (
            FFmpeg()
            .option("y")
            .input(
                rtsp_url,
                rtsp_transport="tcp",
                rtsp_flags="prefer_tcp",
            )
            .output(filename, vcodec="copy", acodec="copy")
        )

    ffmpeg_thread = threading.Thread(target=start_ffmpeg, args=(ffmpeg_copy, cam_id), daemon=True)
    ffmpeg_thread.start()

    # Wait until stop signal
    stop_event.wait()

    # Stop FFmpeg cleanly
    ffmpeg_copy.terminate()
    try:
        ffmpeg_thread.join(timeout=5)
    except Exception as e:
        log_event(f"Exception {e} waiting for recording thread.", "error", cam_id)

# --- Camera stream ---
def make_stream_fn(cam_id, rtsp_url):
    async def stream():
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        #cap.set(cv2.CAP_PROP_PROTOCOL, 'udp') 
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Set buffer size to 1 frame

        if not cap.isOpened():
            log_event("Failed to open stream", "error", cam_id)
            raise gr.Error(f"Failed to open stream: {rtsp_url}")

        prev_gray = None
        motion_counter = 0
        no_motion_counter = 0
        recording = False
        filename = None
        ffmpeg_thread = None
        stop_event = None
        prev_time = time.time()
        cam_fps = cap.get(cv2.CAP_PROP_FPS) or 20
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        log_event(f"Stream opened with resolution {resolution} {cam_fps:.1f}fps", "info", cam_id)

        while True:
            ret, frame = cap.read()
            if not ret:
                log_event("Stream read failed, attempting to reconnect...", "warn", cam_id)
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                continue

            resized = cv2.resize(frame, YOLO_SIZE)

            # YOLO detection
            results = model.predict(resized, classes=selected_classes if selected_classes else None, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            annotated = results.plot()
            det_count = len(results.boxes) if results.boxes is not None else 0

            object_classes_in_frame = []
            for data in results.boxes.data.tolist():
                # Get the bounding box coordinates, confidence, and class id
                _xmin, _ymin, _xmax, _ymax, confidence, class_id = data
                object_name = model.names[int(class_id)]
                if object_name not in object_classes_in_frame:
                    object_classes_in_frame.append(object_name)
                        
            # Motion detection
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            motion_detected = False
            if prev_gray is not None:
                frame_diff = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
                motion_score = cv2.countNonZero(thresh)
                #log_event(f"Motion score: {motion_score} {('with detected' + ', '.join(object_classes_in_frame)) if object_classes_in_frame else ''}", "info", cam_id)
                if motion_score > MOTION_THRESHOLD:
                    motion_detected = True
                    log_event(f"Motion detected (score: {motion_score}){(' with detected ' + ', '.join(object_classes_in_frame)) if object_classes_in_frame else ''}", "warn", cam_id)
            prev_gray = gray

            if motion_detected:
                motion_counter += 1
                no_motion_counter = 0
            else:
                no_motion_counter += 1

            # Start recording
            if motion_counter >= MOTION_FRAME_COUNT and not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(RECORDINGS_DIR, f"{cam_id}_{timestamp}_{motion_score}{('_' + '_'.join(object_classes_in_frame)) if object_classes_in_frame else ''}.mp4")
                stop_event = threading.Event()

                ffmpeg_thread = threading.Thread(
                    target=start_ffmpeg_recording,
                    args=(rtsp_url, filename, cam_id, stop_event),
                    daemon=True
                )
                ffmpeg_thread.start()
                recording = True
                log_event(f"Recording started {os.path.basename(filename)}", "record", cam_id)

            # Stop recording
            if recording and no_motion_counter >= TAIL_FRAME_COUNT:
                recording = False
                motion_counter = 0
                no_motion_counter = 0
                if stop_event:
                    stop_event.set()

                if ffmpeg_thread:
                    ffmpeg_thread.join(timeout=5)
                log_event(f"Recording stopped {os.path.basename(filename)}", "record", cam_id)

                recorded_file = cv2.VideoCapture(filename)
                recorded_frames = recorded_file.get(cv2.CAP_PROP_FRAME_COUNT)
                if recorded_frames < TAIL_FRAME_COUNT + cam_fps and os.path.isfile(filename):
                    os.remove(filename)
                    log_event(f"Auto-deleted {os.path.basename(filename)} with {recorded_frames} frames", "record", cam_id, file_path=filename)
                else:
                    log_event(f"Recording available {os.path.basename(filename)}", "record", cam_id, file_path=filename)

            # FPS stats
            fps = 1.0 / (time.time() - prev_time)
            prev_time = time.time()
            stats_text = f"FPS: {fps:.1f} | Detections: {det_count}"

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            hd_mode = hd_mode_states.get(cam_id, False)
            if not hd_mode:
                annotated = cv2.resize(annotated, RENDER_SIZE)
            else:
                pass
            yield annotated, stats_text

    return stream

# --- Build UI ---
with gr.Blocks() as demo:
    gr.Markdown("## Portside Condominiums Security Cam Viewer")
    with gr.Accordion("Controls", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                # Confidence slider
                confidence_threshold_slider = gr.Slider(0.1, 0.9, value=CONFIDENCE_THRESHOLD, step=0.05, label="Confidence Threshold")
                confidence_threshold_slider.change(update_conf, inputs=[confidence_threshold_slider], outputs=[])
            with gr.Column(scale=1):
                # motion threshold
                motion_threshold_slider = gr.Slider(100, 10000, value=MOTION_THRESHOLD, step=100, label="Motion Score (px)")
                motion_threshold_slider.change(update_motion_factor, inputs=[motion_threshold_slider], outputs=[])
            with gr.Column(scale=4):
                class_selector = gr.CheckboxGroup(
                    choices=CLASSES_OF_INTEREST,
                    value=CLASSES_OF_INTEREST,  # default = all selected
                    label="Objects to Detect"
            )
            class_selector.change(fn=update_classes, inputs=[class_selector], outputs=[])

    outputs = []
    for i in range(0, len(RTSP_ENTRIES), 5):
        with gr.Row():
            for name, url in RTSP_ENTRIES[i:i+5]:
                with gr.Column():
                    #original = gr.Image(label=name)
                    annotated = gr.Image(label=f"{name}")
                    stats_box = gr.Textbox(
                        label=f"{name} Stats",
                        show_label=False,
                        interactive=False
                    )
                    # Add HD Mode toggle button
                    hd_toggle = gr.Checkbox(label="HD Mode", value=False)
                    hd_toggle.change(fn=update_hd_mode, inputs=[gr.State(value=name), hd_toggle],  outputs=[])
                    outputs.append((annotated, stats_box, name, url))
                    #outputs.append((original, annotated, stats_box, name, url))

    # recordings HTML
    recordings_box = gr.HTML(label="All Recordings")

    # Event log HTML
    log_box = gr.HTML(label="Event Log", value=LOG_STREAM_DIV+"</div>", elem_classes="scrollable-log")

    # Launch streams
    #for original, annotated, stats, name, url in outputs:
    #    demo.load(make_stream_fn(name, url), inputs=None, outputs=[original, annotated, stats])
    for annotated, stats, name, url in outputs:
        demo.load(make_stream_fn(name, url), inputs=None, outputs=[annotated, stats])

    # Recordings stream
    demo.load(
        recordings_stream,
        inputs=None,
        outputs=recordings_box
    )
    # Event log stream
    demo.load(log_stream, inputs=None, outputs=log_box)

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        allowed_paths=[f"./{RECORDINGS_DIR}/"],
       )