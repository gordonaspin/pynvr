import html
import os
from logging import getLogger
from pathlib import Path

import gradio as gr

from camera import Camera
import constants
from context import Context
from logger import log_event, event_log
from nvr import NVR

logger = getLogger("nvr")

class GUI:
    def __init__(self, ctx: Context, nvr: NVR):
        self.ctx = ctx
        self.classes = ctx.classes
        self.nvr = nvr

    def get_frame(self, camera: Camera):
        frame = camera.latest_frame

        if frame is None:
            return None, "Waiting..."

        return frame.copy(), camera.status
    
    # =========================
    # UI HANDLERS
    # =========================
    def update_confidence_threshold(self, val):
        self.nvr.confidence_threshold = val
        log_event(message=f"confidence updated → {val}")

    def update_day_motion_threshold(self, val):
        self.nvr.motion_threshold[0] = val
        log_event(message=f"day motion threshold → {val}")

    def update_night_motion_threshold(self, val):
        self.nvr.motion_threshold[1] = val
        log_event(message=f"night motion threshold → {val}")

    def update_detection_classes(self, names):
        self.nvr.selected_classes = self.nvr.model.class_to_index(names)
        log_event(message=f"classes → {names}")

    def update_hd(self, name, val):
        self.nvr.cameras[name].hd = val
        log_event(message=f"HD mode {"on" if val else "off"}", camera=self.nvr.cameras[name])

    def update_debug(self, val):
        self.nvr.debug = val
        log_event(message=f"Detections {"on" if val else "off"}")

    def update_debug_files(self, val):
        self.nvr.debug_files = val
        log_event(message=f"Debug files {"on" if val else "off"}")
    # =========================
    # UI STREAMS
    # =========================
    # --- Event log streamer ---
    def get_log_html(self):
        content = "".join(x for x in event_log[-constants.MAX_LOG_LINES:])

        return f"""
        <div style="
            height:300px;
            overflow-y:auto;
            border:1px solid #ccc;
            padding:5px;
            font-family:monospace;
            font-size:12px;
            background-color:#1e1e1e;
            color:#ffffff;
        ">
            <div style="font-weight:bold; margin-bottom:8px;">
                📜 Event Log
            </div>
            {content}
        </div>
        """

    def get_recordings_html(self):
        files = []

        for r, _, f in os.walk(self.nvr.recordings_dir):
            for x in f:
                if x.endswith(".mp4"):
                    files.append(os.path.join(r, x))

        try:
            files.sort(key=os.path.getmtime, reverse=True)
        except FileNotFoundError:
            return "<div>Loading...</div>"

        files = files[:constants.MAX_LOG_LINES]

        html_content = """
        <div style="
            height: 300px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 5px;
            font-family: monospace;
            font-size: 12px;
            background-color: #1e1e1e;
            color: #ffffff;
        ">
            <div style="font-weight: bold; margin-bottom: 8px; font-size: medium;">
                🎥 Recordings
            </div>
        """

        for f in files:
            p = Path(f)

            # escape display text (important for safety)
            label = html.escape(f"{p.parent.name}/{p.name}")
            label = f"{p.parent.name}/{p.name}"

            html_content += (
                f'<a href="/gradio_api/file={f}" '
                f'target="_blank" '
                f'style="color:white; text-decoration:underline;">{label}</a><br>'
            )

        html_content += "</div>"

        return html_content

    def on_load(self):
        log_event(f"A browser has connected to the app")

    def run(self, auth=True):
        # BUILD UI
        with gr.Blocks() as demo:
            gr.Markdown("## Portside Condominiums Security Cam Viewer")

            with gr.Accordion("Controls", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        confidence_threshold_slider = gr.Slider(label="Confidence",
                                                                minimum=constants.CONFIDENCE_THRESHOLD_MIN,
                                                                maximum=constants.CONFIDENCE_THRESHOLD_MAX,
                                                                value=self.ctx.confidence_threshold,
                                                                step=0.05,
                                                                )
                    with gr.Column(scale=1):
                        day_motion_threshold_slider = gr.Slider(label="Day Motion",
                                                                minimum=constants.MOTION_THRESHOLD_MIN[0],
                                                                maximum=constants.MOTION_THRESHOLD_MAX[0],
                                                                value=self.ctx.motion_threshold[0],
                                                                step=50,
                                                                )
                    with gr.Column(scale=1):
                        night_motion_threshold_slider = gr.Slider(label="Night Motion",
                                                                minimum=constants.MOTION_THRESHOLD_MIN[1],
                                                                maximum=constants.MOTION_THRESHOLD_MAX[1],
                                                                value=self.ctx.motion_threshold[1],
                                                                step=50,
                                                                )

                    with gr.Column(scale=4):
                        detection_classes = gr.CheckboxGroup(label="Objects",
                                                            choices=self.classes,
                                                            value=self.classes,
                                                            )
                    with gr.Column(scale=1):
                        debug_checkbox = gr.Checkbox(label="Show Detections", value=self.ctx.debug, elem_classes="custom-checkbox")
                        files_checkbox = gr.Checkbox(label="Produce Debug Images", value=self.ctx.debug_files, elem_classes="custom-checkbox")

                confidence_threshold_slider.change(self.update_confidence_threshold, confidence_threshold_slider)
                day_motion_threshold_slider.change(self.update_day_motion_threshold, day_motion_threshold_slider)
                night_motion_threshold_slider.change(self.update_night_motion_threshold, night_motion_threshold_slider)
                detection_classes.change(self.update_detection_classes, detection_classes)
                debug_checkbox.change(fn=self.update_debug, inputs=debug_checkbox,  outputs=[])
                files_checkbox.change(fn=self.update_debug_files, inputs=files_checkbox,  outputs=[])

            outputs = []
            for i in range(0, int(len(self.nvr.cameras)/5)):
                with gr.Row():
                    d = dict(list(self.nvr.cameras.items())[i*5:5+i*5])
                    for camera in d.values():
                        if camera.enabled:
                            with gr.Column():
                                annotated = gr.Image(label=f"{camera.name}", buttons=['fullscreen'])
                                stats_box = gr.Textbox(label=f"{camera.name} Stats",
                                                       show_label=False,
                                                       interactive=False,
                                                       elem_classes="mono-textbox"
                                )
                                # Add HD Mode toggle button
                                hd_checkbox = gr.Checkbox(label="HD Mode",
                                                          value=camera.hd,
                                                          elem_classes="custom-checkbox")
                                hd_checkbox.change(fn=self.update_hd, inputs=[gr.State(value=camera.name), hd_checkbox],  outputs=[])
                                outputs.append((annotated, stats_box, camera))

            with gr.Row():
                with gr.Column(scale=2):
                    # Event log HTML
                    log_box = gr.HTML(label="Event Log")
                    timer = gr.Timer(0.5)  # update every 0.5s
                    timer.tick(fn=self.get_log_html, outputs=log_box)

                with gr.Column(scale=1):
                    # recordings HTML
                    recordings_box = gr.HTML(label="All Recordings")
                    timer = gr.Timer(2.0)  # update every 2.0s
                    timer.tick(fn=self.get_recordings_html, outputs=recordings_box)

            timer = gr.Timer(1.0/20.0)
            
            # Image streams
            for annotated, stats, camera in outputs:
                timer.tick(
                    fn=lambda c=camera: self.get_frame(c),
                    inputs=None,
                    outputs=[annotated, stats]
                    )

            demo.load(fn=self.on_load)


        demo.queue()
        try:
            demo.launch(
                #share=True,
                auth=[self.ctx.username, self.ctx.password] if auth else None,
                server_name=self.ctx.bind_address,
                theme=gr.themes.Soft(),
                allowed_paths=[self.ctx.directory],
                css="""
                    .mono-textbox textarea {
                        font-family: "Courier New", monospace !important;
                        font-size: x-small !important;
                    }
                    .custom-checkbox span {
                    font-family: 'Courier New', monospace !important;
                        font-size: small !important;
                    }
                    .gradio-container > footer,
                    .gradio-container footer,
                    footer,
                    div:has(> .footer) {
                        display: none !important;
                    }
                    """,
                )
        except KeyboardInterrupt as e:
            logger.info("Shutting down on CTRL-C")
            demo.close()
