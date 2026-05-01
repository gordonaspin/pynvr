from collections import defaultdict
from logging import getLogger
from datetime import datetime

import colorsys
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

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
        self.last_event_hash = None
        self.color_map = {}
        file = font_manager.findfont('Verdana', fontext='ttf')
        self.courier_font = ImageFont.truetype(file, 12)
        self.row_height = 40
        self.padding = 10
        self.scale_height = 30

        for i, s in enumerate(self.classes):
            hue = i / len(self.classes)  # evenly spaced
            r, g, b = colorsys.hsv_to_rgb(h=hue, s=0.65, v=0.95)

            self.color_map[s] = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )

    def get_frame(self, camera: Camera):
        frame = camera.latest_frame

        if frame is None:
            return None, "Waiting..."

        return frame.copy(), camera.status
    
    # =========================
    # UI HANDLERS
    # =========================
    def update_confidence_threshold(self, val):
        """ modifies the object detection confidence threshold of the NVR YOLO model """
        self.nvr.confidence_threshold = val
        log_event(message=f"confidence updated → {val}")

    def update_motion_threshold(self, val):
        """ modifies the motion detection threshold of the NVR """
        self.nvr.motion_threshold = val
        log_event(message=f"motion threshold → {val}")

    def update_detection_classes(self, names):
        """ updates the set of object class indexes to detect in the NVR """
        self.nvr.selected_classes = self.nvr.model.class_to_index(names)
        log_event(message=f"classes → {names}")

    def update_hd(self, name, val):
        """ updates the HD option for viewing the camera image """
        self.nvr.cameras[name].hd = val
        log_event(message=f"HD mode {'on' if val else 'off'}", camera=self.nvr.cameras[name])

    def update_camera_debug(self, name, val):
        """ updates the Debug option for the camera """
        self.nvr.cameras[name].debug = val
        log_event(message=f"Debug mode {'on' if val else 'off'}", camera=self.nvr.cameras[name])

    def update_debug(self, val):
        """" updates the debug option of the NVR """
        self.nvr.debug = val
        log_event(message=f"Detections {'on' if val else 'off'}")

    def update_debug_files(self, val):
        """ updates the debug_files option of the NVR """
        self.nvr.debug_files = val
        log_event(message=f"Debug files {'on' if val else 'off'}")

    # =========================
    # UI STREAMS
    # =========================
    def get_log_html(self):
        """ writes the HTML for the log view """
        content = "".join(x for x in event_log[-constants.MAX_LOG_LINES:])

        return f"""
        <div style="
            height:300px;
            overflow-y:auto;
            border:1px solid #ccc;
            padding:5px;
            font-family:monospace;
            font-size:xsmall;
            background-color:#1e1e1e;
            color:#ffffff;
        ">
            <div style="font-weight:bold; margin-bottom:8px;">
                📜 Event Log
            </div>
            {content}
        </div>
        """

    def get_height(self):
        """ computes the height of the timeline image based on the number of cameras """
        return self.scale_height + len(self.nvr.cameras) * self.row_height + self.padding * 2
    # ------------------------------------------------------------
    # Draw timeline as an image + return clickable regions
    # ------------------------------------------------------------
    def draw_timeline(self):

        def tag_colors(tags):
            colors = []
            if isinstance(tags, dict):
                for obj, _ in tags.items():
                    colors.append(self.color_map.get(obj, "#9E9E9E"))
            else:
                for tag in tags:
                    colors.append(self.color_map.get(tag[0], "#9E9E9E"))
            return colors

        def tag_label(tags):
            objects = [f"{obj}({color})" for obj, color in tags]
            return ", ".join(objects) if objects else "motion"
        
        grouped_events = self.nvr.load_events()

        hours = 4
        # Limit to past 4 hours
        now = datetime.now().timestamp()
        window_start = now - hours * 3600

        # Filter events to only those in the last 24 hours
        filtered = {}
        for cam, events in grouped_events.items():
            filtered_events = [
                e for e in events
                if e["end_time"] >= window_start
            ]
            if filtered_events:
                filtered[cam] = filtered_events

        if not filtered:
            # No events → return empty timeline
            img = Image.new("RGB", (900, 200), (31, 41, 55))
            return img, []

        grouped_events = filtered

        # Layout constants
        width = 900
        label_width = 100

        cameras = list(grouped_events.keys())
        height = self.get_height()

        img = Image.new("RGB", (width, height), (31, 41, 55))
        draw = ImageDraw.Draw(img)

        # Time span is fixed: 
        start = window_start
        end = now
        span = end - start

        clickable_regions = []  # (x1, y1, x2, y2, video_path)

        # ------------------------------------------------------------
        # Draw 24-hour time scale at the top
        # ------------------------------------------------------------
        scale_top = self.padding
        scale_bottom = scale_top + self.scale_height - 5
            
        # Background
        draw.rectangle([label_width, scale_top, width - 10, scale_bottom], fill="#2d3748")

        # Hour ticks
        for hour in range(hours + 1):  # 0..n hours
            t = start + hour * 3600
            x = label_width + int((t - start) / span * (width - label_width - 20))

            # Tick mark
            draw.line([(x, scale_top), (x, scale_bottom)], fill="#ffffff", width=1)

            # Label (24-hour clock)
            label = datetime.fromtimestamp(t).strftime("%H:%M")
            draw.text((x + 2, scale_top + 2), label, fill="white")

        for idx, camera in enumerate(cameras):
            y_top = scale_bottom + self.padding + idx * self.row_height
            y_bottom = y_top + self.row_height - 5

            # Draw camera label
            draw.text((10, y_top + 10), camera, font=self.courier_font,fill="white")

            # Draw timeline background
            draw.rectangle(
                [label_width, y_top + 5, width - 10, y_bottom],
                fill="#374151"
            )

            # Draw events
            for e in grouped_events[camera]:
                left = label_width + max(0, int((e["start_time"] - start) / span * (width - label_width - 20)))
                right = label_width + int((e["end_time"] - start) / span * (width - label_width - 20))

                colors = tag_colors(e["tags"])
                for i, color in enumerate(colors):
                    draw.rectangle([left, y_top + 5 + i * (y_bottom - y_top - 5) // len(colors), right, y_bottom], fill=color)

                metadata_str = f"<a href=\"/gradio_api/file={e['metadata']}\" target=\"_blank\">View</a>" if e.get("metadata") else "N/A"
                info_html = f"""
                <b>Camera:</b> {camera} | 
                <b>Tags:</b> {self.nvr._tags_to_str(e["tags"]) if isinstance(e["tags"], dict) else tag_label(e["tags"])} |
                <b>Start:</b> {datetime.fromtimestamp(e['start_time']).strftime('%Y-%m-%d %H:%M:%S')} - 
                <b>End:</b> {datetime.fromtimestamp(e['end_time']).strftime('%Y-%m-%d %H:%M:%S')}<br>
                <b>Metadata:</b> {metadata_str}
                """
                clickable_regions.append((left, y_top+5, right, y_bottom, e["output"], info_html))                # Tooltip

        for index, (cls, color) in enumerate(self.color_map.items()):
            draw.text((label_width + index * 80, y_bottom + 20), cls, font=self.courier_font, fill=color)

        return img, clickable_regions

    def handle_click(self, evt: gr.SelectData, regions):
        x, y = evt.index

        for (x1, y1, x2, y2, video, info_html) in regions:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return video, info_html

        return None, ""

    def on_load(self):
        """ called when the GUI loads for a client """
        log_event(f"A browser has connected")

    def run(self):
        # BUILD UI
        with gr.Blocks() as demo:
            gr.Markdown("## Portside Condominiums Security Cam Viewer")

            # Controls
            with gr.Accordion("Controls", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        confidence_threshold_slider = gr.Slider(label="Detection Confidence",
                                                                minimum=constants.CONFIDENCE_THRESHOLD_MIN,
                                                                maximum=constants.CONFIDENCE_THRESHOLD_MAX,
                                                                value=self.ctx.confidence_threshold,
                                                                step=0.05,
                                                                )
                    with gr.Column(scale=1):
                        motion_threshold_slider = gr.Slider(label="% Pixel Change in Motion",
                                                                minimum=constants.MOTION_THRESHOLD_MIN,
                                                                maximum=constants.MOTION_THRESHOLD_MAX,
                                                                value=self.ctx.motion_threshold,
                                                                step=0.1,
                                                                )

                    with gr.Column(scale=4):
                        detection_classes = gr.CheckboxGroup(label="Objects",
                                                            choices=self.classes,
                                                            value=self.classes,
                                                            )
                    with gr.Column(scale=1):
                        debug_checkbox = gr.Checkbox(label="Debug", value=self.ctx.debug, elem_classes="custom-checkbox")
                        files_checkbox = gr.Checkbox(label="Produce Debug Images", value=self.ctx.debug_files, elem_classes="custom-checkbox")

                confidence_threshold_slider.change(self.update_confidence_threshold, confidence_threshold_slider)
                motion_threshold_slider.change(self.update_motion_threshold, motion_threshold_slider)
                detection_classes.change(self.update_detection_classes, detection_classes)
                debug_checkbox.change(fn=self.update_debug, inputs=debug_checkbox,  outputs=[])
                files_checkbox.change(fn=self.update_debug_files, inputs=files_checkbox,  outputs=[])

            # Cameras
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
                                outputs.append((annotated, stats_box, camera))
                                with gr.Row():
                                    # Add HD Mode toggle button
                                    hd_checkbox = gr.Checkbox(label="HD Mode",
                                                            value=camera.hd,
                                                            elem_classes="custom-checkbox")
                                    hd_checkbox.change(fn=self.update_hd, inputs=[gr.State(value=camera.name), hd_checkbox],  outputs=[])
                                    camera_debug_checkbox = gr.Checkbox(label="Debug",
                                                            value=camera.debug,
                                                            elem_classes="custom-checkbox")
                                    camera_debug_checkbox.change(fn=self.update_camera_debug, inputs=[gr.State(value=camera.name), camera_debug_checkbox],  outputs=[])

            # recording timeline and playback
            with gr.Row():
                selected_video = gr.Textbox(visible=False)
                with gr.Column():
                    timeline_img = gr.Image(type="pil", interactive=False, label="Timeline", buttons=[], container=True)
                with gr.Column():
                    video_player = gr.Video(label="Selected Video", height=self.get_height(), autoplay=True, interactive=False)
                    event_info = gr.HTML(label="Event Info")

                # When selected_video changes, update video player
                selected_video.change(lambda x: x, selected_video, video_player)

                # Store clickable regions in a State object
                regions_state = gr.State([])

                # Clicking the image selects a video
                timeline_img.select(
                    fn=self.handle_click,
                    inputs=[regions_state],
                    outputs=[selected_video, event_info]
                )

                # Timer updates the timeline every 5 seconds
                def refresh():
                    img, regions = self.draw_timeline()
                    return img, regions

                timer = gr.Timer(5)
                timer.tick(fn=refresh, inputs=None, outputs=[timeline_img, regions_state])

                # Initial render
                img, regions = self.draw_timeline()
                timeline_img.value = img
                regions_state.value = regions

            # Event log HTML
            with gr.Row():
                log_box = gr.HTML()
                timer = gr.Timer(1.0)  # update every 0.5s
                timer.tick(fn=self.get_log_html, outputs=log_box)

            # Image streams
            timer = gr.Timer(1.0/20.0)            
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
                auth=[self.ctx.gui_username, self.ctx.gui_password] if all([self.ctx.gui_username, self.ctx.gui_password]) else None,
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
