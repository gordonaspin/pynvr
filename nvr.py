import os
import glob
import threading
import time
import subprocess
import numpy as np


from ffmpeg import FFmpeg

import constants
from context import Context
from logger import log_event
from camera import Camera

# =========================
# NVR ENGINE
# =========================
class NVR:
    def __init__(self, ctx: Context):
        self.recordings_dir = ctx.directory
        self.width = ctx.resolution[0]
        self.height = ctx.resolution[1]
        self.segments_dir = os.path.join(self.recordings_dir, "segments")
        self.images_dir = os.path.join(self.recordings_dir, "images")
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        self.cameras = {}
        for name, cfg in ctx.camera_config.items():
            self.cameras[name] = Camera(name=name,
                                        url=cfg['url'],
                                        enabled=cfg['enabled'],
                                        recordings_dir=os.path.join(self.recordings_dir, name),
                                        segments_dir=os.path.join(self.segments_dir, name),
                                        images_dir=os.path.join(self.images_dir, name),
                                        )


    def start(self, restart=False):
        for camera in self.cameras.values():
            if camera.enabled:
                os.makedirs(camera.recordings_dir, exist_ok=True)
                os.makedirs(camera.segments_dir, exist_ok=True)
                os.makedirs(camera.images_dir, exist_ok=True)
                log_event(message=f"starting recorder", level="info", camera=camera)
                camera.process = self._start_segment_recorder(camera)
        if not restart:
            threading.Thread(target=self._cleanup_segments,daemon=True).start()

    def restart(self):
        for camera in self.cameras.values():
            if camera.enabled:
                ret = camera.process.poll()
                if ret is None:
                    log_event(message=f"stopping recorder", level="info", camera=camera)
                    camera.process.terminate()
                    camera.process.kill()
                    camera.process.wait()
                else:
                    log_event(message=f"recorder exited {ret}", level="error", camera=camera)
        self.start(True)

    def _start_segment_recorder(self, camera: Camera):

        filespec = os.path.join(camera.segments_dir, "%Y%m%d_%H%M%S.ts")
        log_file = open(f"{camera.name}_ffmpeg.log", "w")

        ffmpeg_cmd = [
            "ffmpeg",

            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-use_wallclock_as_timestamps", "1",
            "-i", camera.url,

            # Split + per-branch processing
            "-filter_complex",
            f"[0:v]split=2[v1][v2];"
            f"[v1]scale={self.width}:{self.height}[enc];"
            f"[v2]scale={self.width}:{self.height},format=bgr24[raw]",

            # ---- TS segments (encoded) ----
            "-map", "[enc]",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-g", "30",
            "-keyint_min", "30",
            "-sc_threshold", "0",
            "-f", "segment",
            "-segment_time", "1",
            "-reset_timestamps", "1",
            "-strftime", "1",
            "-segment_format", "mpegts",
            filespec,

            # ---- Raw frames (OpenCV) ----
            "-map", "[raw]",
            "-f", "rawvideo",
            "pipe:1"
        ]
        process =  subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=log_file,
            bufsize=10**8
        )

        return process

    def _cleanup_segments(self):
        threading.current_thread().name = "cleanup_segments"
            
        while True:
            try:
                for camera in self.cameras.values():
                    if camera.enabled:
                        path = os.path.join(camera.segments_dir, "*.ts")
                        files = sorted(glob.glob(path))
                        if len(files) > constants.BUFFER_SECONDS:
                            for f in files[:-constants.BUFFER_SECONDS]:
                                try: os.remove(f)
                                except: pass
                time.sleep(1)
            except Exception as e:
                log_event(message=f"exception in cleanup_segments {e}", level="error")

    def get_segments(self, camera: Camera, n: int):
        files = sorted(glob.glob(os.path.join(camera.segments_dir, "*.ts")))
        return files[-n:]

    def merge_segments(self, files, output):
        list_file = output + ".txt"
        with open(list_file,"w") as f:
            for x in files:
                f.write(f"file '{os.path.abspath(x)}'\n")

        (
            FFmpeg()
            .option("y")
            .input(list_file, f="concat", safe=0)
            .output(output, c="copy")
            .execute()
        )

        os.remove(list_file)

    def frame_generator(self, camera: Camera, width, height):
        import numpy as np
        import time

        frame_size = width * height * 3

        while True:
            frame = None
            raw = self.cameras[camera.name].process.stdout.read(frame_size)
            ok = len(raw) == frame_size

            if ok:
                frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
            yield ok, frame