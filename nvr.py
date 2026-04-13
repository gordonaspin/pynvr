import os
import glob
import threading
import time

from ffmpeg import FFmpeg

from constants import BUFFER_SECONDS
from context import Context
from logger import log_event

# =========================
# NVR ENGINE
# =========================
class NVR:
    def __init__(self, ctx):
        self.cameras = ctx.cameras
        self.recordings_dir = ctx.directory
        self.segments_dir = os.path.join(self.recordings_dir, "segments")
        self.images_dir = os.path.join(self.recordings_dir, "images")

        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.segments_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)


    def start(self):
        for name, camera in self.cameras.items():
            if camera['enabled']:
                cam_dir = os.path.join(self.recordings_dir, name)
                seg_dir = os.path.join(self.segments_dir, name)
                img_dir = os.path.join(self.images_dir, name)
                os.makedirs(cam_dir, exist_ok=True)
                os.makedirs(seg_dir, exist_ok=True)
                os.makedirs(img_dir, exist_ok=True)
                self._start_segment_recorder(name, seg_dir, camera['url'])
                threading.Thread(target=self._cleanup_segments,args=(seg_dir,),daemon=True).start()

    def _start_segment_recorder(self, name, dir, url):

        filespec = os.path.join(dir, "%Y%m%d_%H%M%S.ts")

        process = (
            FFmpeg()
            .option("y")
            .input(url, rtsp_transport="tcp")
            .output(
                filespec,
                f="segment",
                segment_time=1,
                reset_timestamps=1,
                strftime=1,
                vcodec="copy",
                acodec="copy"
            )
        )

        threading.Thread(target=self._ffmpeg_thread, args=(name, process,), daemon=True).start()

    def _ffmpeg_thread(self, name, ffmpeg):
        threading.current_thread().name = f"{name} ffmpeg"
        while True:
            try:
                log_event(f"segment recorder started", "info", name)
                ffmpeg.execute()
                log_event(f"segment recorder after execute", "info", name)
            except Exception as e:
                log_event(f"FFmpeg error: {e}", "error", name)
                time.sleep(5)  # Wait before retrying

    def _cleanup_segments(self, dir):
        threading.current_thread().name = "cleanup_segments"
        path = os.path.join(dir, "*.ts")
        while True:
            try:
                files = sorted(glob.glob(path))
                if len(files) > BUFFER_SECONDS:
                    for f in files[:-BUFFER_SECONDS]:
                        try: os.remove(f)
                        except: pass
                time.sleep(1)
            except Exception as e:
                log_event(f"exception in cleanup_segments {e}", "error")

    def get_segments(self, name, n):
        files = sorted(glob.glob(os.path.join(self.segments_dir, name, "*.ts")))
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
