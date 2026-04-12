import os
import glob
import threading
import time

from ffmpeg import FFmpeg

from constants import RECORDINGS_DIR, SEGMENTS_DIR, IMAGES_DIR, BUFFER_SECONDS
from logger import log_event

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(SEGMENTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# =========================
# NVR ENGINE
# =========================

# =========================
# START NVR
# =========================
def start_segment_recorders(cameras):
    for cam_id, url in cameras:
        cam_dir = os.path.join(RECORDINGS_DIR, cam_id)
        seg_dir = os.path.join(SEGMENTS_DIR, cam_id)
        img_dir = os.path.join(IMAGES_DIR, cam_id)
        os.makedirs(cam_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        start_segment_recorder(cam_id, seg_dir, url)
        threading.Thread(target=cleanup_segments,args=(cam_id, seg_dir),daemon=True).start()

def start_segment_recorder(cam_id, dir, rtsp_url):

    output = os.path.join(dir, "%Y%m%d_%H%M%S.ts")

    process = (
        FFmpeg()
        .option("y")
        .input(rtsp_url, rtsp_transport="tcp")
        .output(
            output,
            f="segment",
            segment_time=1,
            reset_timestamps=1,
            strftime=1,
            vcodec="copy",
            acodec="copy"
        )
    )

    threading.Thread(target=ffmpeg_thread, args=(cam_id, process,), daemon=True).start()

def ffmpeg_thread(cam_id, ffmpeg):
    threading.current_thread().name = f"{cam_id} ffmpeg"
    while True:
        try:
            log_event(f"segment recorder started", "info", cam_id)
            ffmpeg.execute()
            log_event(f"segment recorder after execute", "info", cam_id)
        except Exception as e:
            log_event(f"FFmpeg error: {e}", "error", cam_id)
            time.sleep(5)  # Wait before retrying

def cleanup_segments(cam_id, dir):
    threading.current_thread().name = "cleanup_segments"
    path = os.path.join(dir, cam_id, "*.ts")
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

def get_segments(cam_id, n):
    files = sorted(glob.glob(os.path.join(SEGMENTS_DIR, cam_id, "*.ts")))
    return files[-n:]

def merge_segments(files, output):
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
