import atexit
import signal
import sys
import json
import re
import logging
from urllib.parse import urlparse, urlunparse

import click
from click import version_option

import constants
from context import Context
from gui import GUI

from logger import setup_logging, log_event, KeywordFilter
from nvr import NVR

_NVR = None

def shutdown(signum, frame):
    _NVR.stop_event.set()
    _NVR.stop()
    sys.exit()

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

logger = logging.getLogger("nvr")

def replace_url_credentials(url, new_username, new_password):
    parsed = urlparse(url)

    # Build new netloc
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""

    userinfo = ""
    if new_username is not None:
        userinfo = new_username
        if new_password is not None:
            userinfo += f":{new_password}"
        userinfo += "@"

    new_netloc = f"{userinfo}{hostname}{port}"

    # Reconstruct URL
    new_parsed = parsed._replace(netloc=new_netloc)
    return urlunparse(new_parsed)


@click.command(
        context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120},
        options_metavar="-d <directory> -u -p [options]",
        no_args_is_help=True)
@click.option("-d", "--directory",
              required=True,
              help="Local directory that should be used for storing recordings and logs",
              type=click.Path(exists=True),
              metavar="<directory>")
@click.option("-c", "--nvr-config",
              required=False,
              help="NVR config file",
              metavar="<file>",
              default="nvr.json")
@click.option("-u", "--username",
              required=False,
              help="NVR/Camera username, will override what is specified in the --nvr-config file",
              metavar="<username>")
@click.option("-p", "--password",
              required=False,
              help="NVR/Camera password, will override what is specified in the --nvr-config file",
              metavar="<password>")
@click.option("--gui-username",
              required=False,
              help="GUI authorization username",
              metavar="<GUI username>")
@click.option("--gui-password",
              required=False,
              help="GUI authorization password",
              metavar="<GUI password>")
@click.option("--bind-address",
              help="bind address for GUI",
              metavar="<ip address>",
              default="0.0.0.0", # all interfaces
              show_default=True)
@click.option("--logging-config",
              help="JSON logging config filename (default: logging-config.json)",
              metavar="<filename>",
              default="logging-config.json",
              show_default=True)
@click.option("--motion-threshold",
              help="percent change in pixels for motion detection",
              type=click.FloatRange(min=constants.MOTION_THRESHOLD_MIN, max=constants.MOTION_THRESHOLD_MAX),
              metavar="<threshold>",
              default=constants.MOTION_THRESHOLD,
              show_default=True)
@click.option("--confidence-threshold",
              help="Confidence threshold for object detection",
              type=click.FloatRange(min=constants.CONFIDENCE_THRESHOLD_MIN, max=constants.CONFIDENCE_THRESHOLD_MAX),
              metavar="<threshold>",
              default=constants.CONFIDENCE_THRESHOLD,
              show_default=True)
@click.option("--motion-detect-frame-count",
              help="number of frames with motion required to start recording",
              type=click.IntRange(min=2, max=100),
              metavar="<seconds>",
              default=constants.MOTION_DETECT_FRAME_COUNT,
              show_default=True)
@click.option("--debug",
              help="debug mode",
              is_flag=True)
@version_option()

# pylint: disable=too-many-branches, too-many-statements
def main(directory: str,
         username: str,
         password: str,
         gui_username: str,
         gui_password: str,
         nvr_config: str,
         bind_address: str,
         logging_config: str,
         motion_threshold: float,
         confidence_threshold: float,
         motion_detect_frame_count: int,
         debug: bool,
         ) -> int:
    
    global _NVR
    setup_logging(logging_config)
    if password is not None:
        KeywordFilter.add_keyword(password)

    logger.info(f"Application started with directory={directory} username={username} password={password} motion_threshold={motion_threshold} confidence_threshold={confidence_threshold} motion_detect_frame_count={motion_detect_frame_count}")

    with open(nvr_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    yolo_config = config['yolo']
    resolution = config['resolution']
    camera_config = config['cameras']
    if username and password:
        for camera in camera_config.values():
            camera['url'] = replace_url_credentials(camera['url'], username, password)   

    ctx = Context(
        directory=directory,
        username=username,
        password=password,
        gui_username=gui_username,
        gui_password=gui_password,
        camera_config=camera_config,
        bind_address=bind_address,
        motion_threshold=motion_threshold,
        confidence_threshold=confidence_threshold,
        motion_detect_frame_count=motion_detect_frame_count,
        resolution=resolution,
        model=yolo_config['model'],
        classes=yolo_config['classes'],
        debug=debug,
    )

    _NVR = nvr = NVR(ctx)
    nvr.start()
    atexit.register(nvr.stop)

    gui = GUI(ctx, nvr)
    gui.run()

if __name__ == "__main__":
    main()
