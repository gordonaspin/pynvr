from asyncio import constants
import json
import logging
import click
from click import version_option
from urllib.parse import urlparse, urlunparse
 
from logger import setup_logging, log_event, event_log
from nvr import NVR
from context import Context
from model import Model
from gui import GUI
from camera import Camera
import constants

logger = logging.getLogger("portside-nvrs")

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
@click.option("-u", "--username",
              required=True,
              help="NVR username",
              metavar="<username>")
@click.option("-p", "--password",
              required=True,
              help="NVR password",
              metavar="<password>")
@click.option("-c", "--nvr-config",
              required=True,
              help="NVR config file",
              metavar="<file>",
              default="nvr.json"
              )
@click.option("--logging-config",
              help="JSON logging config filename (default: logging-config.json)",
              metavar="<filename>",
              default="logging-config.json",
              show_default=True)
@click.option("--motion-threshold",
              help="threshold for motion detection (day, night)",
              type=click.Tuple([
                  click.IntRange(min=constants.MOTION_THRESHOLD_MIN[0], max=constants.MOTION_THRESHOLD_MAX[0]),
                  click.IntRange(min=constants.MOTION_THRESHOLD_MIN[1], max=constants.MOTION_THRESHOLD_MAX[1])
                  ]),
              metavar="<threshold>",
              default=(constants.MOTION_THRESHOLD[0], constants.MOTION_THRESHOLD[1]),
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
         nvr_config: str,
         logging_config: str,
         motion_threshold: tuple[int, int],
         confidence_threshold: float,
         motion_detect_frame_count: int,
         debug: bool,
         ) -> int:
    
    setup_logging(logging_config)

    logger.info(f"Application started with directory={directory} username={username} motion_threshold={motion_threshold} confidence_threshold={confidence_threshold} motion_detect_frame_count={motion_detect_frame_count}")

    with open(nvr_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    yolo = config['yolo']
    resolution = config['resolution']
    camera_config = config['cameras']
    for camera in camera_config.values():
        camera['url'] = replace_url_credentials(camera['url'], username, password)    

    ctx = Context(
        directory=directory,
        username=username,
        password=password,
        camera_config=camera_config,
        motion_threshold=[motion_threshold[0], motion_threshold[1]],
        confidence_threshold=confidence_threshold,
        motion_detect_frame_count=motion_detect_frame_count,
        resolution=resolution,
        model='yolov8n.pt',
        classes=yolo['classes'],
        debug=debug,
    )

    model = Model(ctx)
    logger.info(f"Model was trained with image size: {model.model.args['imgsz']}")

    nvr = NVR(ctx, model)
    nvr.start()

    gui = GUI(ctx, nvr)
    gui.run()

if __name__ == "__main__":
    main()