from asyncio import constants
import json
import logging
import click
from click import version_option

from logger import setup_logging, log_event, event_log
from nvr import NVR
from context import Context
from model import Model
from gui import GUI

import constants
from constants import (
    CONFIDENCE_THRESHOLD_MIN,
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_THRESHOLD_MAX,
    MOTION_THRESHOLD_MIN,
    MOTION_THRESHOLD,
    MOTION_THRESHOLD_MAX,
)

logger = logging.getLogger("yolo-rtsp-security-cam")

@click.command(
        context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120},
        options_metavar="-d <directory> -u <apple-id> [options]",
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
@click.option("-c", "--config-file",
              required=True,
              help="NVR config file",
              metavar="<file>",
              default="cameras.json"
              )
@click.option("--logging-config",
              help="JSON logging config filename (default: logging-config.json)",
              metavar="<filename>",
              default="logging-config.json",
              show_default=True)
@click.option("--motion-threshold",
              help="threshold for motion detection (day, night)",
              type=click.Tuple([
                  click.IntRange(min=MOTION_THRESHOLD_MIN[0], max=MOTION_THRESHOLD_MAX[0]),
                  click.IntRange(min=MOTION_THRESHOLD_MIN[1], max=MOTION_THRESHOLD_MAX[1])
                  ]),
              metavar="<threshold>",
              default=(MOTION_THRESHOLD[0], MOTION_THRESHOLD[1]),
              show_default=True)
@click.option("--confidence-threshold",
              help="Confidence threshold for object detection",
              type=click.FloatRange(min=CONFIDENCE_THRESHOLD_MIN, max=CONFIDENCE_THRESHOLD_MAX),
              metavar="<threshold>",
              default=CONFIDENCE_THRESHOLD,
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
         config_file: str,
         logging_config: str,
         motion_threshold: tuple[int, int],
         confidence_threshold: float,
         motion_detect_frame_count: int,
         debug: bool,
         ) -> int:
    
    setup_logging("logging-config.json")

    logger.info(f"Application started with directory={directory} username={username} motion_threshold={motion_threshold} confidence_threshold={confidence_threshold} motion_detect_frame_count={motion_detect_frame_count}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    cameras = config['cameras']
    yolo = config['yolo']


    ctx = Context(
        directory=directory,
        username=username,
        password=password,
        logging_config=logging_config,
        motion_threshold=[motion_threshold[0], motion_threshold[1]],
        confidence_threshold=confidence_threshold,
        motion_detect_frame_count=motion_detect_frame_count,
        cameras=cameras,
        model='yolov8n.pt',
        classes=yolo['classes'],
        debug=debug,
    )

    nvr = NVR(ctx)
    nvr.start()

    model = Model(ctx)
    gui = GUI(ctx, model, nvr)
    gui.run()

if __name__ == "__main__":
    main()