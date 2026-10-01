""" parse command line, set up NVR and FastAPI application """
from asyncio import CancelledError
import json
import logging
from pathlib import Path
import signal
import socket

import click
import uvicorn
from click import version_option

from pynvr.api.endpoints import create_app
from pynvr.config.config import SystemConfig
from pynvr.nvr import NVR

NVR_OBJ = None

def shutdown(signum, _):
    """ called upon OS signal, set stop_event """
    NVR_OBJ.stop_event.set()
    logger.info(f"caught signal {signum}, stop event is set")
    NVR_OBJ.stop()

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

logger = logging.getLogger("pynvr")

def socket_available(host, port):
    """ checks if socket is available """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            logger.error(f"socket {host}:{port} is already in use")
            return False
    return True

@click.command()
@click.option("-c", "--nvr-config", default="nvr.json")
@version_option()

# pylint: disable=too-many-branches, too-many-statements
def main(
    nvr_config
    ):
    """ main entrypoint """

    #pylint: disable=global-statement
    global NVR_OBJ

    nvr_config_path = Path(nvr_config)
    if not nvr_config_path.is_absolute():
        nvr_config_path = nvr_config_path.resolve()

    system_config_json = nvr_config_path.read_text(encoding="utf-8")
    system_config = SystemConfig.model_validate_json(system_config_json)
    with open(system_config.logging_config, encoding="utf-8") as f:
        logging_config = json.load(f)

    logger.info("starting pynvr")

    if not socket_available(system_config.bind_address, system_config.port):
        return

    NVR_OBJ = nvr = NVR(system_config)
    app = create_app(system_config, nvr)

    nvr.start()
    try:
        uvicorn.run(
            app,
            host=system_config.bind_address,
            port=system_config.port,
            log_config=logging_config,
            timeout_graceful_shutdown=0,
            access_log=False,
            workers=1,
            reload=False
            )
    except CancelledError:
        logger.debug("uvicorn server stopped")
    except Exception as e:
        logger.error(f"uvicorn server error {e}")

    logger.info("waiting on NVR threads to finish...")
    for thread in nvr.threads():
        logger.info(f"waiting on thread {thread.name}")
        thread.join()

    for handler in logger.handlers:
        handler.flush()
        handler.close()

    logger.info("done.")

if __name__ == "__main__":
    main()
