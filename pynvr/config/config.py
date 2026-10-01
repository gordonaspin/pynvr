"""
Configuration module for Pynvr.
This module defines the configuration settings for the Pynvr application.
It uses Pydantic models to validate and manage configuration data.
The configuration includes settings for system, cameras, processor, model, and more."""
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, Field, field_validator, model_validator

import keyring
from passlib.context import CryptContext

from pynvr.logger import setup_logging, KeywordFilter

def replace_url_credentials(url, new_username, new_password):
    """ parses RTSP url and replaces username password """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""

    userinfo = ""
    if new_username is not None:
        userinfo = new_username
        if new_password is not None:
            userinfo += f":{new_password}"
        userinfo += "@"

    new_netloc = f"{userinfo}{hostname}{port}"
    new_parsed = parsed._replace(netloc=new_netloc)
    return urlunparse(new_parsed)

class Resolution(BaseModel):
    """
    Resolution configuration settings.
    """
    width: int = Field(default=1920)
    height: int = Field(default=1080)

    def is_valid(self) -> bool:
        """ Check if the resolution is valid (non-zero width and height) """
        return self.width > 0 and self.height > 0

class MosaicConfig(BaseModel):
    """
    Mosaic configuration settings.
    """
    rows: int = Field(default=4)
    columns: int = Field(default=4)
    resolution: Resolution = Field(default_factory=lambda: Resolution(width=3840, height=1046))

class RecorderConfig(BaseModel):
    """
    Recorder configuration settings.
    """
    startup_delay: int = Field(default=15)
    pre_duration: int = Field(default=3)
    post_duration: int = Field(default=3)

class ModelConfig(BaseModel):
    """
    Model configuration settings.
    """
    name: Path = Field(default=Path("./pynvr/model/yolo11n.pt"))
    resolution: Resolution = Field(default_factory=lambda: Resolution(width=1920, height=1080))
    classes: dict[str, bool] = Field(default_factory=lambda: {
        "person": True,
        "car": True,
        "truck": True,
        "bus": True,
        "cat": False,
        "dog": True,
        "bicycle": True,
        "motorcycle": True
        })

class ProcessorConfig(BaseModel):
    """
    Processor configuration settings.
    """
    detect_every_nth_frame: int = Field(default=1)
    device: str = Field(default="cuda")
    night_check_period: int = Field(default=5)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)

class CameraConfig(BaseModel):
    """
    Camera configuration settings.
    """
    enabled: bool = Field(default=True)
    url: str = Field(default="rtsp://username:password@hostname.com/stream")
    resolution: Resolution = Field(default_factory=lambda: Resolution(width=704, height=480))
    recorder: str = Field(default="FFmpegFrame")
    yolo_confidence: float = Field(default=0.4, ge=0.1, le=1.0, multiple_of=0.1)
    track_threshold: float = Field(default=0.35, ge=0.1, le=1.0, multiple_of=0.01)
    match_threshold: float = Field(default=0.4, ge=0.1, le=1.0, multiple_of=0.01)
    track_buffer: int = Field(default=120, ge=30, le=300, multiple_of=1)
    minimum_relative_motion: float = Field(default=0.08, ge=0.05, le=0.2, multiple_of=0.01)
    render_annotations: str = Field(default="always")
    debug: bool = Field(default=False)


class SystemConfig(BaseModel):
    """
    System configuration settings.
    """
    system_name: str = Field(default="Pynvr")

    username: str = Field(default="admin")
    password: str = Field(default="password://undefined")
    gui_username: str = Field(default=None)
    gui_password: str = Field(default=None)

    recordings_directory: Path = Field(default=Path("recordings"))
    logs_directory: Path = Field(default=Path("logs"))

    logging_config: Path = Field(default=Path("logging-config.json"))

    keep_recordings_timedelta: timedelta = Field(default=timedelta(days=7))
    keep_logs_timedelta: timedelta = Field(default=timedelta(days=7))

    bind_address: str = Field(default="0.0.0.0")
    port: int = Field(default=7860)

    mosaic: MosaicConfig = Field(default_factory=MosaicConfig)

    model: ModelConfig = Field(default_factory=ModelConfig)

    processor: ProcessorConfig = Field(default_factory=ProcessorConfig)

    cameras: dict[str, CameraConfig] = Field(default_factory=lambda: {
        "camera1": CameraConfig(),
    })

    debug: bool = Field(default=False)

    @field_validator('keep_recordings_timedelta', 'keep_logs_timedelta', mode='before')
    @classmethod
    def parse_timedelta_dict(cls, value: Any) -> Any:
        """
        Validates and converts a dictionary input into a timedelta object.
        This allows for flexible configuration where users can specify time durations
        in a dictionary format (e.g., {"days": 7, "hours": 12}) instead of a
        timedelta object directly.
        """
        # If the input is a dictionary, unpack it into timedelta
        if isinstance(value, dict):
            return timedelta(**value)
        return value

    @model_validator(mode='after')
    def run_custom_after_logic(self) -> 'SystemConfig':
        """
        Runs custom validation logic after the model is initialized.
        """
        self.logs_directory = setup_logging(self.logging_config)

        # Read password from keyring if it starts with "password://"
        if self.password.startswith("password://"):
            self.password = keyring.get_password(self.password, self.username)

        if self.gui_username is None:
            self.gui_username = self.username

        if self.gui_password is None:
            self.gui_password = self.password

        if self.gui_password.startswith("password://"):
            self.gui_password = keyring.get_password(self.gui_password, self.gui_username)

        # Passwords are added to the KeywordFilter for logging purposes
        KeywordFilter.add_keyword(self.password)
        KeywordFilter.add_keyword(self.gui_password)

        try:
            hashed_gui_password = CryptContext(
                schemes=["bcrypt"],
                deprecated="auto").hash(self.gui_password)
        except AttributeError:
            pass

        self.gui_password = hashed_gui_password

        for camera in self.cameras.values():
            camera.url = replace_url_credentials(camera.url, self.username, self.password)

        return self
