"""
    Types used in the API.
    These are Pydantic models that define the structure
    of the data returned by the API endpoints.
"""
from pathlib import Path
from typing import Literal, Union
from annotated_types import Ge, Le, MultipleOf
from pydantic import BaseModel

class ConfigValue(BaseModel):
    """ Class representing a configuration slider """
    default: float
    minimum: float
    maximum: float
    step: float
    value: float = None

    #pylint: disable=arguments-differ
    def model_post_init(self, __context) -> None:
        """ validate """
        self.value = self.default
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError(
                f"default {self.default} is out of range [{self.minimum}, {self.maximum}]")

    @staticmethod
    def from_config(default, model_cls, model_field_name) -> "ConfigValue":
        """ Create a ConfigValue from a Pydantic model field """
        def get_constraints(model_cls, field_name):
            field = model_cls.model_fields[field_name]
            result = {}
            for item in field.metadata:
                if isinstance(item, Ge):
                    result["min"] = item.ge
                elif isinstance(item, Le):
                    result["max"] = item.le
                elif isinstance(item, MultipleOf):
                    result["step"] = item.multiple_of

            return result

        return ConfigValue(
            default=default,
            minimum=get_constraints(model_cls, model_field_name)["min"],
            maximum=get_constraints(model_cls, model_field_name)["max"],
            step=get_constraints(model_cls, model_field_name)["step"]
        )

class CameraResponse(BaseModel):
    """
    Represents the response for a camera."""
    name: str
    debug: bool

class CameraDebugResponse(BaseModel):
    """
    Represents the response for camera debug information."""
    status: str
    camera: str
    debug: bool

class CameraSettingsResponse(BaseModel):
    """
    Represents the response for camera settings."""
    yolo_confidence: ConfigValue
    track_threshold: ConfigValue
    match_threshold: ConfigValue
    track_buffer: ConfigValue
    minimum_relative_motion: ConfigValue
    classes: dict[str, bool]

class ClassesResponse(BaseModel):
    """
    Represents the response for classes."""
    classes: dict[str, bool]

class SystemNameResponse(BaseModel):
    """
    Represents the response for system name."""
    system_name: str

class DimensionsResponse(BaseModel):
    """
    Represents the response for mosaic dimensions."""
    rows: int
    columns: int
    width: int
    height: int

class RecordingEvent(BaseModel):
    """
    Represents a recording event.
    """
    camera: str
    tags: dict[str, list[str]]
    media_filename: Path
    start_time: float
    end_time: float | None
    start_fmt: str
    end_fmt: str | None
    metadata_filename: Path
    recorder_type: str

class EventsResponse(BaseModel):
    """
    Represents the response for events."""
    events: list[RecordingEvent]

class LogEntry(BaseModel):
    """
    Represents a log entry."""
    timestamp: float
    level: str
    message: str
    file_path: Path | None
    anchor: Path | None

class LogsResponse(BaseModel):
    """
    Represents the response for logs."""
    log_entries: list[LogEntry]

class CameraSettingResponse(BaseModel):
    """
    Represents the response for a camera setting."""
    status: str
    camera: str
    setting: str
    value: float | bool

class LoginForm(BaseModel):
    """
    Represents the login form data.
    """
    username: str
    password: str

class SettingValue(BaseModel):
    """
    Represents a setting value for camera or system configuration.
    """
    value: float | bool

class SettingValueResponse(BaseModel):
    """
    Represents the response for a setting value update.
    """
    status: str
    value: float | bool

class ClassToggle(BaseModel):
    """
    Represents a class toggle for camera or system configuration.
    """
    class_name: str
    value: bool

class ClassToggleResponse(BaseModel):
    """
    Represents the response for a class toggle operation.
    """
    status: str
    camera: str
    class_name: str
    value: bool

class ServerTimeResponse(BaseModel):
    """
    Represents the response for server time.
    """
    epoch: float

class CameraStatus(BaseModel):
    """
    Represents the status of a camera.
    """
    ts: float
    name: str
    state: str
    state_value: int
    objects_dict: dict[str, list[str]]
    night: bool
    recording: bool
    read_fps: int
    record_fps: int

SSEEventType = Literal["cameraStatus", "logLine", "newEvent"]

class SSEEvent(BaseModel):
    """
    Represents a Server-Sent Event (SSE) for streaming updates to clients.
    """
    type: SSEEventType
    data: Union[CameraStatus, LogEntry, RecordingEvent]

    def to_sse(self) -> str:
        """
        Converts the SSEEvent to a string formatted for Server-Sent Events (SSE)."""
        return f"event: {self.type}\ndata: {self.model_dump_json()}\n\n"
