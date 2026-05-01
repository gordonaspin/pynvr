# pynvr - a Python Network Video Recorder

`pynvr` is a capable NVR that records from IP camera streams over RTSP.

## Overview
`pynvr` uses ffmpeg to read RTSP streams. Each stream has its own ffmpeg subprocess that reads the stream and simultaneously writes to segment files and stdout. The segment files are not re-encoded, and the stdout output stream frames are converted to OpenCV2 format and resized to a frame size defined in the nvr.json config file. pynvr starts a thread per camera to read frames from the stdout stream and puts the latest frame to a per-camere queue. pynvr starts a second thread per camera to process the frame from the queue. Frame processing determines motion and object identificaation. When thresholds are met, recording is started. After a period of no motion, the recording is stopped and pynvr joins the segments together, re-encoding them to H.264.

`pynvr` has a user interface with controls to adjust thresholds and objects to be detected. pynvr is a server process that does not need a client to attach. The GUI implementation uses Gradio to render frames from the cameras. The GUI presents rows of up to 5 cameras per row, a log window and and list of hyperlinks to recordings.
## Architecture / Design
`pynvr` implements an efficient, robust pipeline to stream per cameraa for motion and object detection and recording file creation. The pipeline is as follows:
```code
                         +---------------------------------------+
                         |     ffmpeg reader sub-process         +
                         | writes to mpeg segment files          +
                         | writes raw downsized frames to stdout +
                         +---------------------------------------+
                                            |
                                            v
                         +---------------------------------------+
                         |         frame reader thread           +
                         | reads frames from stdout              +
                         | enqueues frame, lastest frame wins    +
                         +---------------------------------------+
                                            |
                                            v
                         +---------------------------------------+      +--------------------------------------+
                         |       frame processor thread          +      +            GUI threads               +
                         | gets frame from queue                 + <--- + uses frame from memory               +
                         | detects motion and objects.           +      + renders frame to GUI                 +
                         +---------------------------------------+      +--------------------------------------+
                                            |
                                            v
                         +---------------------------------------+
                         |     ffmpeg merge sub-process          +
                         | asynchronously merges mpeg segment    +
                         | files and reencodes to H264 MP4 file  +
                         +---------------------------------------+

```
## Installation
Clone the repo
```bash
git clone http://github.com/gordonaspin/pynyr.git
```
Install the required python libraries
```bash
pip install -r requirements.txt
```
## Usage - Command line options
#### -d | --directory path/to/folder
Required argument of a path to a pre-existing folder to write recordings to.
#### -c | --nvr-config filename.json
Optional argument to specify the NVR configuration file. Default is nvr.json. See config section for JSON format spec.
#### -u username -p password
If supplied `pynvr` will apply those credentials to the RTSP urls specified in your NVR config file. If not supplied, `pyvr` will use the credentials from the RTSP URLs in the NVR config file.
#### --gui-username username --gui_password password
If supplied `pynvr` will apply these credentials to the GUI and will present a login challenge that accepts these credentials only.
```bash
python app.py -d <recordings folder> -u rtsp-username -p rtsp-password
```
## Config
Configuration is provided in a nvr.json file. "downsize_resolution" specifies the [x, y] dimensions in pixels to resize frames to for YOLO processing and rendering on the GUI. "yolo.model" specifies the name of the YOLO model to use. "yolo.classes" is an array of coco classes of objects to detect in the image processing. Each camera is named and specifies the URL and a boolean to set enabled/disabled.
```json
{
    "resolution": [704, 480],
    "yolo": {
        "classes": ["person", "car", "truck", "bus", "bicycle", "motorcycle", "cat", "dog"],
        "model": "yolov8n.pt"
    },
    "cameras": {
        "Cam1": {
            "url": "rtsp://username:password@hostname:554/cam/realmonitor?channel=3&subtype=1",
            "enabled": true
        },
        "Cam2": {
            "url": "rtsp://username:password@hostname:554/cam/realmonitor?channel=4&subtype=1",
            "enabled": true
        },
        "Cam3": {
            "url": "rtsp://username:password@hostname:554/cam/realmonitor?channel=5&subtype=1",
            "enabled": true
        },
        "Cam4": {
            "url": "rtsp://username:password@hostname:554/cam/realmonitor?channel=2&subtype=1",
            "enabled": true
        },
        "Cam5": {
            "url": "rtsp://username:password@hostname:554/cam/realmonitor?channel=1&subtype=1",
            "enabled": true
        }
    }
}
```
### Detailed usage
```bash
Usage: app.py -d <directory> -u -p [options]

Options:
  -d, --directory <directory>     Local directory that should be used for storing recordings and logs  [required]
  -c, --nvr-config <file>         NVR config file  [required]
  -u, --username <username>       NVR/Camera username, will override what is specified in the --nvr-config file
  -p, --password <password>       NVR/Camera password, will override what is specified in the --nvr-config file
  --gui-username <GUI username>   GUI authorization username
  --gui-password <GUI password>   GUI authorization password
  --bind-address <ip address>     bind address for GUI  [default: 0.0.0.0]
  --logging-config <filename>     JSON logging config filename (default: logging-config.json)  [default: logging-
                                  config.json]
  --motion-threshold <threshold>  threshold for motion detection (day, night)  [default: 500, 4000]
  --confidence-threshold <threshold>
                                  Confidence threshold for object detection  [default: 0.3; 0.1<=x<=0.9]
  --motion-detect-frame-count <seconds>
                                  number of frames with motion required to start recording  [default: 40; 2<=x<=100]
  --debug                         debug mode, produces .jpg files of motion contours
  --version                       Show the version and exit.
  -h, --help                      Show this message and exit.
```
### Logging
`pynvr` uses python logging to write to log files, configured by the logging-config.json file. The password passed in on the command line is filtered from logs. `pynvr` also produces log files per camera attached to each of the ffmpeg sub-processes and a log file per recording merge.