# pynvr - a Python Network Video Recorder (NVR)

`pynvr` is a capable NVR that records from IP camera streams over RTSP.

## Overview
`pynvr` is an intelligent NVR that records video when motion is detected for certain classes of objects such as cars, trucks, people etc.
`pynvr` uses ffmpeg to read RTSP video streams, each stream is read in the native format and resolution output by the camera stream. The video stream from the camera is read by `pynvr` and processed using YOLO object detection. Detected objects are tracked frame by frame and if moving, `pynvr` creates a recording with metadata.

Each camera has its own ffmpeg subprocess that reads the stream and simultaneously writes frames to stdout and optionally FFmpeg segment files. The segment files are not re-encoded. The stdout output stream frames are converted to OpenCV2/bgr24 format and resized to a YOLO-suitable frame size defined in the pynvr.json config file. `pynvr` starts a thread per camera to read frames from the stdout stream and puts the latest frame to a per-camera queue. `pynvr` starts a second thread per camera to process the frame from the queue. Frame processing determines object identification and tracks object motion. When thresholds are met, recording is started. After a period of no motion, the recording is stopped and `pynvr` writes frames to a video file or joins the segments together, encoding to H.264 in an mp4/m4v container.

`pynvr` is a server process that does not need a client to attach. The GUI implementation is svelte components and Javascript. The `pynvr` GUI has controls to adjust motion-detection thresholds per camera. The GUI presents a mosaic of all enabled cameras, in the dimensions defined in nvr.json. If there are less cameras than rows * colums, black frames will be inserted. Clicking on a frame in the mosaic will zoom in to that camera, and clicking on a zoomed in camera will return to the mosaic.

The GUI presents a clickable timeline of events. Clicking the event will display the recorded video and associated metadata. The timeline can be panned back in history as far as the oldest event. The timeline can also be zoomed by holding the Shift-key and mouse-scroll. On mobile, the pan and zoom is performed by one finger and two-finger scroll, respectively.

## Installation
Clone the repo
```bash
git clone http://github.com/gordonaspin/pynyr.git
```
Install the required python libraries
```bash
pip install -r requirements.txt
```
Install the python build module
```bash
pip install build
```
Install nodejs and npm, if needed
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```
Run the python build
```bash
python -m build
```
Generate the OpenAPI spec
```bash
python generate_api.py
```
Install frontend project dependencies
```bash
cd frontend
npm install
```
Build the frontend GUI application
```bash
npm run build
```
Install external dependencies
```bash
sudo apt install ffmpeg
```
## Usage - Command line options

#### -c | --config pynvr.json
Optional argument to specify the PYNVR configuration file. Default is pynvr.json. See config section for JSON format spec.

```bash
python pynvr/app.py -c pynvr.json
```
## Docker container usage
`pynvr` can be built as a docker image, using the provided script which you customize to your needs.
```bash
build.docker.sh
```
The configuration files, pynvr.json and pynvr-logging-config.json are not built into the container. The `pynvr` application and gui frontend are in the container. It is necessary to expose port 7860 to the host and map a host folder for the configuration files and recordings folder., e.g.:
```bash
docker run -it -d --restart=always --network host --ipc=host --gpus all --name pynvr -v path/to/a/folder:/home/docker dockerimage <pynvr-options>
```
If you wish to use your host keyring from inside the container, you can map the container DBUS to your host DBUS. The id of the docker user is 1000 and must match your id. Change the USER_UID and GROUP_GID in the Docker file to match yours.
```bash
docker run --privileged -it -d  --restart always --network host --ipc=host --gpus all -v /run/user/$(id -u):/run/user/$(id -u) -e DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus --name pynvr -v /home/gordon/Development/pynvr:/home/docker gordonaspin/pynvr:latest
```

## Config
`pynvr` is configuration-driven. Most configuration values have sensible defaults which can be seen in pynvr/config/config.py. Those with defaults can be omitted from the config file.
Some configuration items are UI-accessible as commented here.
```json
{
    "system_name": "Security Cameras",                  // The system name is displayed on the UI for branding
    "recordings_directory": "recordings",               // path to folder for recordings and metadata file creation
    "username": "admin",                                // username used to authenticate to cameras
    "password": "password://undefined",                 // password used to authenticate to camera.
                                                        // if the value starts with "password://", the remainder of the value will be
                                                        // retrieved from the python keyring using "password://value"
                                                        // e.g. "password://someusername" results in a call keyring.get_password("password://value")
    "gui_username": "admin",                            // optional username for UI, if not specified it defaults to username for cameras
    "gui_password": "somepassword",                     // optional password for UI, if not specified it defaults to password for cameras with same keyring behavior
    "keep_recordings_timedelta": {                      // how long to keep recordings - as a python timedelta
        "days": 3
    },
    "keep_logs_timedelta": {                            // how long to keep logs
        "days": 1
    },
    "bind_address": "0.0.0.0",                          // network address to bind the UI server
    "port": 7860,                                       // network port the UI server listens on
    "ffmpeg_timeout": 60,                               // timeout used to manage/retry slow network camera connections
    "logging_config": "pynvr-logging-config.json",      // name of the logging config file
    "debug": true,                                      // debug mode on/off, UI-accessible

    "mosaic": {                                         // configuration of the video mosaic in the UI
        "rows": 2,
        "columns": 5,
        "resolution": {                                 // resolution of the mosaic
            "width": 3840,
            "height": 1046
        }
    },
    "model": {                                          // YOLO model configuration
        "name": "pynvr/model/yolo11n.pt"                // path to YOLO model file
        "resolution": {                                 // resolution to scale camera frames to for object detection
            "width": 640,
            "height": 640
        },
        "classes": {                                    // names of coco object class names to detect, with enable/disable flag, UI-accessible
            "person": true,
            "car": true,
            "truck": true,
            "bus": true,
            "cat": false,
            "dog": true,
            "bicycle": true,
            "motorcycle": true
            },
    },
    "processor": {                                      // Frame processor configuration
        "detect_every_nth_frame": 1,                    // how often to run YOLO, adjust according to device and CPU utilization
        "device": "cuda",                               // What device to use for YOLO, cuda for NVDIA, mps for Apple Silicon, cpu for cpu only
        "night_check_period": 5,                        // period in minutes to check for day or night
        "recorder": {                                   // recorder configuration
            "startup_delay": 15,                        // period in seconds to suppress object detection and recording after start-up 
            "pre_duration": 3,                          // pre-roll in seconds to record when motion detected
            "post_duration": 3                          // post-roll in seconds to continue recording when motion stopped
        }
    },
    "cameras": {                                        // camera configurations
        "C1": {                                         // name of camera and camera configuration
            "enabled": true,                            // enabled, true/false
            "url": "rtsp://username:password@hostname.com:554/cam/realmonitor?channel=3&subtype=1",
                                                        // RTSP url for camera. The username:password values will be replaced with those provided earlier
                                                        // The example here is for a Dahua camera second video stream 
            "resolution": {                             // resolution configuration for the stream. pynvr will probe for the actual camera resolution
                                                        // and use these values only as a fallback.
                "width": 704,
                "height": 480
            },
            "recorder": "FFmpegFrame",                  // type of recorder to use, can be FFmpegSegment or FFmpegFrame or OpenCVFrame or AVFFmpegFrame
            "yolo_confidence": 0.4,                     // YOLO confidence threshold for object detection, UI-accessible, range 0.1-1.0, step 0.01
            "track_threshold": 0.35,                    // Used to create motion tracks, UI-accessible, range 0.1-1.0, step 0.01
            "match_threshold": 0.4,                     // Used to match tracks, UI-accessible, range 0.1-1.0, step 0.01
            "track_buffer": 120,                        // number of frames to keep tracks, UI-accessible, range 30-300, step 1
            "minimum_relative_motion": 0.08,            // threshold for motion detection, UI-accessible, range 0.05-0.2, step 0.01
            "render_annotations": "always",             // draw boxes for detected objects, "always" (default), "never" or "motion"
            "debug": false                              // render debug panel for camera, UI-accessible
        },
        "C2": {
            "enabled": true,
            "url": "rtsp://username:password@hostname.com:554/cam/realmonitor?channel=4&subtype=1",
            "resolution": {
                "width": 704,
                "height": 480
            },
            "recorder": "FFmpegFrame",
            "yolo_confidence": 0.4,
            "track_threshold": 0.35,
            "match_threshold": 0.4,
            "track_buffer": 120,
            "minimum_relative_motion": 0.08, 
            "_render_annotations_comment": "never or motion",
            "render_annotations": "always",
            "debug": false
        },
        ...
    }
}



```
### Detailed usage
```bash
Usage: app.py --config pynvr.json

Options:
  -c, --nvr-config <file>         NVR config file  [defaults to nvr.json]
  --version                       Show the version and exit.
  -h, --help                      Show this message and exit.
```
### Video Frames
`pynvr` displays yolo boxes when there is motion that is of a recognized object. `pynvr` displays the current state of recording and fps information on the top left of the frame in the UI.

### Recordings and Metadata
`pynvr` creates recordings using ffmpeg .ts segment files or frames. `pynvr` creates .json metadata for each recording which details the start/stop time, the objects captured, etc. With the exception of segment recording, `pynvr` yolo detections drawn on the frame are saved in recordings.

### Recorder types
`pynvr` has 4 recorder types:
- FFmpegSegment
- FFmpegFrame
- OpenCVFrame
- AVFFmpegFrame.
#### FFMpegSegment
The FFMpegSegment recorder makes recordings from the segment files produced by FFMpeg. It is a recording of what the camera saw without any overlays indicating objects seen moving.
#### FFmpegFrame, OpenCVFrame and AVFFmpegFrame
FFmpegFrame, OpenCVFrame and AVFFmpegFrame are functionally equivalent, just using different technologies. Each records frames that include overlays of object classes detected and YOLO markup etc. The FFMpegFrame recorder passes frames at their recorded pace to FFmpeg over stdin. The OpenCVFrame recorder uses OpenCV to create the mp4 video file followed by an FFMpeg transformation to make the mp4 streamable. The AVFFmpegFrame recorder uses PyAV to create the recording from the frames, which uses FFMpeg under the covers.

### Logging
`pynvr` uses python logging to write to log files, configured by the pynvr-logging-config.json file. Passwords are filtered from logs. `pynvr` also produces log files per camera attached to each of the ffmpeg sub-processes and a log file per recording merge.

## Architecture / Design
`pynvr` implements an efficient, robust pipeline per camera for motion and object detection and recording file creation. The pipeline is as follows:
```code
                         +---------------------------------------+
                         |     ffmpeg reader sub-process         |
                         | writes to mpeg segment files          |
                         | writes YOLO-sized frames to a pipe    |
                         | writes raw frames to stdout           |
                         | (only writes to stdout if YOLO and    |
                         | camera resolutions match)             |
                         +---------------------------------------+
                                            |
                                            v
                         +---------------------------------------+
                         |         frame reader thread           |
                         | reads frames from stdout (and pipe)   |
                         | enqueues frame, lastest frame wins    |
                         +---------------------------------------+
                                            |
                                            v
                         +---------------------------------------+      +--------------------------------------+      +--------------------------------------+
                         |       frame processor thread          |      |      FastAPI uvicorn application.    |      | Svelte UI and WebRTC GUI uses frame  |
                         | gets frame from queue                 | <--- | hosts REST endpoints, serves static  | <--- | from server and renders frame to GUI |
                         | detects objects, tracks motion        |      | files, sends SSE events to browser   |      | receives realtime events from server |
                         +---------------------------------------+      +--------------------------------------+      +--------------------------------------+
                                            |
                                            v
                         +---------------------------------------+
                         |              recorders                |
                         | asynchronously merge mpeg segment     |
                         | files or frames and create MP4        |
                         | recording, create metadata            |
                         +---------------------------------------+

```
### Technologies used
`python` is used to create the pynvr server and all the video processing with the exception of ffmpeg.
`FastAPI and uvicorn` is used as the API to the server, by default it listens on port 7860.
`ffmpeg` is used to connect to IP cameras using the url configured for the camera in nvr.json. `ffprobe` is used to detect the native resolution of the stream if possible, and that resolution will be used to stream video to the user-interface and make recordings. If the native resolution of the camera cannot be determined by ffprobe, the fallback is the configured resolution. If the resolution is different from the YOLO resolution, then a second rawvideo is output from ffmpeg at the configured resolution will be used for object detection.
`YOLO` is used for object detection and classification. The expected model is yolo11n, and can be changed in the configuration file. YOLO processing will use the GPU for processing, if available and the configured processor.device is not "cpu". If "cpu" is specified, the processor.detect_every_nth_frame should be adjusted upwards to every 3rd or 4th frame, depending on your CPU.
`OpenCV` is used for frame manipulation, blurring, grayscaling etc. as needed to detect and measure motion
`PyAV` is used (with ffmpeg under the covers) to make recordings from frames.
`SvelteKit` (typescript/javascript) is used to create the user interface with Javascript, HTML markup and CSS.
`WebRTC` is used to stream video content to the Mosaic.svelte component in the browser.
`SSE or Server Side Events` is used to push updates to the browser. It is used for camera status displayed as a <div> overlay on the camera image in the Mosaic.svelte component. It is also used to push server logs to the browser EventLog component and recording event metadata to the browser that are subsequently fed to the EventInfo.svelte component and the MediaPlayer component.

### Troubleshooting
`pynvr` writes logs of everything it does, including ffmpeg logs on a per-camera basis. Logging is configured in the pynvr-logging-config file and follows the python logging dict config model. Network failures connecting to cameras and reading RTSP streams are handled with sensible retries.

Some modem routers use an Application Layer Gateway to perform packet inspection on RTSP traffic. I have found this to be troublesome if enabled as it burdens the modem router CPU and impacts traffic flow. My recommendation is to disable the ALG features of your modem router. If you cannot disable it, or your ISP does not permit it, then move the RTSP network traffic from port 554 to something like port 8554. This will typically bypass any triggering of the ALG module in the modem router. This would mean moving your cameras or NVRs to service RTSP connections on port 8554 and opening up port 8554 on your modem router to map to the camera on the internal network (if needed).

In my community, we have 3 distinct NVR systems, each with a number of cameras. These NVR systems are internet-connected via their own modem router. In this scenario, I run pynvr in the cloud and it connects through each modem on port 8554 and each modem router uses port forwarding to connect to the NVR that is behind it also on port 8554. This technique completely bypasses the ALG "feature" imposed by our ISP on the modem routers.