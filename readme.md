# Mac Track — Computer Vision Lab

Local object detection for Apple Silicon (M1/M2/M3) Macs using MediaPipe and YOLOv8 via OpenCV.

<img width="600" height="336" alt="Screen Recording 2026-05-20 at 11 16 46 pm (1)-dithered" src="https://github.com/user-attachments/assets/04ca8f22-891f-4c4f-b3cd-225704721fa9" />

<img width="600" height="338" alt="emojihead_recording_1780020619-dithered" src="https://github.com/user-attachments/assets/3a6e5637-a0a5-4fd9-a3de-72f0ece79f8f" />

<img width="600" height="338" alt="processed_VID_20170916_173146-dithered" src="https://github.com/user-attachments/assets/0c954ade-89a2-46d8-91f5-d5be2fc299c9" />

<img width="600" height="643" alt="processed_c82053f1-b511-4ef2-94ac-4463c16a3956-dithered" src="https://github.com/user-attachments/assets/d552cb58-d94b-4d50-9306-cfaa55a24363" />

## Setup

Open a terminal, navigate to your project folder, and run:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install opencv-python mediapipe ultralytics
```

Then download the MediaPipe EfficientDet model:

```bash
curl -L -o efficientdet.tflite "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
```

## Scripts

| Script | Description |
|---|---|
| `yolo.py` | YOLOv8 object detection on your webcam, GPU-accelerated on Apple Silicon. Supports photo and video capture. |
| `yololive.py` | Live webcam feed with ghostly/glitchy pose and object detection overlays. |
| `yolopost.py` | Batch processes all videos in `input/`, adds object and pose overlays, and saves results to `output/`. |
| `pose.py` | Real-time human pose (skeleton) tracking via MediaPipe. |
| `poseobj.py` | Real-time multi-object tracking via MediaPipe. |
| `emojihead.py` | Overlays an emoji on up to four detected faces in your webcam feed. |

## Examples - Running

**MediaPipe** — lightweight CPU pipeline using Google's Tasks API:
```bash
python poseobj.py

or 

python emojihead.py
```

**YOLOv8** — high-accuracy detection accelerated on Apple Silicon GPU via Metal Performance Shaders:
```bash
python yolo.py

or 

python yololive.py
```

## Keyboard Controls

| Key | Action |
|---|---|
| `SPACE` | Capture a PNG snapshot with tracking boxes |
| `r` | Start / stop video recording |
| `q` | Quit and close all windows |
