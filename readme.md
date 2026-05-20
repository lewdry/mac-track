
# Mac Track - Computer Vision Lab

Hello! This is local object detection for Apple Silicon (M1/M2/M3) Macs using MediaPipe and YOLOv8 via OpenCV.

## Quick Start Setup

Open your terminal, navigate to your cloned project folder, and run the following steps sequentially.

### 1. Environment Configuration

Create and activate a Python 3 virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install opencv-python mediapipe ultralytics
```

Download the MediaPipe EfficientDet model:

```bash
curl -L -o efficientdet.tflite "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
```

### 2. Run Object Tracking

**Option A: MediaPipe Object Tracking**

Runs a lightweight CPU tracking pipeline using Google's Tasks API.

```bash
python poseobj.py
```

**Option B: YOLOv8 M1 GPU Tracking**

Runs a highly accurate detection model accelerated natively on the Apple Silicon GPU via Metal Performance Shaders (MPS).

```bash
python yolo.py
```

---

## Keyboard Controls (Both Scripts)

Click onto the active camera window to focus your OS interface, then use these shortcuts:

- **SPACEBAR**: Capture .png snapshot with yellow tracking boxes.
- **r**: Toggle video recording. Starts/Stops writing live frames to mp4.
- **q**: Safely terminate the camera capture stream and close all tracking windows.