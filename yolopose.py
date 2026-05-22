import os
import cv2
import time
from ultralytics import YOLO

# 1. Initialize YOLOv8 Pose model 
model = YOLO("yolov8n-pose.pt")
model.to("mps")  # Run on Mac GPU

# Ensure the 'output' directory exists in the current working directory
os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(0)
is_recording = False
video_writer = None

print("YOLOv8 Skeleton Tracking Started.")
print("-> Press SPACEBAR to take a photo.")
print("-> Press 'r' to START/STOP recording video.")
print("-> Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    # 2. Run Pose Inference
    results = model(frame, verbose=False)[0]

    # 3. Let YOLO draw the skeleton automatically
    frame = results.plot(boxes=False)

    # If actively recording, overlay the red recording dot and write frame
    if is_recording and video_writer is not None:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        video_writer.write(frame)

    # Display live tracking feed
    cv2.imshow('YOLOv8 Skeleton Pose', frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Action A: Capture photo with skeleton
    if key == 32:
        # Save directly inside the relative 'output' directory
        photo_path = f"output/pose_capture_{int(time.time())}.png"
        
        cv2.imwrite(photo_path, frame)
        print(f" Saved skeleton photo as: {photo_path}")
            
    # Action B: Toggle video recording
    elif key == ord('r'):
        if not is_recording:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Route video saving straight into the 'output' directory
            video_path = f"output/pose_recording_{int(time.time())}.mp4"
            
            video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            is_recording = True
            print(f" STARTED recording: {video_path}")
        else:
            is_recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            print(" STOPPED recording and saved video file.")

    elif key == ord('q'):
        break

if video_writer is not None:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()