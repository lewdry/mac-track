import os
import cv2
import time
from ultralytics import YOLO

# 1. Initialize YOLO26 Nano model
model = YOLO("yolov8n.pt")

# Explicitly target your MacBook's M1 GPU backend (Metal Performance Shaders)
model.to("mps")

# Ensure the 'output' directory exists in the current working directory
os.makedirs("output", exist_ok=True)

# Open Mac's default webcam
cap = cv2.VideoCapture(0)

# Video recording state variables
is_recording = False
video_writer = None

print("YOLOv8 Object Tracking Started on M1 GPU (Metal).")
print("-> Press SPACEBAR to take a photo.")
print("-> Press 'r' to START/STOP recording video.")
print("-> Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Mirror the video feed
    frame = cv2.flip(frame, 1)

    # 2. Run YOLO inference on the live frame
    results = model(frame, verbose=False)[0]

    # 3. Parse and draw detections manually to use your custom UI style
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        item_name = model.names[class_id]

        if confidence > 0.1:
            # Draw your signature cyberpunk yellow bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            label = f"{item_name} {confidence:.2f}"
            text_pos = (x1, max(y1 - 10, 20))
            
            cv2.putText(frame, label, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # If actively recording, overlay the red recording dot and write frame
    if is_recording and video_writer is not None:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        video_writer.write(frame)

    # Display live tracking feed
    cv2.imshow('YOLOv8 M1 Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    
    # Action A: Capture photo with bounding boxes
    if key == 32:
        # Save directly inside the relative 'output' directory
        photo_path = f"output/yolo_capture_{int(time.time())}.png"
        cv2.imwrite(photo_path, frame)
        print(f" Saved YOLO photo as: {photo_path}")
            
    # Action B: Toggle video recording
    elif key == ord('r'):
        if not is_recording:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Route video saving straight into the 'output' directory
            video_path = f"output/yolo_recording_{int(time.time())}.mp4"
            
            video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            is_recording = True
            print(f" STARTED recording: {video_path}")
        else:
            is_recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            print(" STOPPED recording and saved video file to output folder.")

    elif key == ord('q'):
        break

# Cleanup
if video_writer is not None:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()