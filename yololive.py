import os
import cv2
import time
import random
from ultralytics import YOLO

# 1. Initialize YOLOv8 models (Pose + Nano Object Detection)
pose_model = YOLO("yolov8n-pose.pt")
det_model = YOLO("yolov8n.pt") 

pose_model.to("mps")
det_model.to("mps")

os.makedirs("output", exist_ok=True)
cap = cv2.VideoCapture(0)

is_recording = False
video_writer = None
last_recorded_time = 0.0  # Tracks real-world time for synchronization

# Custom styling configurations (Ghostly / Glitchy)
OVERLAY_COLOR = (235, 235, 235)  # Pale silver/white mist
GLITCH_COLOR = (255, 215, 95)   # Pale electric blue/cyan peak
BASE_THICKNESS = 3
TEXT_SCALE = 0.8
ALPHA = 0.5     # Low base opacity for a translucent, ethereal look

# Symmetrical COCO keypoint index mapping for body limbs (excluding head)
BODY_CONNECTIONS = [
    (5, 6),   # Shoulder to shoulder
    (5, 7), (7, 9),    # Left arm
    (6, 8), (8, 10),   # Right arm
    (5, 11), (6, 12),  # Shoulders to hips
    (11, 12),          # Hip to hip
    (11, 13), (13, 15),# Left leg
    (12, 14), (14, 16) # Right leg
]

print("YOLOv8 Ghostly Custom Overlay Started.")
print("-> Press SPACEBAR to take a photo.")
print("-> Press 'r' to START/STOP recording video.")
print("-> Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    overlay = frame.copy()

    # Calculate unstable frame states for organic flickering
    current_alpha = max(0.2, min(0.8, ALPHA + random.uniform(-0.15, 0.15)))
    is_glitching = random.random() < 0.05  # 5% chance of a major tracking displacement glitch

    # 2. Run Live Inference
    det_results = det_model(frame, verbose=False)[0]
    pose_results = pose_model(frame, verbose=False)[0]

    # 3. Draw Object Detection (Fragmented Corner Frames)
    if det_results.boxes is not None:
        for box in det_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            item_name = det_model.names[class_id]

            if confidence > 0.15:
                offset_x = random.randint(-15, 15) if is_glitching else 0
                offset_y = random.randint(-5, 5) if is_glitching else 0
                
                color = GLITCH_COLOR if (is_glitching or random.random() < 0.1) else OVERLAY_COLOR
                thick = BASE_THICKNESS + 2 if is_glitching else BASE_THICKNESS
                
                cv2.line(overlay, (x1 + offset_x, y1 + offset_y), (x1 + 150 + offset_x, y1 + offset_y), color, thick)
                cv2.line(overlay, (x1 + offset_x, y1 + offset_y), (x1 + offset_x, y1 + 150 + offset_y), color, thick)
                cv2.line(overlay, (x2 + offset_x, y2 + offset_y), (x2 - 150 + offset_x, y2 + offset_y), color, thick)
                cv2.line(overlay, (x2 + offset_x, y2 + offset_y), (x2 + offset_x, y2 - 150 + offset_y), color, thick)
                
                label = f"{item_name.upper()} // {confidence:.2f}"
                cv2.putText(overlay, label, (x1 + offset_x, max(y1 - 8 + offset_y, 15)), 
                            cv2.FONT_HERSHEY_PLAIN, TEXT_SCALE, color)

    # 4. Draw Body Pose Estimation (Loose, Fragmented Skeleton)
    if pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
        for person in pose_results.keypoints:
            xy = person.xy[0].cpu().numpy()
            conf = person.conf[0].cpu().numpy() if person.conf is not None else [1.0] * 17

            if len(xy) > 0 and conf[0] > 0.4:
                x_nose, y_nose = map(int, xy[0])
                if x_nose > 0 or y_nose > 0:
                    box_size = 220
                    half_box = box_size // 2
                    top_left = (x_nose - half_box, y_nose - half_box)
                    bottom_right = (x_nose + half_box, y_nose + half_box)
                    nose_overlay = overlay.copy()
                    cv2.rectangle(nose_overlay, top_left, bottom_right, OVERLAY_COLOR, max(1, BASE_THICKNESS - 1))
                    cv2.addWeighted(nose_overlay, current_alpha, overlay, 1 - current_alpha, 0, overlay)

            for i in range(5, 17):
                if i < len(xy) and conf[i] > 0.4:
                    x, y = map(int, xy[i])
                    if x > 0 or y > 0:
                        radius = random.choice([2, 4, 5]) if is_glitching else 3
                        cv2.circle(overlay, (x, y), radius, OVERLAY_COLOR, -1)
                        box_size = 14
                        half_box = box_size // 2
                        top_left = (x - half_box, y - half_box)
                        bottom_right = (x + half_box, y + half_box)
                        cv2.rectangle(overlay, top_left, bottom_right, OVERLAY_COLOR, BASE_THICKNESS)

            for start_idx, end_idx in BODY_CONNECTIONS:
                if random.random() < 0.12:  
                    continue
                    
                if start_idx < len(xy) and end_idx < len(xy):
                    if conf[start_idx] > 0.4 and conf[end_idx] > 0.4:
                        pt1 = list(map(int, xy[start_idx]))
                        pt2 = list(map(int, xy[end_idx]))
                        
                        if pt1 != [0, 0] and pt2 != [0, 0]:
                            if is_glitching:
                                pt1[0] += random.randint(-12, 12)
                                pt2[1] += random.randint(-12, 12)
                            
                            cv2.line(overlay, tuple(pt1), tuple(pt2), OVERLAY_COLOR, BASE_THICKNESS)

    # 5. Blend the dynamic overlay layer with the original frame
    cv2.addWeighted(overlay, current_alpha, frame, 1 - current_alpha, 0, frame)

    # Real-time frame pace correction
    if is_recording and video_writer is not None:
        current_time = time.time()
        elapsed = current_time - last_recorded_time
        target_duration = 1.0 / 30.0  # Duration of one frame at 30 FPS
        
        # Calculate how many frames need to be written to match real time
        frames_to_write = int(elapsed / target_duration)
        
        if frames_to_write > 0:
            for _ in range(frames_to_write):
                video_writer.write(frame)
            last_recorded_time += frames_to_write * target_duration

        # Display recording indicator on the live preview screen only
        record_frame = frame.copy()
        cv2.circle(record_frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.imshow('YOLOv8 Ethereal Overlay', record_frame)
    else:
        cv2.imshow('YOLOv8 Ethereal Overlay', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 32:
        photo_path = f"output/ghost_capture_{int(time.time())}.png"
        cv2.imwrite(photo_path, frame)
        print(f" Saved photo as: {photo_path}")
    elif key == ord('r'):
        if not is_recording:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = f"output/ghost_recording_{int(time.time())}.mp4"
            video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            is_recording = True
            last_recorded_time = time.time()  # Initialize timing anchor
            print(f" STARTED recording: {video_path}")
        else:
            is_recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            print(" STOPPED recording.")
    elif key == ord('q'):
        break

if video_writer is not None:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()