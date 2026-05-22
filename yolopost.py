import os
import cv2
from ultralytics import YOLO

# 1. Initialize both YOLOv8 Nano models
# The library will automatically download 'yolov8n-pose.pt' and 'yolov8n.pt' on first run
pose_model = YOLO("yolov8n-pose.pt")
det_model = YOLO("yolov8n.pt")

# Run on Mac GPU (Metal Performance Shaders)
pose_model.to("mps")
det_model.to("mps")

# 2. Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, "input")
output_dir = os.path.join(base_dir, "output")

# Ensure directories exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Allowed video extensions
valid_extensions = (".mp4", ".avi", ".mov", ".mkv")
video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

if not video_files:
    print(f"No videos found in: {input_dir}")
    exit()

print(f"Found {len(video_files)} video(s) to process with YOLOv8.")

for video_file in video_files:
    input_path = os.path.join(input_dir, video_file)
    output_path = os.path.join(output_dir, f"processed_{video_file}")
    
    print(f"\nProcessing: {video_file}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_file}")
        continue
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference on both models
        pose_results = pose_model(frame, verbose=False)[0]
        det_results = det_model(frame, verbose=False)[0]
        
        # Plot object detection bounding boxes first
        frame = det_results.plot()
        
        # Overlay the skeletons onto the same frame
        # boxes=False prevents the pose model from drawing duplicate person boxes
        frame = pose_results.plot(img=frame, boxes=False)
        
        # Save frame to output video
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"Saved to: {output_path}")

print("\nAll videos processed successfully using YOLOv8.")