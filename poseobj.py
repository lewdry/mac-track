import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Initialize the Modern Object Detector
model_path = 'efficientdet.tflite'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5  # Display items with > 30% confidence
)
detector = vision.ObjectDetector.create_from_options(options)

# Open Mac's default webcam
cap = cv2.VideoCapture(0)

# Video recording state variables
is_recording = False
video_writer = None

print("Modern multi-object tracking started.")
print("-> Press SPACEBAR to take a photo.")
print("-> Press 'r' to START/STOP recording video.")
print("-> Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Mirror the video feed
    frame = cv2.flip(frame, 1)
    
    # Format the image correctly for the modern API
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run inference
    detection_result = detector.detect(mp_image)

    # 2. Draw boxes and extracted confidence numbers
    for detection in detection_result.detections:
        # Get bounding box points
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        
        # Draw yellow bounding box
        cv2.rectangle(frame, start_point, end_point, (0, 255, 255), 2)
        
        # Pull real label and confidence math from the model
        category = detection.categories[0]
        item_name = category.category_name
        confidence = category.score
        
        # Label string (e.g., "person (0.83)")
        label = f"{item_name} ({confidence:.2f})"
        text_pos = (int(bbox.origin_x), int(bbox.origin_y - 10))
        
        # Render text overlay
        cv2.putText(frame, label, text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # If currently recording, add a small red dot overlay to the preview screen 
    # and write the frame to the video file
    if is_recording and video_writer is not None:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        video_writer.write(frame)

    # Display video
    cv2.imshow('Upgraded Object Detection', frame)

    # Listen for keypresses
    key = cv2.waitKey(5) & 0xFF
    
    # Action A: Press Spacebar (Key code 32) to take a photo
    if key == 32:
        filename = f"capture_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f" Saved tracking photo as: {filename}")
            
    # Action B: Press 'r' to toggle video recording
    elif key == ord('r'):
        if not is_recording:
            # Get video layout specifics from your live camera stream
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video encoding properties (mp4v works universally on MacOS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_filename = f"recording_{int(time.time())}.mp4"
            
            # Initialize writer at 20 frames per second
            video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (width, height))
            is_recording = True
            print(f" STARTED recording: {video_filename}")
        else:
            # Safely release stream to finalize the movie file file-structure
            is_recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            print(" STOPPED recording and saved video file.")

    # Action C: Press 'q' to quit
    elif key == ord('q'):
        break

# Clean up remaining open processes
if video_writer is not None:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()