import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Set up the modern MediaPipe Task options
model_path = 'pose_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)

detector = vision.PoseLandmarker.create_from_options(options)

# Open webcam (using index 1 for Mac built-in camera based on your setup)
cap = cv2.VideoCapture(0)
print("Tracking started with bounding boxes. Click the window and press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect poses
    detection_result = detector.detect(mp_image)

    # Draw the tracked elements
    if detection_result.pose_landmarks:
        for landmark_list in detection_result.pose_landmarks:
            
            # Lists to store pixel coordinates for calculating the bounding box
            x_coordinates = []
            y_coordinates = []
            
            for landmark in landmark_list:
                # Convert normalized coordinates back to actual screen pixels
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                
                x_coordinates.append(cx)
                y_coordinates.append(cy)
                
                # Draw a tiny blue circle at each joint node
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            # --- Calculate and Draw Bounding Box ---
            if x_coordinates and y_coordinates:
                # Find the outer edges of the detected body skeleton
                xmin, xmax = min(x_coordinates), max(x_coordinates)
                ymin, ymax = min(y_coordinates), max(y_coordinates)
                
                # Add a bit of padding so the box isn't tightly choking the joints
                padding = 20
                xmin = max(0, xmin - padding)
                ymin = max(0, ymin - padding)
                xmax = min(w, xmax + padding)
                ymax = min(h, ymax + padding)
                
                # Draw the cyan bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                
                # Pull a simulated tracking confidence score (e.g., 0.92 to 0.98) 
                # to create the UI overlay effect
                confidence_score = 0.96 
                label = f"PERSON: {confidence_score:.2f}"
                
                # Draw the background label box and text
                cv2.rectangle(frame, (xmin, ymin - 25), (xmin + 140, ymin), (255, 255, 0), -1)
                cv2.putText(frame, label, (xmin + 5, ymin - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Show the window
    cv2.imshow('Live Tracking Overlay', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()