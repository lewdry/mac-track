import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# model_complexity=0 keeps your MacBook Air running cool
pose = mp_pose.Pose(
    model_complexity=0, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
print("Tracking started. Click the video window and press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Mirror image and convert to RGB
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Live Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()