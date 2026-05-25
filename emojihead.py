import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
import numpy as np

HEAD_LANDMARK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
EMOJI = "🧢"
EMOJI_FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"

def make_emoji_image(size):
    """Render emoji into a square RGBA PIL image of the given pixel size."""
    # Apple Color Emoji only has bitmap strikes at these exact sizes
    VALID_SIZES = [20, 32, 40, 48, 64, 96, 160]
    
    # Pick the largest valid size <= our target, or smallest if target is tiny
    font_size = max((s for s in VALID_SIZES if s <= size), default=VALID_SIZES[0])
    
    try:
        font = ImageFont.truetype(EMOJI_FONT_PATH, font_size)
    except Exception as e:
        print(f"Font load failed: {e}")
        return None

    canvas_size = font_size * 3
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((canvas_size // 4, canvas_size // 4), EMOJI, font=font, embedded_color=True)

    bbox = img.getbbox()
    if not bbox:
        print("Emoji rendered as fully transparent — font may not support embedded_color")
        return None

    img = img.crop(bbox)
    img = img.resize((size, size), Image.LANCZOS)
    return img


def blend_emoji(frame, emoji_img, xmin, ymin, xmax, ymax):
    """Alpha-blend a PIL RGBA emoji image onto the cv2 frame ROI."""
    head_w = xmax - xmin
    head_h = ymax - ymin

    if head_w <= 0 or head_h <= 0:
        return

    emoji_resized = emoji_img.resize((head_w, head_h), Image.LANCZOS)
    emoji_np = np.array(emoji_resized)  # H x W x 4

    alpha = emoji_np[:, :, 3:4] / 255.0
    emoji_bgr = emoji_np[:, :, :3][:, :, ::-1]  # RGB → BGR

    roi = frame[ymin:ymax, xmin:xmax]
    if roi.shape[0] != head_h or roi.shape[1] != head_w:
        return  # clipped at frame edge

    frame[ymin:ymax, xmin:xmax] = (
        emoji_bgr * alpha + roi * (1 - alpha)
    ).astype("uint8")


def get_head_bbox(landmark_list, w, h, padding=40):
    head_x, head_y = [], []
    for idx in HEAD_LANDMARK_INDICES:
        lm = landmark_list[idx]
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        if 0 <= cx < w and 0 <= cy < h:
            head_x.append(cx)
            head_y.append(cy)

    if not head_x:
        return None

    xmin = max(0, min(head_x) - padding)
    xmax = min(w, max(head_x) + padding)
    ymin = max(0, min(head_y) - padding)
    ymax = min(h, max(head_y) + padding)

    # Force square using the larger dimension
    box_w = xmax - xmin
    box_h = ymax - ymin
    side = max(box_w, box_h)

    # Recenter on the original box's center
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2

    # Shift upward a bit — landmarks cluster on lower face
    cy = int(cy - side * 0.1)

    xmin = max(0, cx - side // 2)
    xmax = min(w, cx + side // 2)
    ymin = max(0, cy - side // 2)
    ymax = min(h, cy + side // 2)

    return xmin, ymin, xmax, ymax


# ── Setup ────────────────────────────────────────────────────────────────────

# Pre-render emoji at a large base size; we'll resize per-frame as needed
BASE_EMOJI_IMG = make_emoji_image(256)
if BASE_EMOJI_IMG is None:
    print("ERROR: Could not render emoji. Check font path and Pillow version.")
    exit(1)
else:
    print(f"Emoji pre-rendered OK: {BASE_EMOJI_IMG.size}")

model_path = 'pose_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=4
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("Emoji head active. Press 'q' to quit.")


# --- Recording/photo state ---
import os
import time
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

recording = False
video_writer = None
video_filename = None

print("Press SPACE to capture photo, 'r' to start/stop recording, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.pose_landmarks:
        for landmark_list in detection_result.pose_landmarks:
            bbox = get_head_bbox(landmark_list, w, h)
            if bbox:
                xmin, ymin, xmax, ymax = bbox
                blend_emoji(frame, BASE_EMOJI_IMG, xmin, ymin, xmax, ymax)

    # Handle video recording
    if recording:
        if video_writer is None:
            # Define video codec and create VideoWriter
            timestr = str(int(time.time()))
            video_filename = os.path.join(output_dir, f"emojihead_recording_{timestr}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (w, h))
            print(f"Recording started: {video_filename}")
        video_writer.write(frame)

    cv2.imshow('Emoji Head', frame)
    key = cv2.waitKey(5) & 0xFF

    if key == ord('q'):
        break
    elif key == 32:  # Spacebar
        # Save photo
        timestr = str(int(time.time()))
        photo_filename = os.path.join(output_dir, f"emojiface_capture_{timestr}.png")
        cv2.imwrite(photo_filename, frame)
        print(f"Photo saved: {photo_filename}")
    elif key == ord('r'):
        if not recording:
            recording = True
            video_writer = None  # Will be created on next frame
        else:
            recording = False
            if video_writer is not None:
                video_writer.release()
                print(f"Recording stopped: {video_filename}")
            video_writer = None

if video_writer is not None:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()