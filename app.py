import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe landmark indices
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
INDEX_FINGER_PIP = 6
MIDDLE_FINGER_PIP = 10

PINCH_THRESHOLD_PX = 40
DRAW_COLOR = (0, 255, 255)  # Cyan lines on canvas
LINE_THICKNESS = 5
WINDOW_NAME = "Hand Drawing CV"
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# MediaPipe Hands equivalent settings via Tasks API
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
hand_landmarker_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_landmarker_options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    hand_landmarker.close()
    raise RuntimeError("Could not open webcam (VideoCapture(0)).")

ret, frame = cap.read()
if not ret:
    cap.release()
    hand_landmarker.close()
    raise RuntimeError("Could not read an initial frame from the webcam.")

frame_height, frame_width = frame.shape[:2]
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

prev_x: int | None = None
prev_y: int | None = None
prev_time = time.time()
fps = 0.0
mode = "IDLE"
frame_timestamp_ms = 0


def landmark_to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    """Convert normalized MediaPipe landmark to pixel coordinates."""
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y


def finger_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_index_up(landmarks) -> bool:
    """Index finger extended: tip above PIP joint."""
    return landmarks[INDEX_FINGER_TIP].y < landmarks[INDEX_FINGER_PIP].y


def is_middle_up(landmarks) -> bool:
    """Middle finger extended: tip above PIP joint."""
    return landmarks[MIDDLE_FINGER_TIP].y < landmarks[MIDDLE_FINGER_PIP].y


def overlay_canvas_on_frame(frame_bgr: np.ndarray, canvas_bgr: np.ndarray) -> np.ndarray:
    """Blend drawing canvas over the webcam frame so strokes appear boldly."""
    return cv2.addWeighted(frame_bgr, 1.0, canvas_bgr, 1.0, 0)


def draw_hand_debug(frame_bgr: np.ndarray, hand_landmarks) -> None:
    """Draw index and middle fingertips for visual feedback."""
    index_tip = landmark_to_pixel(hand_landmarks[INDEX_FINGER_TIP], frame_width, frame_height)
    middle_tip = landmark_to_pixel(hand_landmarks[MIDDLE_FINGER_TIP], frame_width, frame_height)
    cv2.circle(frame_bgr, index_tip, 8, (0, 255, 0), -1)
    cv2.circle(frame_bgr, middle_tip, 8, (255, 0, 255), -1)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms += 33
    results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    mode = "IDLE"

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]

        index_tip = landmark_to_pixel(hand_landmarks[INDEX_FINGER_TIP], frame_width, frame_height)
        middle_tip = landmark_to_pixel(hand_landmarks[MIDDLE_FINGER_TIP], frame_width, frame_height)

        curr_x, curr_y = index_tip
        pinch_distance = finger_distance(index_tip, middle_tip)

        draw_hand_debug(frame, hand_landmarks)

        if pinch_distance < PINCH_THRESHOLD_PX:
            # SELECTION / HOVER: reposition anchors, do not draw
            mode = "SELECTION"
            prev_x, prev_y = curr_x, curr_y
        elif is_index_up(hand_landmarks) and not is_middle_up(hand_landmarks):
            # DRAW: only index finger raised
            mode = "DRAW"
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), DRAW_COLOR, LINE_THICKNESS)
            prev_x, prev_y = curr_x, curr_y
        else:
            mode = "IDLE"

    display = overlay_canvas_on_frame(frame, canvas)

    current_time = time.time()
    elapsed = current_time - prev_time
    if elapsed > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
    prev_time = current_time

    cv2.putText(
        display,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"Mode: {mode}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        canvas[:] = 0
        prev_x, prev_y = None, None

cap.release()
hand_landmarker.close()
cv2.destroyAllWindows()
