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
PATTERN_COLOR = (0, 255, 0)  # Green for pattern dots/lines
PATTERN_DOT_RADIUS = 6
LINE_THICKNESS = 5
WINDOW_NAME = "Hand Drawing CV"
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# Hand skeleton connections for pattern drawing
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross connections
    (5, 9), (9, 13), (13, 17),
]

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
app_mode = "DRAW"  # DRAW or PATTERN
draw_mode_state = "IDLE"
frame_timestamp_ms = 0


def landmark_to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y


def finger_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_index_up(landmarks) -> bool:
    return landmarks[INDEX_FINGER_TIP].y < landmarks[INDEX_FINGER_PIP].y


def is_middle_up(landmarks) -> bool:
    return landmarks[MIDDLE_FINGER_TIP].y < landmarks[MIDDLE_FINGER_PIP].y


def overlay_canvas_on_frame(frame_bgr: np.ndarray, canvas_bgr: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(frame_bgr, 1.0, canvas_bgr, 1.0, 0)


def draw_hand_debug(frame_bgr: np.ndarray, hand_landmarks) -> None:
    index_tip = landmark_to_pixel(hand_landmarks[INDEX_FINGER_TIP], frame_width, frame_height)
    middle_tip = landmark_to_pixel(hand_landmarks[MIDDLE_FINGER_TIP], frame_width, frame_height)
    cv2.circle(frame_bgr, index_tip, 8, (0, 255, 0), -1)
    cv2.circle(frame_bgr, middle_tip, 8, (255, 0, 255), -1)


def draw_hand_pattern(frame_bgr: np.ndarray, canvas_bgr: np.ndarray, hand_landmarks) -> None:
    """Draw dots at all finger tips and palm landmarks, then connect with lines."""
    points = []
    for i, lm in enumerate(hand_landmarks):
        px, py = landmark_to_pixel(lm, frame_width, frame_height)
        points.append((px, py))

    # Draw lines connecting hand skeleton
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = points[start_idx]
        pt2 = points[end_idx]
        cv2.line(canvas_bgr, pt1, pt2, PATTERN_COLOR, 2, cv2.LINE_AA)

    # Draw dots at all landmarks
    for i, (px, py) in enumerate(points):
        # Tips get bigger dots
        if i in (4, 8, 12, 16, 20):
            cv2.circle(canvas_bgr, (px, py), PATTERN_DOT_RADIUS + 2, (0, 0, 255), -1)
        else:
            cv2.circle(canvas_bgr, (px, py), PATTERN_DOT_RADIUS, PATTERN_COLOR, -1)

    # Overlay the pattern on frame
    cv2.addWeighted(frame_bgr, 0.6, canvas_bgr, 0.4, 0, frame_bgr)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms += 33
    results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if app_mode == "DRAW":
        draw_mode_state = "IDLE"

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]

            index_tip = landmark_to_pixel(hand_landmarks[INDEX_FINGER_TIP], frame_width, frame_height)
            middle_tip = landmark_to_pixel(hand_landmarks[MIDDLE_FINGER_TIP], frame_width, frame_height)

            curr_x, curr_y = index_tip
            pinch_distance = finger_distance(index_tip, middle_tip)

            draw_hand_debug(frame, hand_landmarks)

            if pinch_distance < PINCH_THRESHOLD_PX:
                draw_mode_state = "SELECTION"
                prev_x, prev_y = curr_x, curr_y
            elif is_index_up(hand_landmarks) and not is_middle_up(hand_landmarks):
                draw_mode_state = "DRAW"
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), DRAW_COLOR, LINE_THICKNESS)
                prev_x, prev_y = curr_x, curr_y
            else:
                draw_mode_state = "IDLE"

        display = overlay_canvas_on_frame(frame, canvas)

        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
        prev_time = current_time

        cv2.putText(display, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, f"Mode: DRAW ({draw_mode_state})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display)

    elif app_mode == "PATTERN":
        pattern_layer = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        if results.hand_landmarks:
            draw_hand_pattern(frame, pattern_layer, results.hand_landmarks[0])

        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
        prev_time = current_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Mode: PATTERN (show hand)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 0
        prev_x, prev_y = None, None
    elif key == ord("m"):
        if app_mode == "DRAW":
            app_mode = "PATTERN"
            canvas[:] = 0
            prev_x, prev_y = None, None
        else:
            app_mode = "DRAW"
            canvas[:] = 0

cap.release()
hand_landmarker.close()
cv2.destroyAllWindows()
