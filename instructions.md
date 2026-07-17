# Hand Drawing CV Application

Build a real-time hand-tracking drawing app using MediaPipe Hands and OpenCV.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Press `q` to quit. Press `c` to clear the canvas.

## Requirements

1. Initialize MediaPipe Hands with `min_detection_confidence=0.7` and `min_tracking_confidence=0.5`.
2. Initialize OpenCV video capture: `cap = cv2.VideoCapture(0)`.
3. Maintain a persistent black image named `canvas` (same size as camera frames) to store drawn points using `cv2.line()`.
4. Extract landmark 8 (INDEX_FINGER_TIP) and landmark 12 (MIDDLE_FINGER_TIP). Convert normalized float positions to exact X, Y pixel coordinates relative to window size.
5. Logic:
   - If distance between Index Tip and Middle Tip is less than 40 pixels → **SELECTION/HOVER** mode (do not draw; update previous X, Y anchors to prevent trailing lines).
   - If only Index Tip is up → **DRAW** mode. Use `cv2.line()` from `(prev_x, prev_y)` to `(curr_x, curr_y)`.
6. Combine the webcam frame and canvas using `cv2.addWeighted()` or bitwise operations so the drawing appears boldly over the camera stream.
7. Show frame rate (FPS) on-screen to track optimization performance.
