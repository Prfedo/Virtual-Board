# Hand Detection & Tracking Module (Member 1)

This document covers **only** the Hand Detection & Tracking module. Drawing,
gesture logic, UI, and AI features are owned by the rest of the team and are
not part of this file.

## 1. Recommended Project Structure

```
virtual-board/
├── hand_tracker.py          # This module (Member 1)
├── test_hand_tracking.py    # Standalone demo/test for this module
├── requirements.txt
├── canvas.py                 # Member 2 - Drawing Engine
├── gesture_detector.py       # Member 3 - Gesture Recognition
├── toolbar.py                 # Member 4 - UI
├── main.py                    # Integrates all modules
└── README.md
```

`hand_tracker.py` has no dependency on any other project file, so it can be
developed, tested, and version-controlled independently.

## 2. Installation

```bash
pip install -r requirements.txt
```

or individually:

```bash
pip install opencv-python
pip install mediapipe
```

## 3. Running the Test Program

```bash
python test_hand_tracking.py
```

Press **Q** to quit. The webcam window shows the hand skeleton, the FPS
counter, and the tracked index fingertip coordinates.

## 4. API Usage

```python
from hand_tracker import HandTracker

tracker = HandTracker(max_num_hands=1)

result = tracker.process(frame)   # frame = a BGR image from OpenCV

if result["hand_detected"]:
    index_x, index_y = result["index_tip"]

tracker.close()  # release resources when done
```

`HandTracker` can also be used as a context manager:

```python
with HandTracker() as tracker:
    result = tracker.process(frame)
```

## 5. Returned Data Shape

`process(frame)` always returns a dictionary with the same keys, whether or
not a hand was detected — so callers never need to check if a key exists,
only whether it's `None`.

```python
{
    "hand_detected": True,
    "landmarks": {0: (x, y), 1: (x, y), ..., 20: (x, y)},          # pixel coords
    "normalized_landmarks": {0: (nx, ny, nz), ..., 20: (nx, ny, nz)},  # 0.0-1.0 + depth
    "thumb_tip":  (x, y),
    "index_tip":  (x, y),
    "middle_tip": (x, y),
    "ring_tip":   (x, y),
    "pinky_tip":  (x, y),
    "handedness": "Right",   # or "Left", or None if no hand
}
```

When no hand is detected, `hand_detected` is `False`, `landmarks` and
`normalized_landmarks` are empty dicts `{}`, and every fingertip key is
`None`.

## 6. Landmark IDs (MediaPipe's 21-point hand model)

| ID | Name          | ID | Name          | ID | Name          |
|----|---------------|----|---------------|----|---------------|
| 0  | Wrist         | 7  | Index DIP     | 14 | Ring PIP      |
| 1  | Thumb CMC     | 8  | **Index TIP** | 15 | Ring DIP      |
| 2  | Thumb MCP     | 9  | Middle MCP    | 16 | **Ring TIP**  |
| 3  | Thumb IP      | 10 | Middle PIP    | 17 | Pinky MCP     |
| 4  | **Thumb TIP** | 11 | Middle DIP    | 18 | Pinky PIP     |
| 5  | Index MCP     | 12 | **Middle TIP**| 19 | Pinky DIP     |
| 6  | Index PIP     | 13 | Ring MCP      | 20 | **Pinky TIP** |

Each finger has 4 points running from its base to its tip. The tip
landmarks (4, 8, 12, 16, 20) are the ones exposed as named shortcuts
(`thumb_tip`, `index_tip`, etc.) since they're what most gesture and
drawing logic needs.

## 7. Integration: Member 2 (Drawing Engine)

Member 2 only needs the index fingertip pixel coordinates to know where to
draw on the canvas:

```python
from hand_tracker import HandTracker

tracker = HandTracker()

while True:
    # ... get `frame` from the shared webcam loop in main.py ...
    result = tracker.process(frame)

    if result["hand_detected"]:
        x, y = result["index_tip"]
        canvas.draw_point(x, y)   # Member 2's own drawing function
    else:
        canvas.stop_stroke()      # e.g. lift the "pen" when hand is lost
```

Because coordinates are already converted to pixels (matching the frame's
width/height), Member 2 can draw directly onto a canvas of the same
resolution without doing any conversion.

## 8. Integration: Member 3 (Gesture Recognition)

Member 3 can use either the full `landmarks` dict (pixel coordinates, good
for measuring on-screen distances) or `normalized_landmarks` (0.0–1.0 +
depth, good for resolution-independent comparisons like "is the thumb close
to the index finger"):

```python
from hand_tracker import HandTracker

tracker = HandTracker()

result = tracker.process(frame)

if result["hand_detected"]:
    all_points = result["landmarks"]           # {0: (x,y), ..., 20: (x,y)}
    norm_points = result["normalized_landmarks"] # {0: (x,y,z), ...}

    thumb_tip = all_points[4]
    index_tip = all_points[8]

    # Example: simple pinch detection using normalized coordinates
    nx1, ny1, _ = norm_points[4]
    nx2, ny2, _ = norm_points[8]
    distance = ((nx1 - nx2) ** 2 + (ny1 - ny2) ** 2) ** 0.5

    if distance < 0.05:
        print("Pinch gesture detected")
```

`result["handedness"]` (`"Left"` / `"Right"` / `None`) is also available if
gesture logic ever needs to treat hands differently.

## 9. Performance Considerations

- `static_image_mode=False` (the default) lets MediaPipe track landmarks
  between frames instead of re-detecting from scratch every time — this is
  what makes real-time video tracking fast.
- `max_num_hands=1` (default) is faster than tracking 2 hands; only raise it
  if the project later needs two-hand support.
- Frame resizing (e.g. capturing at 1280x720 instead of 4K) reduces
  MediaPipe's per-frame processing time significantly.
- Always call `tracker.close()` (or use the `with HandTracker() as tracker:`
  context manager) to release native resources when done.

## 10. Testing Checklist

- [ ] Webcam opens without errors.
- [ ] Program exits cleanly and prints a message when no webcam is found.
- [ ] Hand skeleton draws correctly when a hand is in frame.
- [ ] "No hand detected" message shows when hand leaves the frame.
- [ ] Index fingertip coordinates update smoothly as the hand moves.
- [ ] FPS counter displays and updates every frame.
- [ ] Pressing `q` closes the window and releases the webcam (check your OS
      camera indicator light turns off).
- [ ] No crash when the hand exits/re-enters the frame repeatedly.

## 11. Common Errors and Fixes

| Error | Likely Cause | Fix |
|---|---|---|
| `ERROR: Could not open webcam` | Camera in use by another app, or wrong index | Close other apps using the camera; try `cv2.VideoCapture(1)` |
| `ModuleNotFoundError: No module named 'cv2'` | opencv-python not installed | `pip install opencv-python` |
| `ModuleNotFoundError: No module named 'mediapipe'` | mediapipe not installed | `pip install mediapipe` |
| Low FPS / laggy tracking | High resolution or CPU-bound machine | Lower `CAP_PROP_FRAME_WIDTH/HEIGHT`, close other apps |
| Hand skeleton flickers on/off | Confidence thresholds too strict/loose, poor lighting | Adjust `min_detection_confidence` / `min_tracking_confidence`, improve lighting |
| `AttributeError` on `mp.solutions.hands` | Very old or incompatible mediapipe version | `pip install --upgrade mediapipe` |

## 12. Future Improvement Ideas

- Extend `max_num_hands` to 2 and return a list of per-hand results instead
  of a single dict, for two-handed gestures.
- Add landmark smoothing (e.g. a simple moving average) to reduce jitter,
  which would make Member 2's drawing lines steadier.
- Add a confidence/quality score to the returned dict so gesture logic can
  ignore low-confidence frames.