# Virtual Board

Webcam whiteboard: track one hand, draw with your index finger, and pick tools from an on-screen toolbar.

## Project layout

```
virtual-board/
├── hand_tracker.py          # Member 1 — MediaPipe hand landmarks
├── Test_hand_tracking.py    # Standalone tracker demo
├── canvas.py                 # Member 2 — drawing layer
├── gesture_detector.py       # Member 3 — draw / select / idle
├── toolbar.py                 # Member 4 — colors, eraser, clear, save
├── main.py                    # Wires every module together
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run the full app

```bash
python main.py
```

### Gestures

| Pose | Action |
|------|--------|
| Index finger up (others down) | Draw, or erase if **ERASER** is selected |
| Index + middle up | Hover the toolbar and tap a button |
| Anything else / no hand | Pen up |

Stay out of the top toolbar strip while drawing so you do not paint over the buttons.

### Keyboard

| Key | Action |
|-----|--------|
| `q` or `Esc` | Quit |
| `c` | Clear the board |
| `s` | Save `saved_drawings/board_YYYYMMDD_HHMMSS.png` |
| `[` / `]` | Smaller / larger brush |
| `,` / `.` | Smaller / larger eraser |

## Module APIs

### Hand tracking (`hand_tracker.py`)

```python
from hand_tracker import HandTracker

tracker = HandTracker(max_num_hands=1)
result = tracker.process(frame)

if result["hand_detected"]:
    index_x, index_y = result["index_tip"]

tracker.close()
```

`process(frame)` always returns the same keys. Missing data is `None` or `{}`.

### Gestures (`gesture_detector.py`)

```python
from gesture_detector import GestureDetector

gesture = GestureDetector().detect(result)  # "draw" | "select" | "idle"
```

### Canvas (`canvas.py`)

```python
from canvas import Canvas

board = Canvas()
board.ensure_size(frame)
board.draw_stroke((x, y), color=(0, 0, 255), thickness=8)
board.stop_stroke()
frame = board.overlay(frame)
board.clear()
path = board.save()
```

### Toolbar (`toolbar.py`)

```python
from toolbar import Toolbar

ui = Toolbar()
frame = ui.draw(frame)
name = ui.hit_test(x, y)          # "RED" / "CLEAR" / ... or None
action = ui.apply(name)           # "CLEAR" / "SAVE" / None
color = ui.current_color          # BGR tuple; eraser is (0, 0, 0)
thickness = ui.thickness
```

## Tracker-only demo

```bash
python Test_hand_tracking.py
```

Press **Q** to quit. Shows the skeleton, index fingertip, and FPS.

## Tips

- Use even lighting and keep the whole hand in view.
- If the camera fails to open, close other apps using it or try index `1` in `main.py`.
- Lower capture resolution in `main.py` if FPS is low.
