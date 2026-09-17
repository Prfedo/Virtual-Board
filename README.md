# Virtual-Board
our summer intern project 
# Member 4 — User Interface (Toolbar)

This branch (`member4-ui`) contains the UI layer for the AI Virtual Painter project: the color palette, action buttons, and brush-size control that sit on top of the webcam feed.

## What this module does

- Draws a row of buttons directly on each video frame: **RED, BLUE, GREEN, ERASER, CLEAR, SAVE**
- Draws a slider (OpenCV trackbar) to control brush size (1–50)
- Highlights whichever color is currently selected
- Detects when a point (currently a mouse click, later a fingertip) lands on a button
- Exposes clean functions so other modules can use this UI without needing to know how it's drawn

## Files

| File | Purpose |
|---|---|
| `toolbar.py` | All UI logic: button layout, drawing, click detection, and a standalone test loop |

## How to run it standalone

```bash
pip install opencv-python
python toolbar.py
```

- Click a color button to select it (highlighted with a white border)
- Click CLEAR or SAVE to trigger their action (currently just prints a message)
- Drag the "Brush Size" slider under the window to change brush thickness
- Press `ESC` to quit

> Note: This file currently uses **mouse clicks** as a stand-in for fingertip input, since Member 1's hand-tracking module isn't integrated yet.

## Public functions (for integration)

Other members / `main.py` should use these instead of copying this file's logic:

```python
from toolbar import draw_toolbar, check_toolbar_click

frame = draw_toolbar(frame)              # draw buttons + brush size on a frame
result = check_toolbar_click(x, y)       # returns "RED" / "BLUE" / "GREEN" /
                                          # "ERASER" / "CLEAR" / "SAVE" / None
```

- `x, y` should be the fingertip coordinates from Member 1's hand detector (once available).
- `check_toolbar_click` does **not** change any state — it only reports which button was touched. The caller (in this file's `on_mouse_click`, or eventually `main.py`) decides what to do with that result.

## Shared state / variables

| Variable | Type | Meaning |
|---|---|---|
| `current_color` | `(B, G, R)` tuple | Color Member 2's drawing engine should draw with |
| `brush_size` | `int` (1–50) | Thickness Member 2's drawing engine should use |

Member 2 should read `current_color` and `brush_size` from this module each frame.

## Integration plan

- [ ] Replace mouse-click stand-in with real `(x, y)` fingertip position from Member 1
- [ ] Wire `CLEAR` action to Member 2's canvas-clear function
- [ ] Wire `SAVE` action to Member 5's save/export function
- [ ] Confirm color format `(B, G, R)` matches what Member 2 and Member 3 expect
- [ ] Merge into `main` once tested against the real hand-tracking + drawing pipeline

## Known limitations

- No hover-preview before clicking (only shows selection after the fact)
- Brush size has no visual size preview (just a number)
- UI is fixed in position; not responsive to different camera resolutions
