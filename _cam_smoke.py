import time
import cv2
from canvas import Canvas
from gesture_detector import GestureDetector
from hand_tracker import HandTracker
from toolbar import Toolbar
from main import PointSmoother

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("CAMERA: not available (could not open index 0)")
    raise SystemExit(0)

ok, frame = cap.read()
if not ok or frame is None:
    print("CAMERA: opened but failed to read a frame")
    cap.release()
    raise SystemExit(1)

print("CAMERA: got frame", frame.shape)
frame = cv2.flip(frame, 1)
tracker = HandTracker()
gestures = GestureDetector()
toolbar = Toolbar()
canvas = Canvas()
smoother = PointSmoother()
canvas.ensure_size(frame)

hands = 0
for i in range(15):
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.flip(frame, 1)
    canvas.ensure_size(frame)
    tracking = tracker.process(frame)
    gesture = gestures.detect(tracking)
    point = smoother.update(tracking.get("index_tip") if tracking.get("hand_detected") else None)
    if tracking.get("hand_detected"):
        hands += 1
    if gesture == "draw" and point is not None and not toolbar.contains(point[0], point[1]):
        canvas.draw_stroke(point, toolbar.current_color, toolbar.thickness)
    else:
        canvas.stop_stroke()
    composed = canvas.overlay(frame)
    composed = toolbar.draw(composed)
    print(f"frame {i}: detected={tracking['hand_detected']} gesture={gesture} overlay={composed.shape}")
    time.sleep(0.03)

print("hands seen in 15 frames:", hands)
tracker.close()
cap.release()
print("CAMERA SMOKE TEST OK")
