"""
main.py

Integrates hand tracking, gestures, the drawing canvas, and the toolbar
into one webcam app.

Gestures
    Index finger up          → draw (or erase, if ERASER is selected)
    Index + middle up        → select a toolbar button
    Any other pose / no hand → pen up

Keyboard
    q / ESC  quit
    c        clear the board
    s        save the drawing
    ] / [    bigger / smaller brush
    . / ,    bigger / smaller eraser
"""

import time

import cv2

from canvas import Canvas
from gesture_detector import GestureDetector
from hand_tracker import HandTracker
from toolbar import Toolbar


class PointSmoother:
    """Exponential moving average so strokes jitter less."""

    def __init__(self, alpha: float = 0.45):
        self.alpha = alpha
        self.point = None

    def update(self, xy):
        if xy is None:
            self.point = None
            return None
        if self.point is None:
            self.point = (int(xy[0]), int(xy[1]))
            return self.point
        ax, ay = xy
        px, py = self.point
        self.point = (
            int(self.alpha * ax + (1.0 - self.alpha) * px),
            int(self.alpha * ay + (1.0 - self.alpha) * py),
        )
        return self.point

    def reset(self):
        self.point = None


def _draw_hud(frame, gesture, fps, message):
    h = frame.shape[0]
    cv2.putText(
        frame, f"FPS: {int(fps)}   Gesture: {gesture}",
        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )
    if message:
        cv2.putText(
            frame, message, (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
    help_lines = [
        "Index = draw    Index+Middle = toolbar    q=quit  c=clear  s=save",
    ]
    cv2.putText(
        frame, help_lines[0], (10, h - 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
    )


def _cursor_color(gesture, toolbar: Toolbar):
    if gesture == "select":
        return (255, 255, 255)
    if toolbar.selected_name == "ERASER":
        return (80, 80, 80)
    return toolbar.current_color


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("  Close other apps using the camera, or try camera index 1.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(max_num_hands=1)
    gestures = GestureDetector()
    toolbar = Toolbar()
    canvas = Canvas()
    smoother = PointSmoother()

    window_name = "Virtual Board"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    status_message = ""
    status_until = 0.0

    print("Virtual Board started.")
    print("  Index finger up          → draw")
    print("  Index + middle up        → pick a toolbar button")
    print("  q / ESC to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARNING: Failed to read a frame. Retrying...")
                continue

            frame = cv2.flip(frame, 1)
            canvas.ensure_size(frame)

            tracking = tracker.process(frame)
            gesture = gestures.detect(tracking)

            index_tip = tracking.get("index_tip") if tracking.get("hand_detected") else None
            point = smoother.update(index_tip)

            if gesture == "select" and point is not None:
                canvas.stop_stroke()
                button = toolbar.hit_test(point[0], point[1])
                action = toolbar.apply(button)
                if action == "CLEAR":
                    canvas.clear()
                    status_message = "Board cleared"
                    status_until = time.time() + 2.0
                elif action == "SAVE":
                    path = canvas.save()
                    status_message = f"Saved {path}"
                    status_until = time.time() + 3.0
                    print(status_message)
                if button is None:
                    toolbar.reset_hit()
            elif gesture == "draw" and point is not None and not toolbar.contains(point[0], point[1]):
                canvas.draw_stroke(point, toolbar.current_color, toolbar.thickness)
            else:
                canvas.stop_stroke()
                toolbar.reset_hit()

            frame = canvas.overlay(frame)
            frame = toolbar.draw(frame)

            if point is not None:
                radius = max(6, toolbar.thickness)
                cv2.circle(frame, point, radius, _cursor_color(gesture, toolbar), 2)
                cv2.circle(frame, point, 4, (255, 255, 255), cv2.FILLED)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            msg = status_message if now < status_until else ""
            _draw_hud(frame, gesture, fps, msg)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                canvas.clear()
                status_message = "Board cleared"
                status_until = time.time() + 2.0
            if key == ord("s"):
                path = canvas.save()
                status_message = f"Saved {path}"
                status_until = time.time() + 3.0
                print(status_message)
            if key == ord("]"):
                toolbar.brush_size = min(50, toolbar.brush_size + 2)
            if key == ord("["):
                toolbar.brush_size = max(2, toolbar.brush_size - 2)
            if key == ord("."):
                toolbar.eraser_size = min(80, toolbar.eraser_size + 4)
            if key == ord(","):
                toolbar.eraser_size = max(10, toolbar.eraser_size - 4)
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. Goodbye.")


if __name__ == "__main__":
    main()
