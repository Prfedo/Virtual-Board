import os
import numpy as np
import cv2
from canvas import Canvas
from shape_snap import recognize
from toolbar import Toolbar, draw_toolbar, check_toolbar_click
from gesture_detector import GestureDetector
from main import PointSmoother

errors = []


def check(name, cond, detail=""):
    if cond:
        print("OK  ", name)
    else:
        errors.append(f"{name} {detail}".strip())
        print("FAIL", name, detail)


board = Canvas(200, 100)
board.draw_stroke((10, 50), (0, 0, 255), 4)
board.draw_stroke((40, 50), (0, 0, 255), 4)
check("stroke paints pixels", int(board.layer.sum()) > 0, str(board.layer.sum()))
frame = np.full((100, 200, 3), 30, dtype=np.uint8)
out = board.overlay(frame)
check("overlay same shape", out.shape == frame.shape)
path = board.save(folder="_test_saved")
check("save writes file", os.path.isfile(path), path)
board.clear()
check("clear wipes", int(board.layer.sum()) == 0)

board.draw_stroke((10, 50), (0, 255, 0), 5)
small = np.zeros((50, 80, 3), dtype=np.uint8)
board.ensure_size(small)
check("ensure_size resizes", board.layer.shape[:2] == (50, 80))
check("ensure_size keeps ink", int(board.layer.sum()) > 0)

big = np.zeros((120, 160, 3), dtype=np.uint8)
out2 = board.overlay(big)
check("overlay auto-resize", out2.shape == big.shape)

ui = Toolbar()
frame = np.zeros((480, 1280, 3), dtype=np.uint8)
drawn = ui.draw(frame.copy())
check("toolbar draw", drawn.shape == frame.shape)
check("hit red", ui.hit_test(50, 40) == "RED")
check("hit miss", ui.hit_test(5, 200) is None)
check("apply red", ui.apply("BLUE") is None and ui.selected_name == "BLUE")
check("apply clear action", ui.apply("CLEAR") == "CLEAR")
check("debounce same button", ui.apply("CLEAR") is None)
ui.reset_hit()
check("save after reset", ui.apply("SAVE") == "SAVE")
ui.reset_hit()
check("snap toggle on", ui.apply("SNAP") == "SNAP" and ui.shape_snap is True)
check("snap debounce", ui.apply("SNAP") is None and ui.shape_snap is True)
ui.reset_hit()
check("snap toggle off", ui.apply("SNAP") == "SNAP" and ui.shape_snap is False)
check("hit snap", ui.hit_test(800, 40) == "SNAP")
check("contains toolbar", ui.contains(10, 10) and not ui.contains(10, 200))
check("legacy hit", check_toolbar_click(50, 40) == "RED")
check("legacy draw", draw_toolbar(frame.copy()).shape == frame.shape)

g = GestureDetector()
idle = {"hand_detected": False, "landmarks": {}}
check("idle no hand", g.detect(idle) == "idle")
check("idle none", g.detect(None) == "idle")


def pose(index_up, middle_up, ring_up=False, pinky_up=False):
    def finger(up, mcp, pip, dip, tip):
        y_tip = 10 if up else 80
        return {mcp: (0, 90), pip: (0, 50), dip: (0, 30), tip: (0, y_tip)}

    lm = {}
    lm.update(finger(index_up, 5, 6, 7, 8))
    lm.update(finger(middle_up, 9, 10, 11, 12))
    lm.update(finger(ring_up, 13, 14, 15, 16))
    lm.update(finger(pinky_up, 17, 18, 19, 20))
    return {"hand_detected": True, "landmarks": lm}


check("draw pose", g.detect(pose(True, False)) == "draw")
check("select pose", g.detect(pose(True, True)) == "select")
check("idle fist-ish", g.detect(pose(False, False)) == "idle")

s = PointSmoother(alpha=0.5)
check("smooth none", s.update(None) is None)
p1 = s.update((10, 10))
p2 = s.update((20, 20))
check("smooth ints", isinstance(p1[0], int) and p2 != p1)

line_pts = [(20 + i, 80 + (1 if i % 4 == 0 else 0)) for i in range(90)]
check("snap line", (recognize(line_pts) or {}).get("kind") == "line")

theta = np.linspace(0, 2 * np.pi, 48)
circle_pts = [(220 + 70 * np.cos(t), 180 + 70 * np.sin(t)) for t in theta]
check("snap circle", (recognize(circle_pts) or {}).get("kind") == "circle")

rect_pts = (
    [(40 + i, 40) for i in range(120)]
    + [(160, 40 + i) for i in range(60)]
    + [(160 - i, 100) for i in range(120)]
    + [(40, 100 - i) for i in range(60)]
)
check("snap rect", (recognize(rect_pts) or {}).get("kind") == "rect")

sq_pts = (
    [(50 + i, 50) for i in range(80)]
    + [(130, 50 + i) for i in range(80)]
    + [(130 - i, 130) for i in range(80)]
    + [(50, 130 - i) for i in range(80)]
)
check("snap square", (recognize(sq_pts) or {}).get("kind") == "square")

print("module failures:", len(errors))
if errors:
    raise SystemExit("\n".join(errors))

print("Creating HandTracker (may download model)...")
from hand_tracker import HandTracker

tracker = HandTracker(max_num_hands=1)
blank = np.zeros((480, 640, 3), dtype=np.uint8)
result = tracker.process(blank)
check("empty frame keys", set(result) >= {
    "hand_detected", "landmarks", "index_tip", "thumb_tip", "handedness"
})
check("empty frame no hand", result["hand_detected"] is False)
check("index_tip none", result["index_tip"] is None)

frame2, result2 = tracker.process_and_draw(blank.copy())
check("process_and_draw shape", frame2.shape == blank.shape)
tracker.close()

print("ALL PASSED")
