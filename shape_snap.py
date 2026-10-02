"""
Recognize a freehand stroke and replace it with a clean line, rectangle,
square, or circle. Used when the SNAP toolbar mode is on.
"""

import cv2
import numpy as np

MIN_POINTS = 8
MIN_SPAN = 28


def _path_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _line_rms(pts: np.ndarray):
    start, end = pts[0], pts[-1]
    chord = float(np.linalg.norm(end - start))
    if chord < 1.0:
        return 1e9, chord
    direction = (end - start) / chord
    vecs = pts - start
    proj = vecs @ direction
    perp = vecs - np.outer(proj, direction)
    rms = float(np.sqrt(np.mean(np.sum(perp ** 2, axis=1))))
    return rms, chord


def _fit_circle(pts: np.ndarray):
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(pts))))
    b = x * x + y * y
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(max(c + cx * cx + cy * cy, 1.0)))
    dists = np.hypot(x - cx, y - cy)
    rms = float(np.sqrt(np.mean((dists - r) ** 2)))
    return rms, float(cx), float(cy), r


def _point_segment_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray):
    ab = b - a
    length2 = float(np.dot(ab, ab)) + 1e-6
    t = np.clip(((pts - a) @ ab) / length2, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    return np.linalg.norm(pts - proj, axis=1)


def _rect_rms(pts: np.ndarray, rect) -> float:
    box = cv2.boxPoints(rect)
    edge_d = [_point_segment_dist(pts, box[i], box[(i + 1) % 4]) for i in range(4)]
    nearest = np.min(np.stack(edge_d, axis=1), axis=1)
    return float(np.sqrt(np.mean(nearest ** 2)))


def _axis_align(p1, p2, threshold_deg=14):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return p1, p2
    angle = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))
    if angle < threshold_deg:
        return (x1, y1), (x2, y1)
    if angle > 90.0 - threshold_deg:
        return (x1, y1), (x1, y2)
    return p1, p2


def _as_int_point(xy):
    return (int(round(xy[0])), int(round(xy[1])))


def recognize(points):
    """
    Return a shape dict, or None to keep the freehand stroke.

    Shapes:
        {"kind": "line", "p1": (x, y), "p2": (x, y)}
        {"kind": "circle", "center": (x, y), "radius": int}
        {"kind": "rect", "center": (x, y), "size": (w, h), "angle": float}
        {"kind": "square", "center": (x, y), "size": float, "angle": float}
    """
    if points is None or len(points) < MIN_POINTS:
        return None

    pts = np.array(points, dtype=np.float32)
    minxy = pts.min(axis=0)
    maxxy = pts.max(axis=0)
    span = maxxy - minxy
    if float(max(span[0], span[1])) < MIN_SPAN:
        return None

    diag = float(np.linalg.norm(span))
    path = _path_length(pts)
    line_rms, chord = _line_rms(pts)
    closed = chord < 0.30 * max(diag, 1.0)
    straightness = chord / (path + 1e-6)

    if (
        not closed
        and straightness > 0.80
        and line_rms < max(10.0, 0.08 * chord)
    ):
        p1 = _as_int_point(pts[0])
        p2 = _as_int_point(pts[-1])
        p1, p2 = _axis_align(p1, p2)
        return {"kind": "line", "p1": p1, "p2": p2}

    circ_rms, cx, cy, radius = _fit_circle(pts)
    rel_circ = circ_rms / max(radius, 1.0)

    rect = cv2.minAreaRect(pts)
    (_rcx, _rcy), (rw, rh), angle = rect
    rw, rh = abs(float(rw)), abs(float(rh))
    box_err = _rect_rms(pts, rect) if rw >= 1 and rh >= 1 else 1e9

    if (
        not closed
        and line_rms <= box_err
        and line_rms <= circ_rms
        and straightness > 0.68
        and line_rms < max(12.0, 0.12 * chord)
    ):
        p1 = _as_int_point(pts[0])
        p2 = _as_int_point(pts[-1])
        p1, p2 = _axis_align(p1, p2)
        return {"kind": "line", "p1": p1, "p2": p2}

    looks_round = rel_circ < 0.18 and radius >= 16
    looks_box = box_err < max(12.0, 0.12 * diag) and min(rw, rh) >= 18
    if not closed and not looks_round and not looks_box:
        return None

    aspect = min(rw, rh) / max(rw, rh) if max(rw, rh) else 0.0
    circle_wins = looks_round and circ_rms < box_err * 0.75

    if circle_wins:
        return {
            "kind": "circle",
            "center": (int(round(cx)), int(round(cy))),
            "radius": int(max(8, round(radius))),
        }

    if looks_box or closed:
        center = (float(rect[0][0]), float(rect[0][1]))
        if aspect > 0.82:
            side = (rw + rh) / 2.0
            return {
                "kind": "square",
                "center": center,
                "size": side,
                "angle": float(angle),
            }
        return {
            "kind": "rect",
            "center": center,
            "size": (rw, rh),
            "angle": float(angle),
        }

    if looks_round:
        return {
            "kind": "circle",
            "center": (int(round(cx)), int(round(cy))),
            "radius": int(max(8, round(radius))),
        }

    return None


def _ink_thickness(thickness: int) -> int:
    return max(2, int(thickness) * 2)


def draw_shape(image, shape, color, thickness):
    """Draw a recognized shape onto `image` (canvas layer or preview frame)."""
    t = _ink_thickness(thickness)
    kind = shape["kind"]
    if kind == "line":
        cv2.line(image, shape["p1"], shape["p2"], color, t)
        return
    if kind == "circle":
        cv2.circle(image, shape["center"], shape["radius"], color, t)
        return
    if kind == "square":
        side = shape["size"]
        box = cv2.boxPoints((shape["center"], (side, side), shape["angle"]))
    else:
        box = cv2.boxPoints((shape["center"], shape["size"], shape["angle"]))
    pts = np.int32(np.round(box))
    cv2.polylines(image, [pts], True, color, t)


def draw_freehand(image, points, color, thickness):
    t = _ink_thickness(thickness)
    if not points:
        return
    if len(points) == 1:
        x, y = int(points[0][0]), int(points[0][1])
        cv2.circle(image, (x, y), max(1, int(thickness)), color, cv2.FILLED)
        return
    for i in range(1, len(points)):
        a = (int(points[i - 1][0]), int(points[i - 1][1]))
        b = (int(points[i][0]), int(points[i][1]))
        cv2.line(image, a, b, color, t)


def draw_preview(frame, points, color, thickness):
    """Live stroke plus a ghost of the snapped shape, if one is detected."""
    if not points:
        return frame
    draw_freehand(frame, points, color, max(1, thickness // 2 or 1))
    shape = recognize(points)
    if shape is not None:
        draw_shape(frame, shape, (255, 255, 255), thickness)
        draw_shape(frame, shape, color, max(1, thickness - 1))
    return frame
