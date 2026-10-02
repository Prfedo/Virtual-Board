"""
canvas.py

Drawing Engine for the Virtual-Board project.
Owner: Member 2

Keeps a separate drawing layer (same size as the webcam frame) so strokes
stay on the board even as the camera image changes. Other modules only
need to call draw_stroke / stop_stroke / clear / save.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class Canvas:
    """Persistent drawing layer overlaid on the live camera feed."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self._prev_point = None

    def ensure_size(self, frame):
        """Keep the drawing layer the same size as the webcam frame."""
        h, w = frame.shape[:2]
        if w == self.width and h == self.height:
            return
        if self.layer is not None and self.layer.size:
            self.layer = cv2.resize(self.layer, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            self.layer = np.zeros((h, w, 3), dtype=np.uint8)
        self.width, self.height = w, h
        self._prev_point = None

    def draw_stroke(self, point, color, thickness):
        """
        Continue a stroke to `point`. Consecutive points are connected
        with a line so fast motion still looks like a continuous path.
        """
        if point is None:
            self.stop_stroke()
            return

        x, y = int(point[0]), int(point[1])
        thickness = max(1, int(thickness))

        if self._prev_point is None:
            cv2.circle(self.layer, (x, y), thickness, color, cv2.FILLED)
        else:
            cv2.line(self.layer, self._prev_point, (x, y), color, thickness * 2)

        self._prev_point = (x, y)

    def stop_stroke(self):
        """Lift the pen so the next point starts a new stroke."""
        self._prev_point = None

    def clear(self):
        """Wipe the entire drawing."""
        self.layer[:] = 0
        self._prev_point = None

    def overlay(self, frame):
        """
        Composite the drawing onto a camera frame.

        Ink (non-black pixels) replaces the camera; empty canvas shows
        the live video. Eraser strokes are black, so they punch a hole
        back to the camera feed.
        """
        if frame.shape[:2] != self.layer.shape[:2]:
            self.ensure_size(frame)

        gray = cv2.cvtColor(self.layer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        background = cv2.bitwise_and(frame, frame, mask=mask_inv)
        foreground = cv2.bitwise_and(self.layer, self.layer, mask=mask)
        return cv2.add(background, foreground)

    def save(self, folder: str = "saved_drawings") -> str:
        """Save the drawing (transparent-looking black background) as a PNG."""
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"board_{stamp}.png"
        cv2.imwrite(str(path), self.layer)
        return str(path)
