"""
toolbar.py

UI / Toolbar module for the Virtual-Board project.
Owner: Member 4

Draws the color palette and action buttons on each frame, and reports
which button a point (fingertip) lands in. Does not draw on the canvas
itself — main.py applies the result.
"""

import cv2


class Toolbar:
    """Color palette + CLEAR / SAVE sitting along the top of the frame."""

    HEIGHT = 80

    # (name, BGR color or None for actions, x1, y1, x2, y2)
    BUTTONS = [
        ("RED",    (0, 0, 255),      10, 10, 100, 70),
        ("BLUE",   (255, 0, 0),     110, 10, 200, 70),
        ("GREEN",  (0, 255, 0),     210, 10, 300, 70),
        ("YELLOW", (0, 255, 255),   310, 10, 400, 70),
        ("ERASER", (40, 40, 40),    410, 10, 520, 70),
        ("CLEAR",  (50, 50, 50),    530, 10, 630, 70),
        ("SAVE",   (90, 90, 90),    640, 10, 740, 70),
        ("SNAP",   (140, 80, 20),   750, 10, 870, 70),
    ]

    COLOR_MAP = {
        "RED": (0, 0, 255),
        "BLUE": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "YELLOW": (0, 255, 255),
        "ERASER": (0, 0, 0),
    }

    def __init__(self):
        self.current_color = self.COLOR_MAP["RED"]
        self.brush_size = 8
        self.eraser_size = 40
        self.selected_name = "RED"
        self.shape_snap = False
        self._last_hit = None

    @property
    def thickness(self) -> int:
        if self.selected_name == "ERASER":
            return self.eraser_size
        return self.brush_size

    def contains(self, x, y) -> bool:
        """True if the point is inside the toolbar strip (do not draw there)."""
        return y < self.HEIGHT

    def hit_test(self, x, y):
        """Return the button name under (x, y), or None."""
        for name, _color, x1, y1, x2, y2 in self.BUTTONS:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return name
        return None

    def apply(self, name):
        """
        Apply a button selection.

        Color / eraser changes take effect immediately.
        SNAP toggles shape cleanup and is returned so main.py can
        show a status message.
        CLEAR and SAVE are returned as action strings so main.py can
        call the canvas. Repeating the same button every frame is ignored.
        """
        if name is None or name == self._last_hit:
            if name is None:
                self._last_hit = None
            return None

        self._last_hit = name

        if name in self.COLOR_MAP:
            self.selected_name = name
            self.current_color = self.COLOR_MAP[name]
            return None

        if name == "SNAP":
            self.shape_snap = not self.shape_snap
            return "SNAP"

        if name in ("CLEAR", "SAVE"):
            return name

        return None

    def reset_hit(self):
        """Call when the finger leaves the toolbar so the next tap registers."""
        self._last_hit = None

    def draw(self, frame):
        """Paint buttons + status text onto `frame` (in place) and return it."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], self.HEIGHT), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        for name, color, x1, y1, x2, y2 in self.BUTTONS:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.putText(
                frame, name, (x1 + 8, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
            )
            selected = name == self.selected_name or (name == "SNAP" and self.shape_snap)
            if selected:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

        snap = "ON" if self.shape_snap else "OFF"
        status = (
            f"Brush: {self.brush_size}  |  Eraser: {self.eraser_size}  "
            f"|  Mode: {self.selected_name}  |  SNAP: {snap}"
        )
        cv2.putText(
            frame, status, (10, self.HEIGHT + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        return frame


# Kept for teammates / older snippets that imported functions instead of Toolbar.
_default_toolbar = Toolbar()
current_color = _default_toolbar.current_color
brush_size = _default_toolbar.brush_size


def draw_toolbar(frame):
    return _default_toolbar.draw(frame)


def check_toolbar_click(x, y):
    return _default_toolbar.hit_test(x, y)
