"""
gesture_detector.py

Gesture Recognition for the Virtual-Board project.
Owner: Member 3

Reads the landmark dict from HandTracker and classifies a simple pose:

    draw   — only the index finger is up  (write on the board)
    select — index + middle are up        (hover / tap the toolbar)
    idle   — any other pose, or no hand   (pen lifted)

Finger "up" is: fingertip is above its PIP joint in image coordinates
(y grows downward, so tip.y < pip.y).
"""


class GestureDetector:
    # MediaPipe landmark ids used for the four non-thumb fingers.
    # (tip, pip)
    FINGERS = {
        "index": (8, 6),
        "middle": (12, 10),
        "ring": (16, 14),
        "pinky": (20, 18),
    }

    def fingers_up(self, landmarks: dict) -> dict:
        """
        Return {finger_name: True/False} for index/middle/ring/pinky.
        `landmarks` is the pixel dict from HandTracker: {id: (x, y)}.
        """
        raised = {}
        for name, (tip_id, pip_id) in self.FINGERS.items():
            if tip_id not in landmarks or pip_id not in landmarks:
                raised[name] = False
                continue
            tip_y = landmarks[tip_id][1]
            pip_y = landmarks[pip_id][1]
            raised[name] = tip_y < pip_y
        return raised

    def detect(self, tracking_result: dict) -> str:
        """
        Classify the current hand pose.

        Returns one of: "draw", "select", "idle".
        """
        if not tracking_result or not tracking_result.get("hand_detected"):
            return "idle"

        landmarks = tracking_result.get("landmarks") or {}
        if not landmarks:
            return "idle"

        up = self.fingers_up(landmarks)
        index = up["index"]
        middle = up["middle"]
        ring = up["ring"]
        pinky = up["pinky"]

        # Pointing: index only — used for drawing.
        if index and not middle and not ring and not pinky:
            return "draw"

        # Peace / two fingers — used to pick toolbar buttons without inking.
        if index and middle and not ring and not pinky:
            return "select"

        return "idle"
