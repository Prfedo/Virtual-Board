"""
hand_tracker.py

Hand Detection & Tracking module for the Virtual-Board project.
Owner: Member 1 (Hand Detection & Tracking)

Wraps MediaPipe Tasks HandLandmarker (MediaPipe 1.x). Older tutorials used
`mp.solutions.hands`, which was removed in MediaPipe 1.0.

Public API is unchanged: process(frame) still returns the same dictionary
shape so canvas / gestures / main.py do not need to change.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent / "models"
_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe 21-point skeleton (start landmark, end landmark).
_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _ensure_model(model_path: Path = _MODEL_PATH) -> str:
    """Download the Hand Landmarker bundle once if it is not on disk."""
    if model_path.is_file() and model_path.stat().st_size > 0:
        return str(model_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {model_path} ...")
    urllib.request.urlretrieve(_MODEL_URL, model_path)
    return str(model_path)


class HandTracker:
    """
    A reusable wrapper around MediaPipe HandLandmarker.

    Usage:
        tracker = HandTracker()
        result = tracker.process(frame)

        if result["hand_detected"]:
            index_x, index_y = result["index_tip"]
    """

    FINGER_TIPS = {
        "thumb_tip": 4,
        "index_tip": 8,
        "middle_tip": 12,
        "ring_tip": 16,
        "pinky_tip": 20,
    }

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        static_image_mode: bool = False,
        model_path: str | None = None,
    ):
        self.max_num_hands = max_num_hands
        self._timestamp_ms = 0

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        resolved_model = model_path or _ensure_model()
        running_mode = (
            VisionRunningMode.IMAGE
            if static_image_mode
            else VisionRunningMode.VIDEO
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved_model),
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._static_image_mode = static_image_mode
        self._landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame):
        """Run hand detection + tracking on a single BGR OpenCV frame."""
        empty_result = self._empty_result()

        if frame is None:
            return empty_result

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._static_image_mode:
            detection = self._landmarker.detect(mp_image)
        else:
            now_ms = int(time.time() * 1000)
            if now_ms <= self._timestamp_ms:
                now_ms = self._timestamp_ms + 1
            self._timestamp_ms = now_ms
            detection = self._landmarker.detect_for_video(mp_image, now_ms)

        if not detection.hand_landmarks:
            return empty_result

        landmarks = detection.hand_landmarks[0]
        handedness_label = None
        if detection.handedness:
            first = detection.handedness[0][0]
            handedness_label = getattr(first, "category_name", None) or getattr(
                first, "display_name", None
            )

        frame_height, frame_width = frame.shape[:2]
        return self._extract_hand_data(
            landmarks, frame_width, frame_height, handedness_label
        )

    def _extract_hand_data(self, landmarks, frame_width, frame_height, handedness_label):
        pixel_landmarks = {}
        normalized_landmarks = {}

        for idx, lm in enumerate(landmarks):
            pixel_x = int(lm.x * frame_width)
            pixel_y = int(lm.y * frame_height)
            pixel_landmarks[idx] = (pixel_x, pixel_y)
            z = getattr(lm, "z", 0.0)
            normalized_landmarks[idx] = (lm.x, lm.y, z)

        result = {
            "hand_detected": True,
            "landmarks": pixel_landmarks,
            "normalized_landmarks": normalized_landmarks,
            "handedness": handedness_label,
        }
        for name, landmark_id in self.FINGER_TIPS.items():
            result[name] = pixel_landmarks.get(landmark_id)
        return result

    def _empty_result(self):
        result = {
            "hand_detected": False,
            "landmarks": {},
            "normalized_landmarks": {},
            "handedness": None,
        }
        for name in self.FINGER_TIPS:
            result[name] = None
        return result

    def draw_landmarks(self, frame, result=None, hand_landmarks_raw=None):
        """Optional debug helper: draw dots + bones onto `frame` in place."""
        if result is None or not result.get("hand_detected"):
            return frame

        pts = result["landmarks"]
        for start, end in _HAND_CONNECTIONS:
            if start in pts and end in pts:
                cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
        for idx, (x, y) in pts.items():
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        if result.get("index_tip") is not None:
            cv2.circle(frame, result["index_tip"], 8, (0, 0, 255), -1)
        return frame

    def process_and_draw(self, frame):
        """Demo helper: process() plus skeleton overlay."""
        result = self.process(frame)
        self.draw_landmarks(frame, result=result)
        return frame, result

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
