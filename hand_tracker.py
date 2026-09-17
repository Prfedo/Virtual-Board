"""
hand_tracker.py

Hand Detection & Tracking module for the Virtual-Board project.
Owner: Member 1 (Hand Detection & Tracking)

This module is responsible ONLY for:
    - Reading a video frame
    - Running MediaPipe Hands on it
    - Extracting the 21 hand landmarks
    - Converting them to pixel + normalized coordinates
    - Exposing a simple, clean dictionary that other modules can consume

It does NOT draw on a whiteboard, does NOT recognize gestures, and does
NOT build any UI. Those are the responsibilities of Members 2, 3 and 4.
Member 2 (Drawing Engine) and Member 3 (Gesture Recognition) are expected
to import `HandTracker` from this file and call `.process(frame)`.

------------------------------------------------------------------------
MEDIAPIPE HAND LANDMARK MAP (21 points per hand)
------------------------------------------------------------------------
MediaPipe numbers every landmark on the hand from 0 to 20. Here is what
each index physically represents:

     0  - Wrist
     1  - Thumb CMC   (base of thumb)
     2  - Thumb MCP
     3  - Thumb IP
     4  - Thumb TIP                <-- fingertip
     5  - Index MCP   (base of index finger)
     6  - Index PIP
     7  - Index DIP
     8  - Index TIP                <-- fingertip (used for drawing)
     9  - Middle MCP
    10  - Middle PIP
    11  - Middle DIP
    12  - Middle TIP               <-- fingertip
    13  - Ring MCP
    14  - Ring PIP
    15  - Ring DIP
    16  - Ring TIP                 <-- fingertip
    17  - Pinky MCP
    18  - Pinky PIP
    19  - Pinky DIP
    20  - Pinky TIP                <-- fingertip

Pattern to remember: each finger (thumb, index, middle, ring, pinky) has
4 landmarks along it (base -> tip), and they are laid out in order after
the wrist (landmark 0). The "TIP" landmarks (4, 8, 12, 16, 20) are the
ones most modules care about, since they represent fingertip positions.
------------------------------------------------------------------------
"""

import time
import cv2
import mediapipe as mp


class HandTracker:
    """
    A reusable wrapper around MediaPipe Hands.

    Usage:
        tracker = HandTracker()
        result = tracker.process(frame)

        if result["hand_detected"]:
            index_x, index_y = result["index_tip"]

    Design notes:
        - This class does NOT open the webcam itself. It only receives
          frames (via `process`) and returns tracking data. This keeps it
          reusable: the test script, the final app, or any teammate's
          code can feed it frames from any source (webcam, video file,
          etc.) without changing this class.
        - Drawing on the frame is OPTIONAL and separate from tracking, so
          Members 2/3 can use the tracking data without ever touching
          OpenCV drawing functions.
    """

    # Convenient name -> landmark index map for fingertips.
    # Exposed as a class attribute so other modules can reference it
    # (e.g. HandTracker.FINGER_TIPS["index_tip"]) without hardcoding 8.
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
    ):
        """
        Initialize the MediaPipe Hands model.

        Args:
            max_num_hands: How many hands to track at once. The project
                spec asks for one hand initially. To support two hands
                later, a teammate only needs to change this to 2 -- the
                internal loop over `multi_hand_landmarks` already
                supports multiple hands (see `_extract_hand_data`).
            min_detection_confidence: Minimum confidence for the initial
                hand detection to be considered successful.
            min_tracking_confidence: Minimum confidence for the
                landmark tracker to keep tracking between frames.
            static_image_mode: False = optimized for video streams
                (tracks between frames, faster). True = treats every
                frame as a brand-new image (slower, more accurate on
                stills). Video streams should keep this False.
        """
        self.max_num_hands = max_num_hands

        # mp.solutions.hands is MediaPipe's high-level "Hands" solution.
        # It bundles palm detection + 21-point landmark tracking.
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):
        """
        Run hand detection + tracking on a single BGR frame (as returned
        by cv2.VideoCapture.read()).

        Returns a dictionary shaped like this when a hand IS detected:

            {
                "hand_detected": True,
                "landmarks": {0: (x, y), 1: (x, y), ..., 20: (x, y)},
                "normalized_landmarks": {0: (nx, ny, nz), ...},
                "thumb_tip": (x, y),
                "index_tip": (x, y),
                "middle_tip": (x, y),
                "ring_tip": (x, y),
                "pinky_tip": (x, y),
                "handedness": "Right",
            }

        And like this when NO hand is detected:

            {
                "hand_detected": False,
                "landmarks": {},
                "normalized_landmarks": {},
                "thumb_tip": None,
                "index_tip": None,
                "middle_tip": None,
                "ring_tip": None,
                "pinky_tip": None,
                "handedness": None,
            }

        This "always return the same keys" design means Member 2 and
        Member 3 never have to check `if "index_tip" in result` -- the
        key always exists, it's just `None` when there's no hand.

        Args:
            frame: A BGR image (numpy array) from OpenCV.

        Returns:
            dict: Tracking result described above.
        """
        empty_result = self._empty_result()

        if frame is None:
            # Defensive check: caller passed a bad/empty frame.
            return empty_result

        # MediaPipe expects RGB images, OpenCV gives us BGR by default.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # For performance, mark the frame as not writeable while
        # MediaPipe processes it (this is MediaPipe's recommended
        # pattern and avoids an unnecessary internal copy).
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if not results.multi_hand_landmarks:
            # No hand detected in this frame -- return the "empty" shape
            # instead of None, so calling code doesn't need extra checks.
            return empty_result

        # Only the first detected hand is used for now (max_num_hands=1
        # by default). If max_num_hands > 1, `results.multi_hand_landmarks`
        # already contains multiple hands; extending this method to
        # return a list of per-hand dicts is straightforward later.
        hand_landmarks = results.multi_hand_landmarks[0]

        handedness_label = None
        if results.multi_handedness:
            handedness_label = results.multi_handedness[0].classification[0].label

        frame_height, frame_width = frame.shape[:2]
        return self._extract_hand_data(
            hand_landmarks, frame_width, frame_height, handedness_label
        )

    def _extract_hand_data(self, hand_landmarks, frame_width, frame_height, handedness_label):
        """
        Convert a single MediaPipe hand_landmarks object into our
        pixel-coordinate dictionary format.

        MediaPipe gives coordinates normalized to [0.0, 1.0] relative to
        the frame size. Multiplying by frame width/height converts them
        into actual pixel coordinates, which is what drawing and gesture
        code typically wants to work with.
        """
        pixel_landmarks = {}
        normalized_landmarks = {}

        for idx, lm in enumerate(hand_landmarks.landmark):
            pixel_x = int(lm.x * frame_width)
            pixel_y = int(lm.y * frame_height)
            pixel_landmarks[idx] = (pixel_x, pixel_y)

            # Keep normalized (x, y, z) too -- useful for Member 3's
            # gesture recognition, since normalized coordinates are
            # resolution-independent (e.g. comparing finger distances
            # regardless of webcam resolution). z is MediaPipe's rough
            # depth estimate (relative to the wrist).
            normalized_landmarks[idx] = (lm.x, lm.y, lm.z)

        result = {
            "hand_detected": True,
            "landmarks": pixel_landmarks,
            "normalized_landmarks": normalized_landmarks,
            "handedness": handedness_label,
        }

        # Add named fingertip shortcuts, e.g. result["index_tip"] = (x, y)
        for name, landmark_id in self.FINGER_TIPS.items():
            result[name] = pixel_landmarks[landmark_id]

        return result

    def _empty_result(self):
        """Build the standard 'no hand detected' dictionary."""
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
        """
        OPTIONAL debug helper: draws the 21 landmarks and hand
        connections on top of `frame`.

        This is intentionally separate from `process()` so that Member 2
        and Member 3 can use the tracking data WITHOUT ever calling this
        method or depending on OpenCV drawing/visualization at all.

        Two ways to use it:
            1. Recommended (simple): pass the dict returned by
               `process()`:
                   result = tracker.process(frame)
                   tracker.draw_landmarks(frame, result=result)

            2. Advanced: if you need MediaPipe's native connection
               drawing style, use `process_and_draw()` instead, which
               keeps the raw MediaPipe landmark object around.

        Args:
            frame: The BGR frame to draw on (modified in place).
            result: The dict returned by `process()`.

        Returns:
            The same frame, with landmarks drawn on it (if a hand was
            detected).
        """
        if result is None or not result.get("hand_detected"):
            return frame

        for idx, (x, y) in result["landmarks"].items():
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Highlight the index fingertip specifically, since that's the
        # point Member 2's Drawing Engine cares about most.
        if result.get("index_tip") is not None:
            cv2.circle(frame, result["index_tip"], 8, (0, 0, 255), -1)

        return frame

    def process_and_draw(self, frame):
        """
        Convenience method for the demo/test program: runs `process()`
        AND draws MediaPipe's native hand skeleton (with proper finger
        connections, not just dots) using MediaPipe's own drawing_utils.

        This is purely for visualization/debugging and is NOT required
        by Member 2 or Member 3 -- they should use `process()` only.

        Returns:
            (frame, result) tuple: the frame with drawings applied, and
            the same tracking dict that `process()` returns.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        raw_results = self._hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if raw_results.multi_hand_landmarks:
            for hand_landmarks in raw_results.multi_hand_landmarks:
                self._mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_drawing_styles.get_default_hand_landmarks_style(),
                    self._mp_drawing_styles.get_default_hand_connections_style(),
                )

        result = self.process(frame)
        return frame, result

    def close(self):
        """
        Release MediaPipe resources. Call this when you're done with the
        tracker (e.g. when the application exits), similar to closing a
        file handle.
        """
        self._hands.close()

    def __enter__(self):
        """Allows using HandTracker as a context manager: `with HandTracker() as tracker:`"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()