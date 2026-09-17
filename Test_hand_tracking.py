"""
test_hand_tracking.py

Standalone demo/test program for the HandTracker module.

What it does:
    - Opens the default webcam.
    - Feeds each frame into HandTracker.
    - Draws the hand skeleton (landmarks + connections) for visual
      debugging.
    - Highlights the index fingertip and prints its coordinates on
      screen.
    - Shows the current FPS (frames per second).
    - Exits cleanly when the user presses 'q'.

Run with:
    python test_hand_tracking.py
"""

import time
import cv2

from hand_tracker import HandTracker


def main():
    # --- Open the webcam -------------------------------------------------
    # 0 = default system webcam. Change to 1, 2, etc. if you have
    # multiple cameras and the wrong one opens.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        # Fail fast with a clear message instead of crashing deep inside
        # the read loop below.
        print("ERROR: Could not open webcam. Check that:")
        print("  - Your webcam is connected and not used by another app")
        print("  - You have the correct camera index (try 0, 1, or 2)")
        return

    # Optional: request a specific resolution. Not all webcams honor
    # this exactly, but most will get close.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --- Create the tracker ----------------------------------------------
    tracker = HandTracker(max_num_hands=1)

    prev_time = time.time()

    print("Hand tracking started. Press 'q' to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success or frame is None:
                # A dropped frame shouldn't crash the whole app -- just
                # skip it and try again on the next loop iteration.
                print("WARNING: Failed to read frame from webcam. Retrying...")
                continue

            # Mirror the frame horizontally so it feels like a mirror
            # (natural for hand-tracking demos). Purely cosmetic.
            frame = cv2.flip(frame, 1)

            # Run detection + built-in MediaPipe-style drawing in one call.
            frame, result = tracker.process_and_draw(frame)

            # --- Use the tracking result -------------------------------
            if result["hand_detected"]:
                index_x, index_y = result["index_tip"]

                # Extra highlight circle on the index fingertip so it's
                # obvious which point is being tracked for drawing.
                cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), cv2.FILLED)

                # Display the fingertip coordinates on-screen.
                coord_text = f"Index tip: ({index_x}, {index_y})"
                cv2.putText(
                    frame, coord_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
            else:
                cv2.putText(
                    frame, "No hand detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            # --- FPS calculation ----------------------------------------
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
            prev_time = current_time

            cv2.putText(
                frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            cv2.imshow("Virtual Board - Hand Tracking Test", frame)

            # Exit cleanly when 'q' is pressed. waitKey(1) keeps the
            # video feed real-time; without it the window would freeze.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit key pressed. Shutting down...")
                break

    finally:
        # --- Always release resources, even if an error occurred -------
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and windows closed.")


if __name__ == "__main__":
    main()