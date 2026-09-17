# =========================================================
# Member 4 - UI / Toolbar module
# This file draws the color palette + CLEAR button on top of
# the webcam feed, and figures out which button is being
# "touched" at a given (x, y) point.
#
# NOTE: Right now there is no hand-tracking yet, so this file
# uses your MOUSE CLICK as a stand-in for the fingertip position.
# Once Member 1's hand tracker is ready, replace the mouse
# logic with the (x, y) fingertip coordinates it provides.
# =========================================================

import cv2

# ---------------------------------------------------------
# 1. Define every button ONE time, as a list, instead of
#    typing out coordinates over and over. Each button is:
#    (name, color_to_draw, x1, y1, x2, y2)
# ---------------------------------------------------------
BUTTONS = [
    ("RED",    (0, 0, 255),      10, 10, 100, 70),
    ("BLUE",   (255, 0, 0),     110, 10, 200, 70),
    ("GREEN",  (0, 255, 0),     210, 10, 300, 70),
    ("ERASER", (0, 0, 0),       310, 10, 400, 70),
    ("CLEAR",  (50, 50, 50),    410, 10, 500, 70),
    ("SAVE",   (150, 150, 150), 510, 10, 600, 70),
]

# This keeps track of which color is currently selected.
# It starts as RED by default.
current_color = (0, 0, 255)

# Keeps track of the brush thickness. The trackbar (slider)
# will update this value while the program runs.
brush_size = 5


def draw_toolbar(frame):
    """
    Draws every button onto the given frame.
    Called once per frame, AFTER all other drawing is done
    and BEFORE cv2.imshow.
    """
    for name, color, x1, y1, x2, y2 in BUTTONS:
        # Fill the button with its color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        # Put a white text label on top of the button
        cv2.putText(frame, name, (x1 + 8, y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # If this button's color is the one currently selected,
        # draw a white border around it so the user can see
        # what's active right now. CLEAR and SAVE aren't colors,
        # so they never get this highlight.
        if name not in ("CLEAR", "SAVE") and color == current_color:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

    # Show the current brush size as text, next to the buttons.
    cv2.putText(frame, f"Brush: {brush_size}", (610, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def check_toolbar_click(x, y):
    """
    Given a point (x, y) - a fingertip later, a mouse click now -
    check whether that point lands inside any button's box.
    Returns the button name ("RED", "BLUE", ..., "CLEAR") or None.
    """
    for name, color, x1, y1, x2, y2 in BUTTONS:
        if x1 < x < x2 and y1 < y < y2:
            return name
    return None


def on_mouse_click(event, x, y, flags, param):
    """
    Temporary stand-in for finger detection.
    Whenever you left-click the video window, this runs
    and checks if you clicked a button.
    """
    global current_color

    if event == cv2.EVENT_LBUTTONDOWN:
        result = check_toolbar_click(x, y)

        if result == "RED":
            current_color = (0, 0, 255)
        elif result == "BLUE":
            current_color = (255, 0, 0)
        elif result == "GREEN":
            current_color = (0, 255, 0)
        elif result == "ERASER":
            current_color = (0, 0, 0)
        elif result == "CLEAR":
            print("Action: CLEAR (tell Member 2 to wipe the canvas)")
        elif result == "SAVE":
            print("Action: SAVE (tell Member 5 to save the drawing)")


# ---------------------------------------------------------
# Main loop - only runs if you execute THIS file directly.
# When Member 5 integrates everything into main.py, they will
# import draw_toolbar() and check_toolbar_click() instead of
# running this loop.
# ---------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # Safety check: stop clearly if the webcam didn't open,
    # instead of crashing on a confusing error later.
    if not cap.isOpened():
        print("Error: could not open webcam.")
        exit()

    cv2.namedWindow("Virtual Board")
    cv2.setMouseCallback("Virtual Board", on_mouse_click)

    # Creates a slider under the window, from 1 to 50,
    # starting at 5. The lambda does nothing on change -
    # we read the slider's value ourselves every frame instead.
    cv2.createTrackbar("Brush Size", "Virtual Board", 5, 50, lambda x: None)

    while True:
        ret, frame = cap.read()

        # If a frame wasn't captured properly, skip this loop
        # instead of crashing.
        if not ret:
            print("Warning: failed to grab frame.")
            break

        # Flip horizontally so it acts like a mirror
        # (feels natural when the user moves their hand).
        frame = cv2.flip(frame, 1)

        # Read the slider's current position every frame and
        # store it in our brush_size variable so draw_toolbar()
        # can display it, and so main.py can later hand this
        # value to Member 2's drawing engine.
        brush_size = cv2.getTrackbarPos("Brush Size", "Virtual Board")
        # Avoid a brush size of 0, which would draw nothing.
        if brush_size < 1:
            brush_size = 1

        # Draw all buttons on top of the frame
        frame = draw_toolbar(frame)

        # Show the currently selected color as a small preview
        cv2.putText(frame, "Selected:", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (140, 90), (180, 120), current_color, -1)

        # ALWAYS show the frame last, after every drawing step
        cv2.imshow("Virtual Board", frame)

        if cv2.waitKey(1) == 27:  # ESC key exits
            break

    cap.release()
    cv2.destroyAllWindows()