"""
calibrate_drone.py
===================
Interactive calibration tool. Run this BEFORE main.py
to find the correct PIXELS_PER_YARD value for your drone altitude.

Usage:
    python calibrate_drone.py

1. The first frame of your video opens.
2. Click on TWO yard-line markings you can identify.
3. Enter the real-world yard distance between them.
4. Copy the output PIXELS_PER_YARD into main.py.
"""

import cv2
import numpy as np
import sys

VIDEO_PATH = "input_videos/game_footage.mp4"
clicked_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv2.circle(param, (x, y), 6, (0, 255, 0), -1)
        if len(clicked_points) > 1:
            cv2.line(param, clicked_points[-2], clicked_points[-1],
                     (0, 255, 255), 2)
        cv2.imshow("Calibration — click 2 yard lines, then press ENTER", param)
        print(f"  Point {len(clicked_points)}: pixel ({x}, {y})")


def run():
    print("=" * 55)
    print("  DRONE CALIBRATION TOOL")
    print("=" * 55)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        sys.exit(1)

    display = frame.copy()
    win     = "Calibration — click 2 yard lines, then press ENTER"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_callback, display)

    print("\nClick TWO yard-line markings in the frame, then press ENTER.")
    print("(Example: click the 20-yard line, then the 30-yard line.)\n")

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(clicked_points) >= 2:   # Enter
            break
        if key == 27:                                  # Esc
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    p1, p2      = clicked_points[0], clicked_points[1]
    pixel_dist  = float(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
    real_yards  = float(input("How many yards apart are those two lines? → "))

    ppy = pixel_dist / real_yards

    print(f"\n{'='*55}")
    print(f"  PIXELS_PER_YARD = {ppy:.2f}")
    print(f"{'='*55}")
    print(f"\n  Open main.py and set:  PIXELS_PER_YARD = {ppy:.2f}\n")

    with open("calibration.txt", "w") as f:
        f.write(f"PIXELS_PER_YARD={ppy:.4f}\n")
    print("  Also saved to calibration.txt")


if __name__ == "__main__":
    run()
