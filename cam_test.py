"""
webcam_test.py — Picamera2 live preview test
---------------------------------------------
Opens the Pi camera via libcamera/Picamera2 and shows a live feed
using OpenCV. Use this to confirm your camera and display are working
before running the full matcher.

Press Q in the preview window to quit.

Usage:
    python webcam_test.py
    python webcam_test.py --width 1280 --height 720   # higher resolution

Install deps if needed:
    sudo apt install python3-picamera2
    # or: pip install picamera2
"""

import cv2
import argparse
from picamera2 import Picamera2
from libcamera import controls


def main():
    parser = argparse.ArgumentParser(description="Picamera2 preview test")
    parser.add_argument("--width",  type=int, default=640,  help="Capture width  (default: 640)")
    parser.add_argument("--height", type=int, default=480,  help="Capture height (default: 480)")
    args = parser.parse_args()

    print(f"[INFO] Starting Picamera2 at {args.width}×{args.height} ...")

    picam2 = Picamera2()

    # RGB888 → 3-channel array, OpenCV reads it as BGR without any conversion.
    # Using XRGB8888 would give a 4-channel array and require slicing [:, :, :3].
    picam2.configure(picam2.create_video_configuration(
        main={"format": "RGB888", "size": (args.width, args.height)}
    ))
    picam2.start()
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Continuous,
        "AfSpeed": controls.AfSpeedEnum.Fast,
    })
    print("[INFO] Continuous autofocus enabled")
    print("[INFO] Camera started — press Q in the window to quit")

    # startWindowThread() + namedWindow on main thread — needed on Pi OS
    cv2.startWindowThread()
    cv2.namedWindow("PiCam Test")

    while True:
        frame = picam2.capture_array()

        if frame is None:
            continue

        # Overlay a status label so it's obvious the feed is live
        cv2.putText(
            frame,
            f"Picamera2  {args.width}x{args.height}  |  Q to quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 100),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("PiCam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
