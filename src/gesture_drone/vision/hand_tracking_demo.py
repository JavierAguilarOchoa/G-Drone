"""Live webcam demo for MediaPipe's trained gesture recognizer."""

import cv2
import time
from pathlib import Path

from gesture_drone.vision.gesture_recognizer import GestureRecognizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_recognizer.task"


def run_demo() -> None:
    """Show the webcam with hand landmarks until ``q`` is pressed."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the MacBook camera.")

    try:
        with GestureRecognizer(MODEL_PATH) as recognizer:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the camera.")

                frame = cv2.flip(frame, 1)
                timestamp_ms = int(time.monotonic() * 1000)
                recognition = recognizer.recognize(frame, timestamp_ms)
                recognizer.draw_landmarks(frame, recognition.landmarks)

                gesture = recognition.gesture
                gesture_text = (
                    f"Gesture: {gesture.gesture.value} | "
                    f"Confidence: {gesture.confidence:.0%}"
                )
                gesture_color = (0, 255, 0) if gesture.safe_to_command else (0, 255, 255)

                cv2.putText(
                    frame,
                    "MediaPipe Gesture Recognizer | Press q to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    gesture_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    gesture_color,
                    2,
                )
                cv2.imshow("Gesture Drone Hand Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
