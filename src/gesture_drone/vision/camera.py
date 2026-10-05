"""Utilities for displaying a live camera feed."""

import cv2


def run_camera(camera_index: int = 0) -> None:
    """Open a camera feed and display it until the user presses ``q``.

    Args:
        camera_index: Camera device to open. ``0`` is normally the built-in
            MacBook camera.

    Raises:
        RuntimeError: If the camera cannot be opened.
    """
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. "
            "Check macOS camera permissions and that the camera is not in use."
        )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            # Mirror the image so it behaves like a normal video call preview.
            frame = cv2.flip(frame, 1)
            cv2.imshow("Gesture Drone Camera", frame)

            # OpenCV returns the pressed key as an integer. The low byte is
            # compared with the ASCII value for q.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
