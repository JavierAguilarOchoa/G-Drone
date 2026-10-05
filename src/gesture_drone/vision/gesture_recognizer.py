"""MediaPipe's trained gesture recognizer wrapper."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from gesture_drone.vision.gesture_classifier import Gesture, GestureResult


@dataclass(frozen=True)
class Recognition:
    gesture: GestureResult
    landmarks: object


class GestureRecognizer:
    """Recognize MediaPipe's trained canned hand gestures."""

    MIN_CONFIDENCE = 0.80

    _GESTURE_MAP = {
        "Open_Palm": Gesture.OPEN_PALM,
        "Closed_Fist": Gesture.CLOSED_FIST,
        "Pointing_Up": Gesture.POINT_UP,
    }

    _CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )

    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Gesture model not found: {model_path}")

        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def recognize(self, frame, timestamp_ms: int) -> Recognition:
        """Recognize the top gesture from a BGR OpenCV frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._recognizer.recognize_for_video(image, timestamp_ms)

        gesture = Gesture.UNKNOWN
        score = 0.0
        if result.gestures and result.gestures[0]:
            category = result.gestures[0][0]
            gesture = self._GESTURE_MAP.get(category.category_name, Gesture.UNKNOWN)
            score = float(category.score or 0.0)

        safe = gesture is not Gesture.UNKNOWN and score >= self.MIN_CONFIDENCE
        if not safe:
            gesture = Gesture.UNKNOWN

        return Recognition(
            gesture=GestureResult(gesture, score, safe),
            landmarks=result,
        )

    def draw_landmarks(self, frame, result) -> None:
        """Draw the recognizer's detected hand landmarks."""
        height, width = frame.shape[:2]
        for hand in result.hand_landmarks:
            points = [
                (int(landmark.x * width), int(landmark.y * height))
                for landmark in hand
            ]
            for start, end in self._CONNECTIONS:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
            for point in points:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)

    def close(self) -> None:
        self._recognizer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
