"""MediaPipe hand-landmark detection."""

import cv2
import mediapipe as mp


class HandTracker:
    """Detect and draw hand landmarks on OpenCV video frames."""

    _CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17), (17, 18),
    )

    def __init__(
        self,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        """Return MediaPipe's hand result for a BGR OpenCV frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._hands.process(rgb_frame)

    def draw_landmarks(self, frame, result) -> None:
        """Draw detected landmarks and connections onto a BGR frame."""
        height, width = frame.shape[:2]

        for hand_landmarks in result.multi_hand_landmarks or []:
            points = [
                (int(landmark.x * width), int(landmark.y * height))
                for landmark in hand_landmarks.landmark
            ]

            for start_index, end_index in self._CONNECTIONS:
                cv2.line(
                    frame,
                    points[start_index],
                    points[end_index],
                    (0, 255, 0),
                    2,
                )

            for point in points:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
