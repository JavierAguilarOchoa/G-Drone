"""Interpretable hand-landmark gesture classification."""

from dataclasses import dataclass
from enum import Enum
from math import hypot
from typing import Sequence


class Gesture(str, Enum):
    UNKNOWN = "UNKNOWN"
    OPEN_PALM = "OPEN_PALM"
    CLOSED_FIST = "CLOSED_FIST"
    POINT_UP = "POINT_UP"


@dataclass(frozen=True)
class GestureResult:
    gesture: Gesture
    confidence: float
    safe_to_command: bool


class GestureClassifier:
    """Classify a single MediaPipe hand using geometric landmark rules.

    The confidence value is a heuristic score, not a calibrated probability.
    It is deliberately gated at 0.80 so uncertain detections become UNKNOWN.
    """

    MIN_CONFIDENCE = 0.80

    def classify(self, hand_landmarks) -> GestureResult:
        """Return a gesture result for one MediaPipe hand landmark object."""
        points = hand_landmarks.landmark
        if len(points) != 21:
            return GestureResult(Gesture.UNKNOWN, 0.0, False)

        palm_size = max(self._distance(points[0], points[9]), 1e-6)
        finger_scores = {
            "index": self._finger_extension_score(points, 8, 6, 5, palm_size),
            "middle": self._finger_extension_score(points, 12, 10, 9, palm_size),
            "ring": self._finger_extension_score(points, 16, 14, 13, palm_size),
            "pinky": self._finger_extension_score(points, 20, 18, 17, palm_size),
        }

        extended = {
            name: score >= 0.80 for name, score in finger_scores.items()
        }
        extended_count = sum(extended.values())

        thumb_open_score = self._thumb_open_score(points, palm_size)
        index_score = finger_scores["index"]
        curled_finger_scores = (
            finger_scores["middle"],
            finger_scores["ring"],
            finger_scores["pinky"],
        )

        if extended_count == 4 and thumb_open_score >= self.MIN_CONFIDENCE:
            confidence = min(
                thumb_open_score,
                *(score for score in finger_scores.values()),
            )
            return self._result(Gesture.OPEN_PALM, confidence)

        if extended_count == 0:
            confidence = min(
                1.0 - thumb_open_score,
                *(1.0 - score for score in finger_scores.values()),
            )
            return self._result(Gesture.CLOSED_FIST, confidence)

        if (
            index_score >= self.MIN_CONFIDENCE
            and all(score < self.MIN_CONFIDENCE for score in curled_finger_scores)
            and thumb_open_score < self.MIN_CONFIDENCE
        ):
            confidence = min(
                index_score,
                *(1.0 - score for score in curled_finger_scores),
                1.0 - thumb_open_score,
            )
            return self._result(Gesture.POINT_UP, confidence)

        return GestureResult(Gesture.UNKNOWN, 0.0, False)

    @staticmethod
    def _result(gesture: Gesture, confidence: float) -> GestureResult:
        confidence = max(0.0, min(1.0, confidence))
        return GestureResult(
            gesture=gesture if confidence >= GestureClassifier.MIN_CONFIDENCE else Gesture.UNKNOWN,
            confidence=confidence,
            safe_to_command=confidence >= GestureClassifier.MIN_CONFIDENCE,
        )

    @staticmethod
    def _distance(first, second) -> float:
        return hypot(first.x - second.x, first.y - second.y)

    @classmethod
    def _finger_extension_score(
        cls,
        points: Sequence,
        tip_index: int,
        pip_index: int,
        mcp_index: int,
        palm_size: float,
    ) -> float:
        """Estimate extension without depending on image up/down direction."""
        tip_distance = cls._distance(points[tip_index], points[0])
        pip_distance = cls._distance(points[pip_index], points[0])
        distance_ratio = tip_distance / max(pip_distance, 1e-6)
        distance_score = (distance_ratio - 1.0) / 0.45

        # An extended finger usually has its tip farther from the wrist than
        # its PIP joint. This ratio is invariant to rotation in the image.
        score = distance_score
        return max(0.0, min(1.0, score))

    @classmethod
    def _thumb_open_score(cls, points: Sequence, palm_size: float) -> float:
        """Estimate whether the thumb is extended away from the palm."""
        # The thumb is extended when its tip is clearly outside the palm.
        # This uses wrist-to-tip distance rather than x/y direction, so it
        # works for either hand and for rotated hands.
        tip_distance = cls._distance(points[4], points[0]) / palm_size
        score = (tip_distance - 0.75) / 0.55
        return max(0.0, min(1.0, score))
