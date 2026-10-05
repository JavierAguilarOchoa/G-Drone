from types import SimpleNamespace

from gesture_drone.vision.gesture_classifier import Gesture, GestureClassifier


def _hand(points):
    return SimpleNamespace(
        landmark=[SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]
    )


def test_invalid_landmark_count_is_unknown():
    result = GestureClassifier().classify(SimpleNamespace(landmark=[]))

    assert result.gesture is Gesture.UNKNOWN
    assert not result.safe_to_command

