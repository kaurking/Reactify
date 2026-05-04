from dataclasses import dataclass
from typing import Optional, Dict
import time


@dataclass
class DetectionResult:
    gesture: Optional[str]
    confidence: float
    landmarks: Optional[Dict] = None


class GestureDetector:
    """
    Placeholder detector.

    Later this file becomes:
    - MediaPipe setup
    - landmark extraction
    - gesture rules / templates / classifiers
    """

    def __init__(self):
        self.start_time = time.time()

    def detect(self, frame) -> DetectionResult:
        """
        Fake detection for testing pipeline.

        Current behavior:
        - every ~5 seconds triggers "thumbs_up"
        - otherwise returns no gesture
        """

        elapsed = int(time.time() - self.start_time)

        if elapsed > 0 and elapsed % 5 == 0:
            return DetectionResult(
                gesture="thumbs_up",
                confidence=0.95,
                landmarks={"placeholder": True}
            )

        return DetectionResult(
            gesture=None,
            confidence=0.0
        )