import cv2
import time

from gesture_detector import GestureDetector
from core import MemeLibrary, TriggerEngine, AudioPlayer, VisualRenderer


class MemeCamApp:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

        self.detector = GestureDetector()

        self.library = MemeLibrary()
        self.library.load_defaults()

        self.trigger_engine = TriggerEngine(
            effects=self.library.effects,
            min_confidence=0.15,
        )

        self.audio_player = AudioPlayer()
        self.visual_renderer = VisualRenderer()

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        print("Reactify running. Press Q to quit.")
        print(f"Loaded effects: {list(self.library.effects.keys())}")

        last_debug_print = 0.0
        debug_interval = 0.5

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Could not read frame")
                    break

                frame = cv2.flip(frame, 1)

                detection = self.detector.detect(frame)
                effect = self.trigger_engine.get_triggered_effect(detection)

                now = time.time()

                if now - last_debug_print >= debug_interval:
                    print(
                        f"[DEBUG] gesture={detection.gesture}, "
                        f"confidence={detection.confidence}, "
                        f"has_landmarks={detection.landmarks is not None}, "
                        f"effect={effect.name if effect else None}"
                    )
                    last_debug_print = now

                if effect:
                    print(
                        f"Triggered: {effect.name} "
                        f"(confidence: {detection.confidence})"
                    )
                    self.audio_player.play(effect)
                    self.visual_renderer.trigger_overlay(effect)

                frame = self.visual_renderer.render(frame)

                cv2.imshow("Reactify Preview", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

        finally:
            self.detector.close()
            cap.release()
            cv2.destroyAllWindows()
            self.audio_player.close()


def run_cli_app(camera_index: int = 0):
    app = MemeCamApp(camera_index=camera_index)
    app.run()