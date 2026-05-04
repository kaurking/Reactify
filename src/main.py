import cv2
import time
import pygame
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

from gesture_detector import GestureDetector, DetectionResult


BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "assets" / "images"
SOUND_DIR = BASE_DIR / "assets" / "sounds"


@dataclass
class MemeEffect:
    name: str
    image_path: Optional[Path] = None
    sound_path: Optional[Path] = None
    cooldown: float = 3.0


class MemeLibrary:
    def __init__(self):
        self.effects: Dict[str, MemeEffect] = {}

    def load_defaults(self):
        self.effects = {
            "thumbs_up": MemeEffect(
                name="thumbs_up",
                image_path=IMAGE_DIR / "thumbs_up.png",
                sound_path=SOUND_DIR / "nice.mp3",
                cooldown=3.0
            ),
            "mouth_open": MemeEffect(
                name="mouth_open",
                image_path=IMAGE_DIR / "vine_boom.png",
                sound_path=SOUND_DIR / "vine_boom.mp3",
                cooldown=2.0
            ),
            "hands_up": MemeEffect(
                name="hands_up",
                image_path=IMAGE_DIR / "hype.png",
                sound_path=SOUND_DIR / "airhorn.mp3",
                cooldown=5.0
            )
        }


class TriggerEngine:
    def __init__(self, effects: Dict[str, MemeEffect], min_confidence: float = 0.8):
        self.effects = effects
        self.min_confidence = min_confidence
        self.last_trigger_times: Dict[str, float] = {}

    def get_triggered_effect(self, detection: DetectionResult) -> Optional[MemeEffect]:
        if detection.gesture is None:
            return None

        if detection.confidence < self.min_confidence:
            return None

        effect = self.effects.get(detection.gesture)

        if effect is None:
            return None

        now = time.time()
        last_time = self.last_trigger_times.get(effect.name, 0)

        if now - last_time < effect.cooldown:
            return None

        self.last_trigger_times[effect.name] = now
        return effect


class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()

    def play(self, effect: MemeEffect):
        if effect.sound_path is None:
            return

        if not effect.sound_path.exists():
            print(f"Sound missing: {effect.sound_path}")
            return

        try:
            sound = pygame.mixer.Sound(str(effect.sound_path))
            sound.play()
        except Exception as e:
            print(f"Could not play sound: {e}")


class VisualRenderer:
    def __init__(self):
        self.active_overlay = None
        self.overlay_until = 0.0

    def trigger_overlay(self, effect: MemeEffect, duration: float = 1.5):
        if effect.image_path is None:
            return

        if not effect.image_path.exists():
            print(f"Image missing: {effect.image_path}")
            return

        overlay = cv2.imread(str(effect.image_path), cv2.IMREAD_UNCHANGED)

        if overlay is None:
            print(f"Could not load image: {effect.image_path}")
            return

        self.active_overlay = overlay
        self.overlay_until = time.time() + duration

    def render(self, frame):
        if self.active_overlay is None:
            return frame

        if time.time() > self.overlay_until:
            self.active_overlay = None
            return frame

        return self._draw_overlay(frame, self.active_overlay)

    def _draw_overlay(self, frame, overlay):
        frame_h, frame_w = frame.shape[:2]

        target_w = int(frame_w * 0.35)
        scale = target_w / overlay.shape[1]
        target_h = int(overlay.shape[0] * scale)

        overlay_resized = cv2.resize(overlay, (target_w, target_h))

        x = frame_w - target_w - 30
        y = 30

        if overlay_resized.shape[2] == 4:
            alpha = overlay_resized[:, :, 3] / 255.0
            overlay_rgb = overlay_resized[:, :, :3]

            for c in range(3):
                frame[y:y + target_h, x:x + target_w, c] = (
                    alpha * overlay_rgb[:, :, c]
                    + (1 - alpha) * frame[y:y + target_h, x:x + target_w, c]
                )
        else:
            frame[y:y + target_h, x:x + target_w] = overlay_resized

        return frame


class MemeCamApp:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

        self.detector = GestureDetector()

        self.library = MemeLibrary()
        self.library.load_defaults()

        self.trigger_engine = TriggerEngine(self.library.effects)
        self.audio_player = AudioPlayer()
        self.visual_renderer = VisualRenderer()

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        print("Reactify running. Press Q to quit.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read frame")
                break

            detection = self.detector.detect(frame)

            effect = self.trigger_engine.get_triggered_effect(detection)

            if effect:
                print(f"Triggered: {effect.name}")
                self.audio_player.play(effect)
                self.visual_renderer.trigger_overlay(effect)

            frame = self.visual_renderer.render(frame)

            cv2.imshow("Reactify Preview", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = MemeCamApp(camera_index=0)
    app.run()