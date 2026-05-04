from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import time

import cv2
import numpy as np
import pygame

from core.profile_store import list_profiles
from gesture_detector import DetectionResult


BASE_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = BASE_DIR / "assets" / "images"
SOUND_DIR = BASE_DIR / "assets" / "sounds"


@dataclass
class MemeEffect:
    name: str
    image_path: Optional[Path] = None
    sound_path: Optional[Path] = None
    cooldown: float = 3.0
    overlay_duration: float = 1.5


class MemeLibrary:
    def __init__(self):
        self.effects: Dict[str, MemeEffect] = {}

    def load_defaults(self):
        self.reload()

    def reload(self):
        self.effects = {}

        profiles = list_profiles()

        for profile in profiles:
            effect = self._effect_from_profile(profile)

            if effect is not None:
                self.effects[effect.name] = effect

        if not self.effects:
            self.effects["absolute_cinema"] = MemeEffect(
                name="absolute_cinema",
                image_path=IMAGE_DIR / "Absolute_Cinema.png",
                sound_path=SOUND_DIR / "vine-boom.mp3",
                cooldown=3.0,
                overlay_duration=1.8,
            )

    def _effect_from_profile(self, profile: dict) -> Optional[MemeEffect]:
        gesture = profile.get("gesture")

        if not gesture:
            return None

        return MemeEffect(
            name=gesture,
            image_path=self._resolve_project_path(
                profile.get("image"),
                self._default_image_path(gesture),
            ),
            sound_path=self._resolve_project_path(
                profile.get("sound"),
                self._default_sound_path(gesture),
            ),
            cooldown=self._profile_float(profile, "cooldown", 3.0),
            overlay_duration=self._profile_float(profile, "overlay_duration", 1.5),
        )

    @staticmethod
    def _resolve_project_path(
        path_value: Optional[str],
        default: Optional[Path] = None,
    ) -> Optional[Path]:
        if not path_value:
            return default

        path = Path(path_value)

        if path.is_absolute():
            return path

        return BASE_DIR / path

    @staticmethod
    def _profile_float(profile: dict, key: str, default: float) -> float:
        try:
            return float(profile.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _default_image_path(gesture: str) -> Optional[Path]:
        if gesture == "absolute_cinema":
            return IMAGE_DIR / "Absolute_Cinema.png"

        return None

    @staticmethod
    def _default_sound_path(gesture: str) -> Optional[Path]:
        if gesture == "absolute_cinema":
            return SOUND_DIR / "vine-boom.mp3"

        return None


class TriggerEngine:
    def __init__(
        self,
        effects: Dict[str, MemeEffect],
        min_confidence: float = 0.15,
        required_hold_seconds: float = 0.05,
    ):
        self.effects = effects
        self.min_confidence = min_confidence
        self.required_hold_seconds = required_hold_seconds

        self.last_trigger_times: Dict[str, float] = {}

        self.current_candidate: Optional[str] = None
        self.candidate_start_time: float = 0.0
        self.candidate_ready: bool = False

    def get_triggered_effect(self, detection: DetectionResult) -> Optional[MemeEffect]:
        now = time.time()

        if detection.gesture is None or detection.confidence < self.min_confidence:
            self._reset_candidate()
            return None

        effect = self.effects.get(detection.gesture)

        if effect is None:
            self._reset_candidate()
            return None

        if detection.gesture != self.current_candidate:
            self.current_candidate = detection.gesture
            self.candidate_start_time = now
            self.candidate_ready = False
            return None

        held_seconds = now - self.candidate_start_time

        if held_seconds < self.required_hold_seconds:
            return None

        if self.candidate_ready:
            return None

        last_time = self.last_trigger_times.get(effect.name, 0.0)

        if now - last_time < effect.cooldown:
            return None

        self.last_trigger_times[effect.name] = now
        self.candidate_ready = True

        return effect

    def _reset_candidate(self):
        self.current_candidate = None
        self.candidate_start_time = 0.0
        self.candidate_ready = False

    def get_hold_progress(self) -> float:
        if self.current_candidate is None:
            return 0.0

        elapsed = time.time() - self.candidate_start_time

        if self.required_hold_seconds <= 0:
            return 1.0

        return max(0.0, min(1.0, elapsed / self.required_hold_seconds))

    def get_current_candidate(self) -> Optional[str]:
        return self.current_candidate


class AudioPlayer:
    def __init__(self):
        self.initialized = False

        try:
            pygame.mixer.init()
            self.initialized = True
        except Exception as error:
            print(f"Could not initialize audio: {error}")

    def play(self, effect: MemeEffect):
        if not self.initialized or effect.sound_path is None:
            return

        if not effect.sound_path.exists():
            print(f"Sound missing: {effect.sound_path}")
            return

        try:
            sound = pygame.mixer.Sound(str(effect.sound_path))
            sound.play()
        except Exception as error:
            print(f"Could not play sound: {error}")

    def close(self):
        if self.initialized:
            pygame.mixer.quit()
            self.initialized = False


class VisualRenderer:
    def __init__(self):
        self.active_overlay_frames = []
        self.active_overlay_durations = []
        self.active_overlay_total_duration = 0.0
        self.overlay_started_at = 0.0
        self.overlay_until = 0.0

    def trigger_overlay(self, effect: MemeEffect):
        if effect.image_path is None:
            return

        if not effect.image_path.exists():
            print(f"Image missing: {effect.image_path}")
            return

        frames, durations = self._load_overlay_frames(effect.image_path)

        if not frames:
            print(f"Could not load image: {effect.image_path}")
            return

        self.active_overlay_frames = frames
        self.active_overlay_durations = durations
        self.active_overlay_total_duration = sum(durations)
        self.overlay_started_at = time.time()
        self.overlay_until = time.time() + effect.overlay_duration

    def render(self, frame):
        if not self.active_overlay_frames:
            return frame

        if time.time() > self.overlay_until:
            self._clear_overlay()
            return frame

        overlay = self._current_overlay_frame()

        if overlay is None:
            return frame

        return self._draw_overlay(frame, overlay)

    def _clear_overlay(self):
        self.active_overlay_frames = []
        self.active_overlay_durations = []
        self.active_overlay_total_duration = 0.0
        self.overlay_started_at = 0.0

    def _load_overlay_frames(self, path: Path):
        if path.suffix.lower() == ".gif":
            return self._load_gif_frames(path)

        overlay = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if overlay is None:
            return [], []

        return [overlay], [0.1]

    def _load_gif_frames(self, path: Path):
        frames = []
        durations = []

        try:
            from PIL import Image, ImageSequence

            with Image.open(path) as image:
                for frame in ImageSequence.Iterator(image):
                    rgba_frame = frame.convert("RGBA")
                    frame_array = np.array(rgba_frame)
                    bgra_frame = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGRA)

                    duration = frame.info.get("duration", image.info.get("duration", 100))
                    duration_seconds = max(float(duration) / 1000.0, 0.03)

                    frames.append(bgra_frame)
                    durations.append(duration_seconds)

        except Exception as error:
            print(f"Could not load GIF: {error}")
            return [], []

        return frames, durations

    def _current_overlay_frame(self):
        if not self.active_overlay_frames:
            return None

        if len(self.active_overlay_frames) == 1:
            return self.active_overlay_frames[0]

        if self.active_overlay_total_duration <= 0:
            return self.active_overlay_frames[0]

        elapsed = time.time() - self.overlay_started_at
        cycle_time = elapsed % self.active_overlay_total_duration
        accumulated = 0.0

        for frame, duration in zip(
            self.active_overlay_frames,
            self.active_overlay_durations,
        ):
            accumulated += duration

            if cycle_time <= accumulated:
                return frame

        return self.active_overlay_frames[-1]

    def _draw_overlay(self, frame, overlay):
        frame_h, frame_w = frame.shape[:2]

        target_w = int(frame_w * 0.45)
        scale = target_w / overlay.shape[1]
        target_h = int(overlay.shape[0] * scale)

        overlay_resized = cv2.resize(overlay, (target_w, target_h))

        x = frame_w - target_w - 30
        y = 30

        if x < 0 or y < 0 or x + target_w > frame_w or y + target_h > frame_h:
            return frame

        if len(overlay_resized.shape) == 3 and overlay_resized.shape[2] == 4:
            alpha = overlay_resized[:, :, 3] / 255.0
            overlay_rgb = overlay_resized[:, :, :3]

            for channel in range(3):
                frame[y:y + target_h, x:x + target_w, channel] = (
                    alpha * overlay_rgb[:, :, channel]
                    + (1 - alpha) * frame[y:y + target_h, x:x + target_w, channel]
                )
        else:
            frame[y:y + target_h, x:x + target_w] = overlay_resized

        return frame
