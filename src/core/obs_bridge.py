from dataclasses import dataclass
from typing import Optional
import threading
import time

try:
    import obsws_python as obs
except ImportError:
    obs = None


@dataclass
class ObsEffect:
    scene_name: str
    image_source_name: Optional[str] = None
    sound_source_name: Optional[str] = None
    duration: float = 1.5
    image_path: str | None = None
    sound_path: str | None = None
    sound_volume: float = 1.0


class OBSBridge:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 4455,
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.password = password
        self.client = None
        self.connected = False

    def connect(self) -> bool:
        if obs is None:
            print("obsws-python is not installed.")
            return False

        try:
            self.client = obs.ReqClient(
                host=self.host,
                port=self.port,
                password=self.password,
            )
            self.connected = True
            print("Connected to OBS websocket.")
            return True

        except Exception as error:
            print(f"Could not connect to OBS: {error}")
            self.connected = False
            return False

    def disconnect(self):
        self.client = None
        self.connected = False

    def trigger_effect(self, effect: ObsEffect):
        if not self.connected or self.client is None:
            return

        if effect.image_source_name and effect.image_path:
            self.set_image_source_file(effect.image_source_name, effect.image_path)

        if effect.sound_source_name and effect.sound_path:
            self.set_media_source_file(effect.sound_source_name, effect.sound_path)
            self.set_source_volume(effect.sound_source_name, effect.sound_volume)

        if effect.image_source_name:
            self.show_source(
                effect.scene_name,
                effect.image_source_name,
                duration=effect.duration,
            )

        if effect.sound_source_name:
            self.restart_media_source(effect.sound_source_name)

    def set_image_source_file(self, source_name: str, image_path: str):
        try:
            self.client.set_input_settings(
                source_name,
                {"file": image_path},
                True,
            )
        except Exception as error:
            print(f"Could not set OBS image source file: {error}")


    def set_media_source_file(self, source_name: str, media_path: str):
        try:
            self.client.set_input_settings(
                source_name,
                {
                    "local_file": media_path,
                    "is_local_file": True,
                },
                True,
            )
        except Exception as error:
            print(f"Could not set OBS media source file: {error}")


    def set_source_volume(self, source_name: str, volume: float):
        try:
            volume = max(0.0, min(float(volume), 2.0))

            self.client.set_input_volume(
                source_name,
                volume,
            )
        except Exception as error:
            print(f"Could not set OBS source volume: {error}")

    def show_source(self, scene_name: str, source_name: str, duration: float = 1.5):
        try:
            item_id = self._get_scene_item_id(scene_name, source_name)

            if item_id is None:
                print(f"OBS source not found: {source_name}")
                return

            self.client.set_scene_item_enabled(scene_name, item_id, True)

            threading.Thread(
                target=self._hide_later,
                args=(scene_name, item_id, duration),
                daemon=True,
            ).start()

        except Exception as error:
            print(f"Could not show OBS source: {error}")

    def restart_media_source(self, source_name: str):
        try:
            # Restart sound/video media source from beginning
            self.client.trigger_media_input_action(
                source_name,
                "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART",
            )
        except Exception as error:
            print(f"Could not restart OBS media source: {error}")

    def start_virtual_camera(self):
        try:
            self.client.start_virtual_cam()
            print("OBS virtual camera started.")
        except Exception as error:
            print(f"Could not start virtual camera: {error}")

    def stop_virtual_camera(self):
        try:
            self.client.stop_virtual_cam()
            print("OBS virtual camera stopped.")
        except Exception as error:
            print(f"Could not stop virtual camera: {error}")

    def _hide_later(self, scene_name: str, item_id: int, duration: float):
        time.sleep(duration)

        try:
            if self.connected and self.client is not None:
                self.client.set_scene_item_enabled(scene_name, item_id, False)
        except Exception as error:
            print(f"Could not hide OBS source: {error}")

    def _get_scene_item_id(self, scene_name: str, source_name: str) -> Optional[int]:
        try:
            response = self.client.get_scene_item_list(scene_name)

            for item in response.scene_items:
                if item["sourceName"] == source_name:
                    return item["sceneItemId"]

        except Exception as error:
            print(f"Could not get scene items: {error}")

        return None