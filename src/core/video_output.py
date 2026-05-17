import platform
import cv2

class VideoOutput:
    def start(self, width: int, height: int):
        pass

    def send_frame(self, frame):
        pass

    def close(self):
        pass


class NullVideoOutput(VideoOutput):
    pass


class SpoutVideoOutput(VideoOutput):
    def __init__(self, sender_name="Reactify"):
        self.sender_name = sender_name
        self.sender = None
        self.width = None
        self.height = None
        self.enabled = platform.system() == "Windows"

    def start(self, width: int, height: int):
        if not self.enabled:
            print("[SPOUT] Disabled: Spout is Windows-only.")
            return

        try:
            import SpoutGL
            from SpoutGL.enums import GL_RGB

            self.SpoutGL = SpoutGL
            self.gl_format = GL_RGB
            self.sender = SpoutGL.SpoutSender()
            self.sender.setSenderName(self.sender_name)
            self.width = width
            self.height = height

            print(f"[SPOUT] Sender started: {self.sender_name}")

        except Exception as error:
            self.sender = None
            print(f"[SPOUT] Could not start sender: {error}")

    def send_frame(self, frame):
        if self.sender is None:
            return

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            height, width = rgb_frame.shape[:2]

            self.sender.sendImage(
                rgb_frame,
                width,
                height,
                self.gl_format,
                False,
                0,
            )

        except Exception as error:
            print(f"[SPOUT] send_frame failed: {error}")

    def close(self):
        if self.sender is not None:
            try:
                self.sender.releaseSender()
            except Exception:
                pass

        self.sender = None