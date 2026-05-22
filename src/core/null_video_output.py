
class NullVideoOutput:
    def __init__(self):
        self.started = False

    def start(self, width: int, height: int):
        self.started = True
        print(f"[NullVideoOutput] started ({width}x{height})")

    def send_frame(self, frame):
        # Do nothing
        pass

    def close(self):
        if self.started:
            print("[NullVideoOutput] closed")
            self.started = False