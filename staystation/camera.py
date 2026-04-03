import numpy as np
from picamera2 import Picamera2


class Camera:
    def __init__(self, resolution: tuple[int, int] = (640, 480)):
        self.picam = Picamera2()
        config = self.picam.create_still_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self.picam.configure(config)

    def start(self) -> None:
        self.picam.start()

    def capture_frame(self) -> np.ndarray:
        return self.picam.capture_array()

    def stop(self) -> None:
        self.picam.stop()
