from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from picamera2 import Picamera2


class Camera:
    def __init__(self, resolution: tuple[int, int] = (640, 480)) -> None:
        if sys.platform != "linux":
            raise RuntimeError("Camera requires picamera2, which is only available on Linux/Pi")
        from picamera2 import Picamera2

        self.picam: Picamera2 = Picamera2()
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
