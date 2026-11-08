from __future__ import annotations

import glob
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import cv2
    from picamera2 import Picamera2


def _find_usb_camera(name_substr: str) -> str:
    for name_file in sorted(glob.glob("/sys/class/video4linux/*/name")):
        card_name = Path(name_file).read_text().strip()
        if name_substr.lower() in card_name.lower():
            device = Path(name_file).parent.name
            return f"/dev/{device}"
    raise RuntimeError(f"No camera matching {name_substr!r} found in /sys/class/video4linux")


class BaseCamera(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def capture_frame(self) -> np.ndarray: ...

    @abstractmethod
    def stop(self) -> None: ...


class PiCamera(BaseCamera):
    def __init__(self, resolution: tuple[int, int] = (640, 480)) -> None:
        if sys.platform != "linux":
            raise RuntimeError("PiCamera requires picamera2, which is only available on Linux/Pi")
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


class UsbCamera(BaseCamera):
    def __init__(
        self,
        device: int | str = "auto",
        name_hint: str = "NexiGo",
        resolution: tuple[int, int] = (640, 480),
    ) -> None:
        import cv2

        if device == "auto":
            device = _find_usb_camera(name_hint)

        self._cap: cv2.VideoCapture = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera at {device!r}")

        width, height = resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def start(self) -> None:
        pass  # VideoCapture opens on construction

    def capture_frame(self) -> np.ndarray:
        import cv2

        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to capture frame from USB camera")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        self._cap.release()


# Type alias so existing annotations (Camera) keep working
Camera = BaseCamera


def make_camera(
    backend: Literal["picamera2", "usb"] = "picamera2",
    device: int | str = "auto",
    resolution: tuple[int, int] = (640, 480),
) -> BaseCamera:
    if backend == "picamera2":
        return PiCamera(resolution=resolution)
    return UsbCamera(device=device, resolution=resolution)
