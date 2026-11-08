from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import cv2
    from picamera2 import Picamera2


def _read_sysfs(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _find_usb_camera(name_hint: str) -> str:
    """Find a USB (UVC) webcam's capture node, e.g. /dev/video0.

    On a Pi, /sys/class/video4linux also lists codec/ISP nodes, and webcams
    often report a generic name (e.g. "USB Camera") rather than the brand.
    So we only consider uvcvideo devices, skip their metadata nodes
    (index != 0), prefer one whose name contains `name_hint`, and otherwise
    fall back to the first USB camera found.
    """
    seen: list[str] = []
    usb_nodes: list[tuple[str, str]] = []  # (dev path, card name)
    nodes = sorted(
        Path("/sys/class/video4linux").glob("video*"),
        key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else 0,
    )
    for node in nodes:
        card_name = _read_sysfs(node / "name")
        driver = (node / "device" / "driver").resolve().name
        seen.append(f"/dev/{node.name}: {card_name!r} (driver={driver})")
        if driver != "uvcvideo":
            continue
        if _read_sysfs(node / "index") not in ("", "0"):
            continue  # metadata node, not a capture node
        usb_nodes.append((f"/dev/{node.name}", card_name))

    for dev, card_name in usb_nodes:
        if name_hint.lower() in card_name.lower():
            return dev
    if usb_nodes:
        return usb_nodes[0][0]

    listing = "\n  ".join(seen) or "(none)"
    raise RuntimeError(
        "No USB camera found. Video devices:\n  "
        f"{listing}\n"
        "Pass an explicit device, e.g. --camera-device /dev/video0"
    )


def parse_camera_device(value: str) -> int | str:
    """Parse a CLI device arg: 'auto', an integer index, or a /dev path."""
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        return value


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
