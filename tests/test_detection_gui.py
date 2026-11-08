"""OpenCV GUI test script — requires a display (local or ssh -X)."""

import argparse
import time
from typing import Any

import cv2
import numpy as np

from staystation.camera import make_camera, parse_camera_device
from staystation.inference_client import detect, health_check


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        color = (0, 255, 0) if d["class_name"] == "cat" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Live detection test with OpenCV window")
    parser.add_argument(
        "--camera",
        choices=["picamera2", "usb"],
        default="picamera2",
        help="Camera backend (default: picamera2)",
    )
    parser.add_argument(
        "--camera-device",
        default="auto",
        help="USB device index, path (e.g. /dev/video0), or 'auto' (default: auto)",
    )
    args = parser.parse_args()

    assert health_check(), "Inference server not running — start Docker first"

    cam = make_camera(
        backend=args.camera,
        device=parse_camera_device(args.camera_device),
        resolution=(640, 480),
    )
    cam.start()
    time.sleep(1)  # warm-up

    frame_count = 0

    try:
        while True:
            t_frame_start = time.perf_counter()

            t0 = time.perf_counter()
            frame = cam.capture_frame()
            t_capture = time.perf_counter() - t0

            t0 = time.perf_counter()
            detections = detect(frame, confidence=0.4)
            t_inference = time.perf_counter() - t0

            t0 = time.perf_counter()
            annotated = draw_detections(frame.copy(), detections)
            cv2.imshow("StayStation - Detection Test", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            t_display = time.perf_counter() - t0

            t_total = time.perf_counter() - t_frame_start
            frame_count += 1

            print(
                f"frame={frame_count:4d} | "
                f"capture={t_capture * 1000:6.1f}ms | "
                f"inference={t_inference * 1000:6.1f}ms | "
                f"display={t_display * 1000:6.1f}ms | "
                f"total={t_total * 1000:6.1f}ms ({1 / t_total:.1f} fps)"
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
