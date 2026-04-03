import argparse
import time
from typing import Any

import numpy as np

from staystation.camera import Camera
from staystation.conditioning import Conditioning
from staystation.inference_client import health_check
from staystation.motor import Motor


def _draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    import cv2

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        color = (0, 255, 0) if d["class_name"] == "cat" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="StayStation")
    parser.add_argument("--viz", action="store_true", help="Show live camera feed with detections")
    args = parser.parse_args()

    if args.viz:
        import cv2

    assert health_check(), "Inference server not reachable — start Docker first"

    motor = Motor()
    camera = Camera()
    conditioning = Conditioning(motor)

    camera.start()
    time.sleep(1)  # warm-up

    try:
        while True:
            frame = camera.capture_frame()
            detections = conditioning.step(frame)

            if args.viz:
                annotated = _draw_detections(frame.copy(), detections)
                cv2.imshow("StayStation", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        motor.cleanup()
        if args.viz:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
