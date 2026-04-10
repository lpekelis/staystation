import argparse
import time
from pathlib import Path
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
    parser.add_argument("--debug", action="store_true", help="Print timing info on every iteration")
    args = parser.parse_args()

    if args.viz:
        import cv2

    assert health_check(), "Inference server not reachable — start Docker first"

    Path("data").mkdir(exist_ok=True)

    motor = Motor()
    camera = Camera()
    conditioning = Conditioning(motor)

    camera.start()
    time.sleep(1)  # warm-up

    frame_count = 0
    try:
        while True:
            t_loop = time.perf_counter()

            t0 = time.perf_counter()
            frame = camera.capture_frame()
            t_capture = time.perf_counter() - t0

            t0 = time.perf_counter()
            detections = conditioning.step(frame)
            t_conditioning = time.perf_counter() - t0

            if args.viz:
                t0 = time.perf_counter()
                annotated = _draw_detections(frame.copy(), detections)
                cv2.imshow("StayStation", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                t_viz = time.perf_counter() - t0
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                t_viz = 0.0

            if args.debug:
                t_total = time.perf_counter() - t_loop
                frame_count += 1
                print(
                    f"frame={frame_count:4d} | "
                    f"capture={t_capture * 1000:6.1f}ms | "
                    f"conditioning={t_conditioning * 1000:6.1f}ms | "
                    f"viz={t_viz * 1000:6.1f}ms | "
                    f"total={t_total * 1000:6.1f}ms ({1 / t_total:.1f} fps)"
                )

            # Enforce 1-second step timing for consistent data recording
            elapsed = time.perf_counter() - t_loop
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        motor.cleanup()
        if args.viz:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
