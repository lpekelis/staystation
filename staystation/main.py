import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from staystation.buzzer import Buzzer
from staystation.camera import Camera
from staystation.conditioning import Conditioning
from staystation.inference_client import health_check
from staystation.motor import Motor
from staystation.model import (
    ConditioningModel1,
    ConditioningModel2,
    ConditioningModel3,
    ConditioningModel,
)


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
    parser.add_argument(
        "--model", type=int, choices=[1, 2, 3], default=1, help="Conditioning model (default: 1)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=30.0,
        help="Gaussian weight std dev over steps (default: 30.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Minimum e_treat - e_no_treat to dispense (default: 1.0)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Success window and refractory period in steps (default: 10)",
    )
    parser.add_argument(
        "--p-explore",
        type=float,
        default=0.05,
        help="Probability of random exploration (default: 0.05)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="[model 3] Treat cycles evaluated for level changes (default: 3)",
    )
    parser.add_argument(
        "--p-promote",
        type=float,
        default=0.67,
        help="[model 3] Success rate to promote a level (default: 0.67)",
    )
    parser.add_argument(
        "--p-demote",
        type=float,
        default=0.01,
        help="[model 3] Success rate below which to demote a level (default: 0.01)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--save-interval", type=int, default=10, help="Save dataset every N steps (default: 10)"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(name)s | %(message)s")
        logging.getLogger("staystation").setLevel(logging.DEBUG)

    if args.viz:
        import cv2

    assert health_check(), "Inference server not reachable — start Docker first"

    Path("data").mkdir(exist_ok=True)

    motor = Motor()
    buzzer = Buzzer()
    camera = Camera()
    shared = dict(
        sigma=args.sigma, gamma=args.gamma, horizon=args.horizon, p_explore=args.p_explore
    )
    model: ConditioningModel
    if args.model == 1:
        model = ConditioningModel1(**shared)
    elif args.model == 2:
        model = ConditioningModel2(**shared)
    else:
        model = ConditioningModel3(
            **shared, k=args.k, p_promote=args.p_promote, p_demote=args.p_demote
        )
    conditioning = Conditioning(
        motor,
        buzzer,
        model=model,
        confidence_threshold=args.confidence,
        save_path="data/dataset.csv",
        save_interval=args.save_interval,
    )

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
        buzzer.cleanup()
        if args.viz:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
