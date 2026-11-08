import argparse
import logging
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from staystation.bandit_model import CatStore, ConditioningModel4
from staystation.buzzer import Buzzer
from staystation.camera import make_camera, parse_camera_device
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
        "--model", type=int, choices=[1, 2, 3, 4], default=1, help="Conditioning model (default: 1)"
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
        default=None,
        help="Success window and refractory period in steps (default: 10, model 4: 5)",
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
        "--cat-id",
        default=None,
        help="[model 4] Cat id: loads/saves fitted params and diagnostics under "
        "data/cats/<id>/. If omitted, a new timestamped id is generated for this run.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=600,
        help="[model 4] Steps (~seconds) of history used in each fit (default: 600)",
    )
    parser.add_argument(
        "--half-life",
        type=float,
        default=300.0,
        help="[model 4] Age half-life in steps for down-weighting observations (default: 300)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.66,
        help="[model 4] Treat when P(cat on mat in 5s | no treat) is below this (default: 0.66)",
    )
    parser.add_argument(
        "--explore-interval",
        type=float,
        default=180.0,
        help="[model 4] Exploration rate: ~1 extra treat per this many steps (default: 180)",
    )
    parser.add_argument(
        "--max-lure-treats",
        type=int,
        default=5,
        help="[model 4] Max treats released while the cat is off the mat (default: 5)",
    )
    parser.add_argument(
        "--camera",
        choices=["picamera2", "usb"],
        default="picamera2",
        help="Camera backend (default: picamera2)",
    )
    parser.add_argument(
        "--camera-device",
        default="auto",
        help="USB device index, path (e.g. /dev/video2), or 'auto' to detect by name (default: auto)",
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
    if args.cat_id and args.model != 4:
        print(f"Warning: --cat-id is only used by --model 4 (got --model {args.model})")

    if args.debug:
        logging.basicConfig(format="%(name)s | %(message)s")
        logging.getLogger("staystation").setLevel(logging.DEBUG)

    if args.viz:
        import cv2

    assert health_check(), "Inference server not reachable — start Docker first"

    Path("data").mkdir(exist_ok=True)

    motor = Motor()
    buzzer = Buzzer()
    camera = make_camera(backend=args.camera, device=parse_camera_device(args.camera_device))
    horizon = args.horizon if args.horizon is not None else (5 if args.model == 4 else 10)
    shared = dict(sigma=args.sigma, gamma=args.gamma, horizon=horizon, p_explore=args.p_explore)
    model: ConditioningModel
    if args.model == 4:
        cat_id = args.cat_id or (f"cat-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}")
        store = CatStore(cat_id)
        resumed = store.params_path.exists()
        print(
            f"Cat id: {cat_id} ({'resuming saved params' if resumed else 'new params'}) "
            f"→ {store.dir}"
        )
        model = ConditioningModel4(
            horizon=horizon,
            lookback=args.lookback,
            half_life=args.half_life,
            threshold=args.threshold,
            explore_interval=args.explore_interval,
            max_lure_treats=args.max_lure_treats,
            store=store,
        )
    elif args.model == 1:
        model = ConditioningModel1(**shared)
    elif args.model == 2:
        model = ConditioningModel2(**shared)
    elif args.model == 3:
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
