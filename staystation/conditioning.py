from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from staystation.inference_client import detect
from staystation.model import ConditioningModel, ConditioningModel1
from staystation.motor import Motor


class Conditioning:
    def __init__(
        self,
        motor: Motor,
        model: ConditioningModel | None = None,
        confidence_threshold: float = 0.5,
        save_path: str | Path = "data/dataset.csv",
        save_interval: int = 10,
    ) -> None:
        self.motor = motor
        self.model = model or ConditioningModel1()
        self.confidence_threshold = confidence_threshold
        self.save_path = Path(save_path)
        self.save_interval = save_interval

        self.dataset = self._load()
        self.step_count = int(self.dataset["step"].max()) + 1 if len(self.dataset) > 0 else 0

        # Initialize last_treat_step from dataset or allow immediate treatment
        treat_rows = self.dataset[self.dataset["treat_dispensed"]]
        if len(treat_rows) > 0:
            self.last_treat_step = int(treat_rows["step"].max())
        else:
            self.last_treat_step = -self.model.horizon

    def step(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run one detect → decide → reward cycle. Returns detections."""
        detections = detect(frame, confidence=self.confidence_threshold)
        cat_detections = [d for d in detections if d["class_name"] == "cat"]
        cat_detected = len(cat_detections) > 0
        cat_confidence = max((d["confidence"] for d in cat_detections), default=0.0)

        # Refractory period: don't treat within `horizon` steps of last treat
        in_refractory = (self.step_count - self.last_treat_step) <= self.model.horizon

        treat = False
        decision = False
        if not in_refractory:
            treat = self.model.should_treat(cat_detected, self.dataset, self.step_count)
            decision = True

        if treat:
            self.last_treat_step = self.step_count
            print(
                f"[step {self.step_count}] Treating (cat={cat_detected} conf={cat_confidence:.2f})"
            )
            self.motor.dispense()

        # Record observation
        row = pd.DataFrame(
            [
                {
                    "step": self.step_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cat_detected": cat_detected,
                    "cat_confidence": cat_confidence,
                    "treat_dispensed": treat,
                    "made_decision": decision,
                }
            ]
        )
        self.dataset = pd.concat([self.dataset, row], ignore_index=True)
        self.step_count += 1

        if self.step_count % self.save_interval == 0:
            self._save()

        return detections

    def _load(self) -> pd.DataFrame:
        if self.save_path.exists():
            df = pd.read_csv(self.save_path)
            # Ensure correct dtypes
            df["cat_detected"] = df["cat_detected"].astype(bool)
            df["treat_dispensed"] = df["treat_dispensed"].astype(bool)
            return df
        return pd.DataFrame(
            {
                "step": pd.Series(dtype="int64"),
                "timestamp": pd.Series(dtype="str"),
                "cat_detected": pd.Series(dtype="bool"),
                "cat_confidence": pd.Series(dtype="float64"),
                "treat_dispensed": pd.Series(dtype="bool"),
                "made_decision": pd.Series(dtype="bool"),
            }
        )

    def _save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.to_csv(self.save_path, index=False)
