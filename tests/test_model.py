"""Visualize conditioning model behavior across different mock scenarios."""

import logging
from datetime import datetime, timezone
from typing import NamedTuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for headless environments

from staystation.model import (
    ConditioningModel,
    ConditioningModel1,
    ConditioningModel2,
    ConditioningModel3,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

_COLUMNS = [
    "step",
    "timestamp",
    "cat_detected",
    "cat_confidence",
    "treat_dispensed",
    "made_decision",
]


def _make_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class SimResult(NamedTuple):
    dataset: pd.DataFrame
    e_treat_1: list[float]  # E[y | C=1, T=1] over time
    e_no_treat_1: list[float]  # E[y | C=1, T=0] over time
    e_treat_0: list[float]  # E[y | C=0, T=1] over time
    e_no_treat_0: list[float]  # E[y | C=0, T=0] over time
    delta_1: list[float]  # E[T=1|C=1] - E[T=0|C=1]
    delta_0: list[float]  # E[T=1|C=0] - E[T=0|C=0]
    treat_steps: list[int]


def simulate(
    model: ConditioningModel,
    n_steps: int,
    p_table: dict[tuple[int, int], float],
    rng: np.random.Generator,
    scenario_name: str = "",
) -> SimResult:
    """Run a simulation with 4 transition probabilities.

    p_table keys are (cat_detected, treat_dispensed) states.
    Values are P(cat present at next step | state).
    """
    rows: list[dict[str, object]] = []
    e_treat_1, e_no_treat_1 = [], []
    e_treat_0, e_no_treat_0 = [], []
    delta_1, delta_0 = [], []
    treat_steps: list[int] = []

    cat = True
    last_treat_step = -model.horizon

    log.debug("=== Scenario: %s ===", scenario_name)
    log.debug("p_table: %s", p_table)
    log.debug(
        "%-6s  %-4s  %-5s  %-9s  %-10s  %-10s  %-10s  %-10s  %-8s  %-8s",
        "step",
        "cat",
        "treat",
        "refract",
        "E[C1,T1]",
        "E[C1,T0]",
        "E[C0,T1]",
        "E[C0,T0]",
        "delta_1",
        "delta_0",
    )

    for step in range(n_steps):
        cat_conf = float(rng.uniform(0.6, 0.95)) if cat else 0.0
        df = pd.DataFrame(rows, columns=_COLUMNS) if rows else pd.DataFrame(columns=_COLUMNS)

        # Refractory check
        in_refractory = (step - last_treat_step) <= model.horizon

        treat = False
        decision = False
        if not in_refractory:
            treat = model.should_treat(cat, df, step)
            decision = True

        if treat:
            treat_steps.append(step)
            last_treat_step = step

        # Record
        rows.append(
            {
                "step": step,
                "timestamp": _make_timestamp(),
                "cat_detected": cat,
                "cat_confidence": cat_conf,
                "treat_dispensed": treat,
                "made_decision": decision,
            }
        )
        df = pd.DataFrame(rows, columns=_COLUMNS)

        # Track predictions for both C=0 and C=1 conditions
        model.fit(df, (1, 1), step)
        _e_treat_1 = model.predict()
        model.fit(df, (1, 0), step)
        _e_no_treat_1 = model.predict()
        model.fit(df, (0, 1), step)
        _e_treat_0 = model.predict()
        model.fit(df, (0, 0), step)
        _e_no_treat_0 = model.predict()

        e_treat_1.append(_e_treat_1)
        e_no_treat_1.append(_e_no_treat_1)
        e_treat_0.append(_e_treat_0)
        e_no_treat_0.append(_e_no_treat_0)
        _delta_1 = _e_treat_1 - _e_no_treat_1
        _delta_0 = _e_treat_0 - _e_no_treat_0
        delta_1.append(_delta_1)
        delta_0.append(_delta_0)

        log.debug(
            "%-6d  %-4s  %-5s  %-9s  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-8.4f  %-8.4f",
            step,
            "T" if cat else "F",
            "T" if treat else "F",
            "T" if in_refractory else "F",
            _e_treat_1,
            _e_no_treat_1,
            _e_treat_0,
            _e_no_treat_0,
            _delta_1,
            _delta_0,
        )

        # Transition: next cat state depends on current (cat, treat) state
        cat = bool(rng.random() < p_table[(int(cat), int(treat))])

    return SimResult(
        df, e_treat_1, e_no_treat_1, e_treat_0, e_no_treat_0, delta_1, delta_0, treat_steps
    )


_MODEL_CLASSES: dict[int, type[ConditioningModel]] = {
    1: ConditioningModel1,
    2: ConditioningModel2,
    3: ConditioningModel3,
}


def _make_model(version: int, horizon: int, gamma: float) -> ConditioningModel:
    cls = _MODEL_CLASSES[version]
    if version == 3:
        return ConditioningModel3(horizon=horizon)
    return cls(sigma=30.0, gamma=gamma, horizon=horizon)  # type: ignore[call-arg]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Simulate and visualize conditioning model behavior."
    )
    parser.add_argument(
        "--model", type=int, choices=[1, 2, 3], default=2, help="Model version to use (default: 2)"
    )
    args = parser.parse_args()

    n_steps = 200
    rng = np.random.default_rng(42)
    horizon = 5
    gamma = 0.1

    # p_table keys: (cat_detected, treat_dispensed)
    scenarios = [
        (
            "A: Treat lures/keeps cat (treat helps)",
            {(0, 0): 0.10, (0, 1): 0.60, (1, 0): 0.30, (1, 1): 0.90},
        ),
        (
            "B: Cat stays regardless (treat irrelevant)",
            {(0, 0): 0.50, (0, 1): 0.50, (1, 0): 0.70, (1, 1): 0.70},
        ),
        (
            "C: Cat always leaves (treat does nothing)",
            {(0, 0): 0.05, (0, 1): 0.10, (1, 0): 0.10, (1, 1): 0.10},
        ),
        (
            "D: Treat only lures cat but doesn't keep it (treat helps but not as much)",
            {(0, 0): 0.10, (0, 1): 0.50, (1, 0): 0.10, (1, 1): 0.10},
        ),
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(15, 5 * len(scenarios)))

    for i, (title, p_table) in enumerate(scenarios):
        model = _make_model(args.model, horizon, gamma)
        result = simulate(model, n_steps, p_table, rng, scenario_name=title)

        slug = title.split(":")[0].lower()
        csv_path = f"data/sim_{slug}_{ts}.csv"
        result.dataset.to_csv(csv_path, index=False)
        log.debug("Saved dataset to %s", csv_path)

        steps = list(range(n_steps))
        cat_present = result.dataset["cat_detected"].astype(bool).values

        # ── Left plot: predictions over time ──────────────────────────────────
        ax1 = axes[i, 0]

        # Cat presence background shading (collected as one span per run)
        in_run = False
        run_start = 0
        for s in range(n_steps):
            if cat_present[s] and not in_run:
                run_start = s
                in_run = True
            elif not cat_present[s] and in_run:
                ax1.axvspan(run_start - 0.5, s - 0.5, alpha=0.15, color="green", linewidth=0)
                in_run = False
        if in_run:
            ax1.axvspan(run_start - 0.5, n_steps - 0.5, alpha=0.15, color="green", linewidth=0)

        # C=1 lines (solid)
        ax1.plot(steps, result.e_treat_1, color="blue", linestyle="-", label="E[y | C=1, T=1]")
        ax1.plot(steps, result.e_no_treat_1, color="red", linestyle="-", label="E[y | C=1, T=0]")
        # C=0 lines (dashed)
        ax1.plot(steps, result.e_treat_0, color="blue", linestyle="--", label="E[y | C=0, T=1]")
        ax1.plot(steps, result.e_no_treat_0, color="red", linestyle="--", label="E[y | C=0, T=0]")

        # Treat rug plot at the bottom
        if result.treat_steps:
            ax1.plot(
                result.treat_steps,
                [0.02] * len(result.treat_steps),
                "|",
                color="orange",
                markersize=10,
                markeredgewidth=1.5,
                label="Treat dispensed",
            )

        # Legend with cat shading patch
        cat_patch = mpatches.Patch(facecolor="green", alpha=0.3, label="Cat present")
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(
            handles=handles + [cat_patch],
            labels=labels + ["Cat present"],
            loc="upper right",
            fontsize=7,
        )
        ax1.set_ylabel("P(cat present in next horizon steps)")
        ax1.set_title(title)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel("Step")

        # ── Right plot: decision boundaries ───────────────────────────────────
        ax2 = axes[i, 1]
        ax2.plot(steps, result.delta_1, color="purple", linestyle="-", label="delta | C=1")
        ax2.plot(steps, result.delta_0, color="darkviolet", linestyle="--", label="delta | C=0")
        ax2.axhline(y=gamma, color="gray", linestyle="--", linewidth=1.5, label=f"gamma={gamma}")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.set_ylabel("Treat benefit (E[T=1] - E[T=0])")
        ax2.set_title(f"{title} — Decision boundaries")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_xlabel("Step")

    plt.tight_layout()
    out_path = f"data/model_trajectories_{ts}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
