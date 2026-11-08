"""Model 4: bandit-style conditioning with a refit logistic model.

At every decision step we refit a weighted logistic regression on a recent
lookback window. It is a discrete-time transition (hazard) model:

    P(on mat at t+1 | on_mat_t, time_on/away_t, treat_recent_t)
      = sigmoid(b0 + b_on·on + b_stay·on·log1p(τ_on) + b_away·off·log1p(τ_away)
                + b_treat·treat_recent)

where treat_recent = a treat was dispensed in the last `horizon` steps. The
decision quantity, P(cat on mat `horizon` steps from now | no treat), is
obtained by rolling this 1-step model forward exactly over all 2^horizon paths.

Why 1-step instead of fitting the 5-step outcome directly: the policy reacts to
the cat (e.g. lures it as soon as it leaves), so any "exclude rows whose
outcome window contains a later treat" filter selects on the outcome and badly
biases the fit. Every 1-step transition is usable as-is.

The fit is a MAP estimate under a Gaussian prior centered on the *previous*
fit (or the "unconditioned" defaults on first launch), so coefficients that the
current window has no data for keep their last value. Fitted parameters and
per-fit diagnostics are persisted per cat id so learning carries across runs.

Policy: treat when P(y | state, no treat) < threshold, otherwise treat with a
small exploration probability. While the cat is off the mat, at most
`max_lure_treats` treats are released since it last left.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

from staystation.model import ConditioningModel

logger = logging.getLogger(__name__)

COEF_NAMES: tuple[str, ...] = ("b0", "b_on", "b_stay", "b_away", "b_treat")

# 1-step prior implying an unconditioned cat. Rolled forward 5 steps (no treat):
#   on mat:  P(on in 5s) ≈ 0.48 at τ=0, 0.29 at τ=10, 0.13 at τ=120
#   off mat: P(on in 5s) ≈ 0.05
# so P(y | no treat) < 0.66 in every state → treat freely. With a treat the
# on-mat probability rises to ≈0.85 (τ=0).
DEFAULT_COEF: dict[str, float] = {
    "b0": -3.9,  # off mat: ~2%/s chance the cat arrives
    "b_on": 6.2,  # just arrived: ~91%/s chance it stays another second
    "b_stay": -0.3,  # the longer on the mat, the likelier to leave
    "b_away": -0.3,  # the longer away, the less likely to come back
    "b_treat": 1.5,  # a recent treat makes the cat likelier to be on the mat
}

TREAT_WAIT_MAX = 1800  # treat_wait() scans time-on-mat up to this many steps


def _sigmoid(z: np.ndarray | float) -> Any:
    return 1.0 / (1.0 + np.exp(-z))


def build_features(
    df: pd.DataFrame,
    current_step: int,
    cat_now: bool,
    horizon: int,
    debounce: int,
) -> pd.DataFrame:
    """Build per-row features and 1-step targets, with a synthetic row for `current_step`.

    `df` must be contiguous rows ordered by step. Rows at or after
    `current_step` are dropped and replaced by one row with `cat_now` and no
    treat, so the last row always describes the current decision state.

    Columns: step, on, tau, treat, treat_recent, y.
      on           — debounced presence (detected in any of the last `debounce` steps)
      tau          — steps spent in the current on/off state (0 on the first step)
      treat_recent — a treat was dispensed in [t - horizon + 1, t]
      y            — `on` at t + 1 (NaN for the last row)
    """
    past = df[df["step"] < current_step]
    steps = np.append(past["step"].to_numpy(dtype=np.int64), current_step)
    raw = np.append(past["cat_detected"].to_numpy(dtype=bool), cat_now)
    treat = np.append(past["treat_dispensed"].to_numpy(dtype=bool), False)

    on = pd.Series(raw).rolling(debounce, min_periods=1).max().to_numpy().astype(bool)
    run_id = np.cumsum(np.r_[True, on[1:] != on[:-1]])
    tau = pd.Series(run_id).groupby(run_id).cumcount().to_numpy()
    treat_recent = pd.Series(treat).rolling(horizon, min_periods=1).max().to_numpy().astype(bool)

    y = np.full(len(steps), np.nan)
    y[:-1] = on[1:]

    return pd.DataFrame(
        {
            "step": steps,
            "on": on,
            "tau": tau,
            "treat": treat,
            "treat_recent": treat_recent,
            "y": y,
        }
    )


def design_matrix(on: np.ndarray, tau: np.ndarray, treat: np.ndarray) -> np.ndarray:
    on_f = on.astype(float)
    log_tau = np.log1p(tau.astype(float))
    return np.column_stack(
        [np.ones_like(on_f), on_f, on_f * log_tau, (1.0 - on_f) * log_tau, treat.astype(float)]
    )


def rollout_prob(
    coef: np.ndarray, on: bool, tau: np.ndarray, treat: bool, horizon: int
) -> np.ndarray:
    """Exact P(on mat at t + horizon) from state (on, tau) under the 1-step model.

    Enumerates all 2^horizon on/off paths. `treat=True` means a treat is
    dispensed now, so treat_recent = 1 for the whole window; `treat=False`
    assumes no treat now or within the window (true at decision points, which
    are outside the refractory period). Vectorized over `tau`.
    """
    tau = np.asarray(tau, dtype=float)
    total = np.zeros_like(tau)
    tr = float(treat)
    for path in range(2**horizon):
        o = float(on)
        t = tau.copy()
        prob = np.ones_like(tau)
        for k in range(horizon):
            bit = (path >> k) & 1
            log_t = np.log1p(t)
            z = coef[0] + coef[1] * o + coef[2] * o * log_t + coef[3] * (1 - o) * log_t
            p = _sigmoid(z + coef[4] * tr)
            prob *= p if bit else 1.0 - p
            t = np.where(bit == o, t + 1, 0.0)
            o = float(bit)
        if o == 1.0:
            total += prob
    return total


def fit_map_logistic(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    prior_mean: np.ndarray,
    prior_sd: float,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Weighted logistic regression with an N(prior_mean, prior_sd²) prior (Newton's method)."""
    beta = prior_mean.copy()
    prior_prec = 1.0 / prior_sd**2
    for _ in range(max_iter):
        p = _sigmoid(X @ beta)
        grad = X.T @ (w * (p - y)) + prior_prec * (beta - prior_mean)
        hess = (X * (w * p * (1 - p))[:, None]).T @ X + prior_prec * np.eye(len(beta))
        delta = np.linalg.solve(hess, grad)
        beta -= delta
        if np.max(np.abs(delta)) < tol:
            break
    return beta


class CatStore:
    """Persists fitted params (params.json) and per-fit diagnostics (diagnostics.csv)."""

    def __init__(self, cat_id: str, root: str | Path = "data/cats") -> None:
        self.cat_id = cat_id
        self.dir = Path(root) / cat_id
        self.params_path = self.dir / "params.json"
        self.diagnostics_path = self.dir / "diagnostics.csv"
        self._diag_fields: list[str] | None = None

    def load_coef(self) -> dict[str, float] | None:
        if not self.params_path.exists():
            return None
        data = json.loads(self.params_path.read_text())
        return {k: float(data["coef"][k]) for k in COEF_NAMES}

    def save_coef(self, coef: dict[str, float], step: int, n_fits: int) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "cat_id": self.cat_id,
            "updated": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "n_fits": n_fits,
            "coef": coef,
        }
        tmp = self.params_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.params_path)  # atomic: never leaves a half-written file

    def append_diagnostics(self, row: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self._diag_fields is None:
            if self.diagnostics_path.exists() and self.diagnostics_path.stat().st_size > 0:
                with self.diagnostics_path.open() as f:
                    self._diag_fields = next(csv.reader(f))
            else:
                self._diag_fields = list(row)
                with self.diagnostics_path.open("w", newline="") as f:
                    csv.DictWriter(f, self._diag_fields).writeheader()
        with self.diagnostics_path.open("a", newline="") as f:
            csv.DictWriter(f, self._diag_fields, extrasaction="ignore").writerow(row)


class ConditioningModel4(ConditioningModel):
    """Bandit conditioning model refitting a logistic regression every decision step.

    Steps are assumed to be ~1 second (main.py runs at 1 Hz), so step counts
    below double as seconds.
    """

    def __init__(
        self,
        horizon: int = 5,
        lookback: int = 600,
        half_life: float = 300.0,
        threshold: float = 0.66,
        explore_interval: float = 180.0,
        max_lure_treats: int = 5,
        debounce: int = 2,
        prior_sd: float = 1.0,
        init_coef: dict[str, float] | None = None,
        store: CatStore | None = None,
    ) -> None:
        self.horizon = horizon
        self.lookback = lookback
        self.half_life = half_life
        self.threshold = threshold
        self.p_explore = 1.0 / explore_interval
        self.max_lure_treats = max_lure_treats
        self.debounce = debounce
        self.prior_sd = prior_sd
        self.store = store

        loaded = store.load_coef() if store is not None else None
        if loaded is not None:
            logger.info("model4 loaded params for cat %s: %s", store.cat_id, loaded)  # type: ignore[union-attr]
        start = loaded or init_coef or DEFAULT_COEF
        self.coef_: np.ndarray = np.array([start[k] for k in COEF_NAMES], dtype=float)

        self.n_fits: int = 0
        self._fitted_step: int | None = None
        self._session_start: int | None = None
        self._last_n_obs: int = 0
        self._last_eff_n: float = 0.0
        self._state: tuple[bool, int] = (False, 0)  # (on_mat, tau) at the fitted step
        self._predict_treat: bool = False

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def coef(self) -> dict[str, float]:
        return {k: float(v) for k, v in zip(COEF_NAMES, self.coef_)}

    def prob(self, on_mat: bool, tau: int, treat: bool) -> float:
        """P(cat on mat `horizon` steps from now | state, treat now)."""
        return float(rollout_prob(self.coef_, on_mat, np.array([tau]), treat, self.horizon)[0])

    def treat_wait(self) -> float:
        """Steps on the mat before P(on in `horizon` | no treat) first drops below threshold.

        0 → treat as soon as the cat arrives; inf → not within TREAT_WAIT_MAX
        steps (only exploration treats a sitting cat).
        """
        taus = np.arange(TREAT_WAIT_MAX + 1)
        below = rollout_prob(self.coef_, True, taus, False, self.horizon) < self.threshold
        return float(np.argmax(below)) if below.any() else math.inf

    def _session_df(self, df: pd.DataFrame, current_step: int) -> pd.DataFrame:
        if self._session_start is None:
            self._session_start = current_step
        return df[(df["step"] >= self._session_start) & (df["step"] < current_step)]

    def treats_since_left(self, df: pd.DataFrame, current_step: int) -> int:
        """Treats released since the cat was last seen (or since this session started)."""
        session = self._session_df(df, current_step)
        seen = session.loc[session["cat_detected"].astype(bool), "step"]
        start = int(seen.max()) + 1 if len(seen) else (self._session_start or 0)
        return int(session.loc[session["step"] >= start, "treat_dispensed"].astype(bool).sum())

    # ── ConditioningModel API ────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        condition: tuple[int, int],
        current_step: int,
    ) -> Self:
        """Refit coefficients (once per step) and set the state used by predict().

        condition = (cat_detected_now, treat).
        """
        session = self._session_df(df, current_step)
        # Extra history beyond the lookback so time-on/off-mat isn't truncated early.
        recent = session.tail(2 * self.lookback)
        feats = build_features(
            recent, current_step, bool(condition[0]), self.horizon, self.debounce
        )
        self._state = (bool(feats["on"].iloc[-1]), int(feats["tau"].iloc[-1]))
        self._predict_treat = bool(condition[1])

        if self._fitted_step == current_step:
            return self

        rows = feats[feats["y"].notna() & (feats["step"] >= current_step - self.lookback)]
        prior_mean = self.coef_.copy()
        if len(rows) > 0:
            age = current_step - rows["step"].to_numpy(dtype=float)
            w = 0.5 ** (age / self.half_life)
            X = design_matrix(
                rows["on"].to_numpy(), rows["tau"].to_numpy(), rows["treat_recent"].to_numpy()
            )
            y = rows["y"].to_numpy(dtype=float)
            self.coef_ = fit_map_logistic(X, y, w, prior_mean, self.prior_sd)
            self._last_eff_n = float(w.sum())
        else:
            self._last_eff_n = 0.0
        self._last_n_obs = len(rows)
        self._fitted_step = current_step
        self.n_fits += 1

        if self.store is not None:
            self.store.save_coef(self.coef, current_step, self.n_fits)
        return self

    def predict(self) -> float:
        """P(cat on mat in `horizon` steps | state and treat from the last fit)."""
        on_mat, tau = self._state
        return self.prob(on_mat, tau, self._predict_treat)

    def should_treat(
        self,
        cat_detected: bool,
        df: pd.DataFrame,
        current_step: int,
    ) -> bool:
        self.fit(df, (int(cat_detected), 0), current_step)
        on_mat, tau = self._state
        p_no_treat = self.prob(on_mat, tau, False)
        p_treat = self.prob(on_mat, tau, True)

        lure_count = 0 if on_mat else self.treats_since_left(df, current_step)
        capped = (not on_mat) and lure_count >= self.max_lure_treats

        explored = False
        if capped:
            treat = False
        elif p_no_treat < self.threshold:
            treat = True
        else:
            explored = random.random() < self.p_explore
            treat = explored

        wait = self.treat_wait()
        logger.debug(
            "model4 on=%s tau=%d p_no_treat=%.3f p_treat=%.3f wait=%.1f lure=%d capped=%s "
            "explore=%s n=%d coef=%s → treat=%s",
            on_mat,
            tau,
            p_no_treat,
            p_treat,
            wait,
            lure_count,
            capped,
            explored,
            self._last_n_obs,
            {k: round(v, 3) for k, v in self.coef.items()},
            treat,
        )

        if self.store is not None:
            self.store.append_diagnostics(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": current_step,
                    "n_fits": self.n_fits,
                    **self.coef,
                    "n_obs": self._last_n_obs,
                    "eff_n": round(self._last_eff_n, 2),
                    "cat_detected": cat_detected,
                    "on_mat": on_mat,
                    "tau": tau,
                    "p_no_treat": round(p_no_treat, 4),
                    "p_treat": round(p_treat, 4),
                    "treat_wait": wait,
                    "treats_since_left": lure_count,
                    "capped": capped,
                    "explored": explored,
                    "treat": treat,
                }
            )
        return treat
