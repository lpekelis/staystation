from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Self

import pandas as pd

# Default priors: Beta(a, b) for each (cat_detected, treat_dispensed) state
DEFAULT_PRIORS: dict[tuple[int, int], tuple[float, float]] = {
    (0, 0): (1.0, 4.0),  # no cat, no treat → likely stays absent
    (0, 1): (2.0, 3.0),  # no cat, treat → treat might attract
    (1, 0): (2.0, 3.0),  # cat present, no treat → might leave
    (1, 1): (4.0, 1.0),  # cat present, treat → likely stays
}

# Default priors for ConditioningModel2: Beta(a, b) for each treat_dispensed state
DEFAULT_PRIORS_2: dict[int, tuple[float, float]] = {
    0: (2.0, 3.0),  # no treat → cat might leave
    1: (4.0, 1.0),  # treat → cat likely stays
}


def _compute_y(df: pd.DataFrame, horizon: int, do_cleaning: bool = False) -> pd.Series:
    """Check if cat was detected within `horizon` steps after each step.

    Detections must be clean (i.e. not influenced by a treat within the horizon) to count as successes at each step.
    """

    def is_success(group: pd.DataFrame) -> float:
        y = group["cat_detected"]
        is_clean = group["treat_dispensed"].cummax() == 0
        if do_cleaning:
            y = y[is_clean]
        return float(y.any())

    y = [is_success(window) for window in df.rolling(window=horizon, on="step")]

    return pd.Series(y, index=df.index).shift(-horizon)


class ConditioningModel(ABC):
    """Abstract base class for conditioning models.

    All models share the sklearn-style fit/predict/should_treat API and must
    expose a `horizon` attribute (number of steps in the success window and
    refractory period).
    """

    horizon: int

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        condition: tuple[int, int],
        current_step: int,
    ) -> Self:
        """Update the model's fitted state given the dataset and condition."""

    @abstractmethod
    def predict(self) -> float:
        """Return the model's probability estimate from the most recent fit."""

    @abstractmethod
    def should_treat(
        self,
        cat_detected: bool,
        df: pd.DataFrame,
        current_step: int,
    ) -> bool:
        """Return True if a treat should be dispensed at the current step."""


class ConditioningModel1(ConditioningModel):
    """Beta-Binomial conditioning model conditioning on (cat_detected, treat_dispensed).

    Estimates P(cat stays within `horizon` steps | current state) for each of
    the 4 possible states (cat_detected, treat_dispensed). Uses Gaussian-weighted
    observations for time-varying posterior estimation.
    """

    def __init__(
        self,
        sigma: float = 30.0,
        gamma: float = 0.05,
        horizon: int = 3,
        p_explore: float = 0.05,
        priors: dict[tuple[int, int], tuple[float, float]] | None = None,
    ) -> None:
        self.sigma = sigma
        self.gamma = gamma
        self.horizon = horizon
        self.p_explore = p_explore
        self.priors = priors or DEFAULT_PRIORS

        self.alpha_: float = 0.0
        self.beta_: float = 0.0

    def _gaussian_weight(self, step: int, current_step: int) -> float:
        return math.exp(-0.5 * ((step - current_step) / self.sigma) ** 2)

    def _compute_y(self, df: pd.DataFrame) -> pd.Series:
        """Check if cat was detected within `horizon` steps after each step.

        Detections must be clean (i.e. not influenced by a treat within the horizon) to count as successes at each step.
        """
        return _compute_y(df, self.horizon)

    def _compute_weights(self, df: pd.DataFrame, current_step: int) -> pd.Series:
        """Compute Gaussian weights for each step based on distance from current_step."""
        weights = df["step"].apply(lambda s: self._gaussian_weight(int(s), current_step))
        return weights

    def fit(
        self,
        df: pd.DataFrame,
        condition: tuple[int, int],
        current_step: int,
    ) -> Self:
        """Fit beta-binomial posterior for a single condition (C, T).

        Filters the dataset to rows matching the condition, computes the success
        metric y (any cat detection within `horizon` steps), and estimates
        posterior parameters using Gaussian-weighted observations.
        """
        a, b = self.priors[condition]

        if df.empty:
            self.alpha_ = a
            self.beta_ = b
            return self

        df2 = df.assign(y=self._compute_y(df), weights=self._compute_weights(df, current_step))

        mask = (
            (df2["cat_detected"] == bool(condition[0]))
            & (df2["treat_dispensed"] == bool(condition[1]))
            & (~df2["y"].isna())
            & (df2["weights"] > 1e-3)
        )

        condition_df = df2[mask]

        weights = condition_df["weights"].values
        y_values = condition_df["y"].values

        n = len(weights)
        if n == 0:
            self.alpha_ = a
            self.beta_ = b
            return self

        w_sum = sum(weights)
        weighted_mean = sum(w * y for w, y in zip(weights, y_values)) / w_sum
        y_count = n * weighted_mean

        self.alpha_ = y_count + a
        self.beta_ = n - y_count + b

        return self

    def predict(self) -> float:
        """Return E[y | condition] from the most recent fit."""
        return self.alpha_ / (self.alpha_ + self.beta_)

    def should_treat(
        self,
        cat_detected: bool,
        df: pd.DataFrame,
        current_step: int,
    ) -> bool:
        """Return True if treating has sufficient expected benefit over not treating."""
        if random.random() < self.p_explore:
            return random.random() < 0.5

        c = int(cat_detected)

        self.fit(df, (c, 1), current_step)
        e_treat = self.predict()

        self.fit(df, (c, 0), current_step)
        e_no_treat = self.predict()

        return (e_treat - e_no_treat) > self.gamma


class ConditioningModel2(ConditioningModel):
    """Beta-Binomial conditioning model with two distributions: T=0 and T=1.

    Estimates P(cat present within `horizon` steps | treat_dispensed) without
    conditioning on whether the cat is currently detected. Uses Gaussian-weighted
    observations for time-varying posterior estimation.
    """

    def __init__(
        self,
        sigma: float = 30.0,
        gamma: float = 0.05,
        horizon: int = 3,
        p_explore: float = 0.05,
        priors: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        self.sigma = sigma
        self.gamma = gamma
        self.horizon = horizon
        self.p_explore = p_explore
        self.priors = priors or DEFAULT_PRIORS_2

        self.alpha_: float = 0.0
        self.beta_: float = 0.0

    def _gaussian_weight(self, step: int, current_step: int) -> float:
        return math.exp(-0.5 * ((step - current_step) / self.sigma) ** 2)

    def _compute_y(self, df: pd.DataFrame) -> pd.Series:
        """Check if cat was detected in every step within `horizon` steps after each row.

        Any treat in the forward window marks the observation as contaminated (NaN).
        """
        return _compute_y(df, self.horizon)

    def _compute_weights(self, df: pd.DataFrame, current_step: int) -> pd.Series:
        """Compute Gaussian weights for each step based on distance from current_step."""
        return df["step"].apply(lambda s: self._gaussian_weight(int(s), current_step))

    def fit(
        self,
        df: pd.DataFrame,
        condition: tuple[int, int],
        current_step: int,
    ) -> Self:
        """Fit beta-binomial posterior for treat state T=condition[1].

        Ignores the cat_detected dimension of condition; filters only on
        treat_dispensed. Uses Gaussian-weighted observations.
        """
        t = condition[1]
        a, b = self.priors[t]

        if df.empty:
            self.alpha_ = a
            self.beta_ = b
            return self

        df2 = df.assign(
            y=self._compute_y(df),
            weights=self._compute_weights(df, current_step),
        )

        mask = (df2["treat_dispensed"] == bool(t)) & (~df2["y"].isna()) & (df2["weights"] > 1e-3)

        condition_df = df2[mask]
        weights = condition_df["weights"].values
        y_values = condition_df["y"].values

        n = len(weights)
        if n == 0:
            self.alpha_ = a
            self.beta_ = b
            return self

        w_sum = sum(weights)
        weighted_mean = sum(w * y for w, y in zip(weights, y_values)) / w_sum
        y_count = n * weighted_mean

        self.alpha_ = y_count + a
        self.beta_ = n - y_count + b

        return self

    def predict(self) -> float:
        """Return E[y | condition] from the most recent fit."""
        return self.alpha_ / (self.alpha_ + self.beta_)

    def should_treat(
        self,
        cat_detected: bool,
        df: pd.DataFrame,
        current_step: int,
    ) -> bool:
        """Return True if treating has sufficient expected benefit over not treating."""
        if random.random() < self.p_explore:
            return random.random() < 0.5

        c = int(cat_detected)

        self.fit(df, (c, 1), current_step)
        e_treat = self.predict()

        self.fit(df, (c, 0), current_step)
        e_no_treat = self.predict()

        return (e_treat - e_no_treat) > self.gamma


class ConditioningModel3(ConditioningModel):
    """Heuristic level-based conditioning model.

    At level 0, dispenses a treat on every available step to attract the cat.
    At level L (≥1), dispenses every L refractory periods (L * horizon steps).

    Graduates to the next level when the last k eligible treat cycles at the
    current level have success rate (cat detected within horizon steps) > p_graduate.
    """

    def __init__(
        self,
        sigma: float = 30.0,
        gamma: float = 0.05,
        horizon: int = 3,
        p_explore: float = 0.05,
        k: int = 3,
        p_graduate: float = 0.67,
    ) -> None:
        self.sigma = sigma
        self.gamma = gamma
        self.horizon = horizon
        self.p_explore = p_explore
        self.k = k
        self.p_graduate = p_graduate

        self.level: int = 0
        self._level_start_treat_idx: int = 0

        self.alpha_: float = 1.0
        self.beta_: float = 1.0

    def _treat_steps(self, df: pd.DataFrame) -> list[int]:
        return sorted(int(s) for s in df[df["treat_dispensed"]]["step"].values)

    def _success(self, df: pd.DataFrame, treat_step: int) -> bool:
        """True if cat was detected within horizon steps after treat_step."""
        future = df[(df["step"] > treat_step) & (df["step"] <= treat_step + self.horizon)]
        return bool(future["cat_detected"].any())

    def _check_graduation(self, df: pd.DataFrame, current_step: int) -> None:
        """Advance level if last k eligible treat cycles at this level succeeded."""
        treats = self._treat_steps(df)
        level_treats = treats[self._level_start_treat_idx :]
        eligible = [s for s in level_treats if s + self.horizon < current_step]
        if len(eligible) < self.k:
            return
        successes = sum(1 for s in eligible[-self.k :] if self._success(df, s))
        if successes / self.k > self.p_graduate:
            self.level += 1
            self._level_start_treat_idx = len(treats)

    def fit(
        self,
        df: pd.DataFrame,
        condition: tuple[int, int],
        current_step: int,
    ) -> Self:
        """Estimate success rate from last k eligible treat cycles."""
        treats = self._treat_steps(df)
        eligible = [s for s in treats if s + self.horizon < current_step]
        last_k = eligible[-self.k :] if eligible else []
        n = len(last_k)
        if n == 0:
            self.alpha_ = 1.0
            self.beta_ = 1.0
            return self
        successes = sum(1 for s in last_k if self._success(df, s))
        self.alpha_ = float(successes) + 1.0
        self.beta_ = float(n - successes) + 1.0
        return self

    def predict(self) -> float:
        """Return success rate estimate from the most recent fit."""
        return self.alpha_ / (self.alpha_ + self.beta_)

    def should_treat(
        self,
        cat_detected: bool,
        df: pd.DataFrame,
        current_step: int,
    ) -> bool:
        """Decide whether to dispense based on current level and timing.

        Level 0: always dispense (refractory period handled externally).
        Level L: dispense only if L * horizon steps have passed since last treat.
        """
        if random.random() < self.p_explore:
            return random.random() < 0.5

        self._check_graduation(df, current_step)

        if self.level == 0:
            return True

        treats = self._treat_steps(df)
        last_treat = treats[-1] if treats else current_step - self.level * self.horizon
        return (current_step - last_treat) >= self.level * self.horizon
