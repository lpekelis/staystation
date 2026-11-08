"""Unit tests for ConditioningModel4 (bandit logistic transition model)."""

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from staystation.bandit_model import (
    COEF_NAMES,
    DEFAULT_COEF,
    CatStore,
    ConditioningModel4,
    build_features,
    design_matrix,
    fit_map_logistic,
    rollout_prob,
)


def make_df(cats: list[bool], treats: list[bool] | None = None) -> pd.DataFrame:
    treats = treats or [False] * len(cats)
    return pd.DataFrame(
        {
            "step": range(len(cats)),
            "cat_detected": cats,
            "treat_dispensed": treats,
        }
    )


def default_coef() -> np.ndarray:
    return np.array([DEFAULT_COEF[k] for k in COEF_NAMES])


class TestBuildFeatures:
    def test_appends_current_row(self) -> None:
        feats = build_features(make_df([False, False]), 2, True, horizon=5, debounce=1)
        assert list(feats["step"]) == [0, 1, 2]
        assert bool(feats["on"].iloc[-1]) is True
        assert not bool(feats["treat"].iloc[-1])

    def test_drops_rows_at_or_after_current_step(self) -> None:
        feats = build_features(make_df([True, True, True]), 1, False, horizon=5, debounce=1)
        assert list(feats["step"]) == [0, 1]

    def test_tau_counts_time_in_state(self) -> None:
        feats = build_features(
            make_df([False, False, True, True, True]), 5, False, horizon=5, debounce=1
        )
        assert list(feats["tau"]) == [0, 1, 0, 1, 2, 0]

    def test_debounce_ignores_single_missed_frame(self) -> None:
        feats = build_features(make_df([True, True, False, True]), 4, True, horizon=5, debounce=2)
        assert feats["on"].all()
        assert list(feats["tau"]) == [0, 1, 2, 3, 4]

    def test_y_is_next_step_presence(self) -> None:
        feats = build_features(make_df([False, True, False]), 3, True, horizon=5, debounce=1)
        assert feats["y"].iloc[:-1].tolist() == [1.0, 0.0, 1.0]
        assert math.isnan(feats["y"].iloc[-1])

    def test_treat_recent_lasts_horizon_steps(self) -> None:
        treats = [False, True] + [False] * 6
        feats = build_features(make_df([True] * 8, treats), 8, True, horizon=3, debounce=1)
        assert feats["treat_recent"].tolist() == [False, True, True, True] + [False] * 5


class TestRolloutProb:
    def test_horizon_one_is_sigmoid(self) -> None:
        coef = default_coef()
        X = design_matrix(np.array([True]), np.array([4]), np.array([False]))
        expected = 1 / (1 + math.exp(-(X @ coef)[0]))
        got = rollout_prob(coef, True, np.array([4]), False, horizon=1)[0]
        assert got == pytest.approx(expected)

    def test_matches_monte_carlo(self) -> None:
        coef = default_coef()
        rng = np.random.default_rng(0)
        hits = 0
        n = 20000
        for _ in range(n):
            on, tau = 1.0, 3.0
            for _ in range(5):
                z = (
                    coef[0]
                    + coef[1] * on
                    + coef[2] * on * math.log1p(tau)
                    + coef[3] * (1 - on) * math.log1p(tau)
                )
                nxt = float(rng.random() < 1 / (1 + math.exp(-z)))
                tau = tau + 1 if nxt == on else 0.0
                on = nxt
            hits += int(on)
        got = rollout_prob(coef, True, np.array([3]), False, horizon=5)[0]
        assert got == pytest.approx(hits / n, abs=0.015)

    def test_treat_increases_probability(self) -> None:
        coef = default_coef()
        for on in (True, False):
            p0 = rollout_prob(coef, on, np.array([0]), False, 5)[0]
            p1 = rollout_prob(coef, on, np.array([0]), True, 5)[0]
            assert p1 > p0


class TestFitMapLogistic:
    def test_recovers_true_params_with_lots_of_data(self) -> None:
        rng = np.random.default_rng(1)
        n = 20000
        on = rng.random(n) < 0.5
        tau = rng.integers(0, 60, n)
        treat = rng.random(n) < 0.2
        X = design_matrix(on, tau, treat)
        true = np.array([-2.0, 4.0, -0.5, -0.2, 1.0])
        y = (rng.random(n) < 1 / (1 + np.exp(-X @ true))).astype(float)
        beta = fit_map_logistic(X, y, np.ones(n), np.zeros(5), prior_sd=10.0)
        np.testing.assert_allclose(beta, true, atol=0.2)

    def test_no_data_returns_prior_mean(self) -> None:
        prior = default_coef()
        beta = fit_map_logistic(np.zeros((0, 5)), np.zeros(0), np.zeros(0), prior, 1.0)
        np.testing.assert_allclose(beta, prior)


class TestConditioningModel4:
    def test_default_prior_is_unconditioned(self) -> None:
        m = ConditioningModel4()
        for on in (True, False):
            for tau in (0, 5, 60):
                assert m.prob(on, tau, False) < m.threshold
        assert m.treat_wait() == 0.0

    def test_treats_on_empty_dataset(self) -> None:
        assert ConditioningModel4().should_treat(True, make_df([]), 0)

    def test_fits_once_per_step(self) -> None:
        m = ConditioningModel4()
        df = make_df([True, True, False, False, True] * 4)
        m.fit(df, (1, 0), 20)
        coef, n_fits = m.coef_.copy(), m.n_fits
        m.fit(df, (0, 1), 20)
        np.testing.assert_array_equal(m.coef_, coef)
        assert m.n_fits == n_fits

    def test_predict_uses_condition(self) -> None:
        m = ConditioningModel4()
        df = make_df([True] * 10)
        p_no = m.fit(df, (1, 0), 10).predict()
        p_yes = m.fit(df, (1, 1), 10).predict()
        assert p_yes > p_no

    def test_ignores_rows_before_session_start(self) -> None:
        m = ConditioningModel4()
        m.should_treat(False, make_df([True] * 100), 100)  # session starts at step 100
        np.testing.assert_allclose(m.coef_, default_coef())

    def test_lure_treats_capped_since_cat_left(self) -> None:
        random.seed(0)
        m = ConditioningModel4(max_lure_treats=2)
        m._session_start = 0
        cats = [True] * 5 + [False] * 15
        treats = [False] * 20
        treats[8] = treats[14] = True  # two lure treats after the cat left at step 5
        df = make_df(cats, treats)
        assert m.treats_since_left(df, 20) == 2
        assert m.should_treat(False, df, 20) is False

    def test_lure_cap_resets_when_cat_returns(self) -> None:
        m = ConditioningModel4(max_lure_treats=2)
        m._session_start = 0
        # Leaves at 5, gets 2 lure treats (capped), returns at 20, leaves again at 25
        cats = [True] * 5 + [False] * 15 + [True] * 5 + [False] * 5
        treats = [False] * 30
        treats[8] = treats[14] = True
        df = make_df(cats, treats)
        assert m.treats_since_left(df, 20) == 2  # capped during the first absence
        assert m.treats_since_left(df, 30) == 0  # fresh budget after returning
        assert m.should_treat(False, df, 30) is True

    def test_lure_cap_does_not_apply_on_mat(self) -> None:
        m = ConditioningModel4(max_lure_treats=0)
        assert m.should_treat(True, make_df([]), 0)

    def test_keeps_unidentified_params_from_previous_fit(self) -> None:
        # Only off-mat data → on-mat coefficients stay near their prior/previous values.
        m = ConditioningModel4()
        m.should_treat(False, make_df([False] * 300), 300)
        assert m.coef["b_stay"] == pytest.approx(DEFAULT_COEF["b_stay"], abs=0.1)


class TestCatStore:
    def test_saves_and_resumes_params(self, tmp_path: Path) -> None:
        store = CatStore("whiskers", root=tmp_path)
        m = ConditioningModel4(store=store)
        df = make_df([True, True, False, True] * 10)
        m.should_treat(True, df, 40)

        saved = json.loads(store.params_path.read_text())
        assert saved["cat_id"] == "whiskers"
        assert saved["step"] == 40

        m2 = ConditioningModel4(store=CatStore("whiskers", root=tmp_path))
        np.testing.assert_allclose(m2.coef_, m.coef_)

    def test_new_cat_starts_from_default(self, tmp_path: Path) -> None:
        m = ConditioningModel4(store=CatStore("new", root=tmp_path))
        np.testing.assert_allclose(m.coef_, default_coef())

    def test_appends_diagnostics(self, tmp_path: Path) -> None:
        store = CatStore("whiskers", root=tmp_path)
        m = ConditioningModel4(store=store)
        df = make_df([True] * 20)
        m.should_treat(True, df, 20)
        m.should_treat(True, df, 21)
        # A new model instance (new launch) appends to the same file
        ConditioningModel4(store=CatStore("whiskers", root=tmp_path)).should_treat(True, df, 22)
        diag = pd.read_csv(store.diagnostics_path)
        assert list(diag["step"]) == [20, 21, 22]
        for col in ("p_no_treat", "treat_wait", "treats_since_left", "explored", *COEF_NAMES):
            assert col in diag.columns


def _simulate_conditioning_cat(n_steps: int, seed: int) -> pd.DataFrame:
    """A cat whose urge to leave the mat shrinks with every treat it gets on the mat."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    m = ConditioningModel4()
    steps: list[int] = []
    cats: list[bool] = []
    treats: list[bool] = []
    cat, tau, conditioning, eating, last_treat = False, 0, 0.0, 0, -100
    for s in range(n_steps):
        df = pd.DataFrame({"step": steps, "cat_detected": cats, "treat_dispensed": treats})
        treat = s - last_treat > m.horizon and m.should_treat(cat, df, s)
        if treat:
            last_treat = s
            if cat:
                conditioning += 0.03
                eating = 4
        steps.append(s)
        cats.append(cat)
        treats.append(treat)
        if cat:
            leave = 0.0 if eating else 0.15 * math.exp(-conditioning) * (1 + tau / 20)
            eating = max(eating - 1, 0)
            cat, tau = (False, 0) if rng.random() < leave else (True, tau + 1)
        else:
            arrive = 0.25 if s - last_treat < 3 else 0.01
            cat, tau = (True, 0) if rng.random() < arrive else (False, tau + 1)
    return pd.DataFrame({"step": steps, "cat_detected": cats, "treat_dispensed": treats})


def test_conditioned_cat_gets_fewer_treats_over_time() -> None:
    df = _simulate_conditioning_cat(1200, seed=0)
    early = df[df["step"] < 600]
    late = df[df["step"] >= 600]
    assert late["treat_dispensed"].sum() < 0.8 * early["treat_dispensed"].sum()
    assert late["cat_detected"].mean() >= early["cat_detected"].mean()
