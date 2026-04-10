"""Unit tests for ConditioningModel1, ConditioningModel2, and ConditioningModel3."""

import math

import pandas as pd
import pytest

from staystation.model import ConditioningModel1, ConditioningModel2, ConditioningModel3

_TS = "2024-01-01T00:00:00+00:00"


def make_df(rows: list[tuple[int, bool, bool]]) -> pd.DataFrame:
    """Build a dataset DataFrame from (step, cat_detected, treat_dispensed) tuples."""
    if not rows:
        return pd.DataFrame(
            columns=["step", "timestamp", "cat_detected", "cat_confidence", "treat_dispensed"]
        ).astype({"cat_detected": bool, "treat_dispensed": bool, "step": int})
    return pd.DataFrame(
        [
            {
                "step": s,
                "timestamp": _TS,
                "cat_detected": c,
                "cat_confidence": 0.8 if c else 0.0,
                "treat_dispensed": t,
            }
            for s, c, t in rows
        ]
    ).astype({"cat_detected": bool, "treat_dispensed": bool})


# ── ConditioningModel1 ────────────────────────────────────────────────────────


class TestConditioningModel1GaussianWeight:
    def test_weight_is_one_at_current_step(self) -> None:
        m = ConditioningModel1(sigma=10.0)
        assert m._gaussian_weight(5, 5) == pytest.approx(1.0)

    def test_weight_decays_with_distance(self) -> None:
        m = ConditioningModel1(sigma=10.0)
        assert m._gaussian_weight(5, 10) > m._gaussian_weight(0, 10)

    def test_weight_is_symmetric(self) -> None:
        m = ConditioningModel1(sigma=10.0)
        assert m._gaussian_weight(0, 5) == pytest.approx(m._gaussian_weight(10, 5))

    def test_weight_below_threshold_for_distant_step(self) -> None:
        m = ConditioningModel1(sigma=1.0)
        assert m._gaussian_weight(0, 100) < 1e-3

    def test_weight_formula(self) -> None:
        m = ConditioningModel1(sigma=5.0)
        expected = math.exp(-0.5 * ((3 - 10) / 5.0) ** 2)
        assert m._gaussian_weight(3, 10) == pytest.approx(expected)


class TestConditioningModel1ComputeY:
    def test_all_cat_clean_window_is_success(self) -> None:
        m = ConditioningModel1(horizon=3, p_explore=0.0)
        df = make_df([(i, True, i == 0) for i in range(6)])
        y = m._compute_y(df)
        assert y.iloc[0] == pytest.approx(1.0)
        assert y.iloc[1] == pytest.approx(1.0)
        assert y.iloc[2] == pytest.approx(1.0)

    def test_no_cat_in_window_is_failure(self) -> None:
        m = ConditioningModel1(horizon=3, p_explore=0.0)
        df = make_df([(i, False, False) for i in range(6)])
        y = m._compute_y(df)
        assert y.iloc[0] == pytest.approx(0.0)

    def test_treat_in_future_window_are_not_counted(self) -> None:
        m = ConditioningModel1(horizon=3, p_explore=0.0)
        # treat at step 2 pollutes windows that look ahead to include step 2
        df = make_df(
            [
                (0, True, False),
                (1, False, False),
                (2, True, True),
                (3, False, False),
                (4, True, False),
                (5, True, True),
            ]
        )
        y = m._compute_y(df)
        # y[0] looks at steps 1,2,3 — step 2 has treat → not counted
        # however step 5 has cat and not treated so y[2] should be success
        assert y.iloc[0] == pytest.approx(0.0)
        assert y.iloc[1] == pytest.approx(0.0)
        assert y.iloc[2] == pytest.approx(1.0)

    def test_tail_rows_are_nan(self) -> None:
        m = ConditioningModel1(horizon=3, p_explore=0.0)
        df = make_df([(i, True, False) for i in range(6)])
        y = m._compute_y(df)
        assert y.iloc[-3:].isna().all()

    def test_returns_series_with_same_index(self) -> None:
        m = ConditioningModel1(horizon=2, p_explore=0.0)
        df = make_df([(i, True, False) for i in range(5)])
        y = m._compute_y(df)
        assert list(y.index) == list(df.index)


class TestConditioningModel1Fit:
    def test_empty_df_falls_back_to_priors(self) -> None:
        m = ConditioningModel1(p_explore=0.0)
        m.fit(make_df([]), (1, 1), 0)
        a, b = m.priors[(1, 1)]
        assert m.alpha_ == pytest.approx(a)
        assert m.beta_ == pytest.approx(b)

    def test_no_matching_condition_falls_back_to_priors(self) -> None:
        m = ConditioningModel1(horizon=3, p_explore=0.0)
        # Only (cat=False, treat=False) rows — condition (1, 1) has no matches
        df = make_df([(i, False, False) for i in range(10)])
        m.fit(df, (1, 1), 20)
        a, b = m.priors[(1, 1)]
        assert m.alpha_ == pytest.approx(a)
        assert m.beta_ == pytest.approx(b)

    def test_all_successes_makes_alpha_dominant(self) -> None:
        m = ConditioningModel1(horizon=3, sigma=1000.0, p_explore=0.0)
        # condition (1,1): treat at every 4th step, followed by 3 steps with cat
        rows = []
        for base in range(0, 40, 4):
            rows.append((base, True, True))
            rows += [(base + j, True, False) for j in range(1, 4)]
        m.fit(make_df(rows), (1, 1), 50)
        assert m.alpha_ > m.beta_

    def test_all_failures_makes_beta_dominant(self) -> None:
        m = ConditioningModel1(horizon=3, sigma=1000.0, p_explore=0.0)
        rows = []
        for base in range(0, 40, 4):
            rows.append((base, True, True))
            rows += [(base + j, False, False) for j in range(1, 4)]
        m.fit(make_df(rows), (1, 1), 50)
        assert m.beta_ > m.alpha_

    def test_fit_returns_self(self) -> None:
        m = ConditioningModel1(p_explore=0.0)
        result = m.fit(make_df([]), (0, 0), 0)
        assert result is m


class TestConditioningModel1Predict:
    def test_predict_returns_alpha_over_total(self) -> None:
        m = ConditioningModel1()
        m.alpha_ = 3.0
        m.beta_ = 1.0
        assert m.predict() == pytest.approx(0.75)

    def test_predict_prior_is_sensible(self) -> None:
        m = ConditioningModel1(p_explore=0.0)
        m.fit(make_df([]), (1, 1), 0)
        p = m.predict()
        assert 0.0 < p < 1.0


class TestConditioningModel1ShouldTreat:
    def test_exploration_always_returns_bool(self) -> None:
        m = ConditioningModel1(p_explore=1.0)
        results = [m.should_treat(True, make_df([]), 0) for _ in range(20)]
        assert all(isinstance(r, bool) for r in results)

    def test_exploration_is_approximately_fair_coin(self) -> None:
        m = ConditioningModel1(p_explore=1.0)
        results = [m.should_treat(True, make_df([]), 0) for _ in range(200)]
        assert 60 < sum(results) < 140

    def test_no_data_uses_priors(self) -> None:
        m = ConditioningModel1(p_explore=0.0)
        result = m.should_treat(True, make_df([]), 0)
        assert isinstance(result, bool)


# ── ConditioningModel2 ────────────────────────────────────────────────────────


class TestConditioningModel2Fit:
    def test_empty_df_falls_back_to_priors(self) -> None:
        m = ConditioningModel2(p_explore=0.0)
        m.fit(make_df([]), (1, 1), 0)
        a, b = m.priors[1]
        assert m.alpha_ == pytest.approx(a)
        assert m.beta_ == pytest.approx(b)

    def test_ignores_cat_dimension(self) -> None:
        rows = []
        for base in range(0, 40, 4):
            rows.append((base, True, True))
            rows += [(base + j, True, False) for j in range(1, 4)]
        df = make_df(rows)

        m1 = ConditioningModel2(horizon=3, sigma=1000.0, p_explore=0.0)
        m1.fit(df, (1, 1), 50)

        m2 = ConditioningModel2(horizon=3, sigma=1000.0, p_explore=0.0)
        m2.fit(df, (0, 1), 50)  # different cat dimension, same treat dimension

        assert m1.alpha_ == pytest.approx(m2.alpha_)
        assert m1.beta_ == pytest.approx(m2.beta_)

    def test_all_successes_makes_alpha_dominant(self) -> None:
        m = ConditioningModel2(horizon=3, sigma=1000.0, p_explore=0.0)
        rows = []
        for base in range(0, 40, 4):
            rows.append((base, True, True))
            rows += [(base + j, True, False) for j in range(1, 4)]
        m.fit(make_df(rows), (1, 1), 50)
        assert m.alpha_ > m.beta_

    def test_fit_returns_self(self) -> None:
        m = ConditioningModel2(p_explore=0.0)
        assert m.fit(make_df([]), (0, 0), 0) is m


class TestConditioningModel2Predict:
    def test_predict_returns_alpha_over_total(self) -> None:
        m = ConditioningModel2()
        m.alpha_ = 2.0
        m.beta_ = 6.0
        assert m.predict() == pytest.approx(0.25)


class TestConditioningModel2ShouldTreat:
    def test_exploration_is_approximately_fair_coin(self) -> None:
        m = ConditioningModel2(p_explore=1.0)
        results = [m.should_treat(True, make_df([]), 0) for _ in range(200)]
        assert 60 < sum(results) < 140


# ── ConditioningModel3 ────────────────────────────────────────────────────────


class TestConditioningModel3TreatSteps:
    def test_extracts_treat_steps_sorted(self) -> None:
        m = ConditioningModel3()
        df = make_df([(0, True, True), (1, True, False), (2, True, True), (3, False, False)])
        assert m._treat_steps(df) == [0, 2]

    def test_empty_df_returns_empty(self) -> None:
        m = ConditioningModel3()
        assert m._treat_steps(make_df([])) == []

    def test_no_treats_returns_empty(self) -> None:
        m = ConditioningModel3()
        df = make_df([(i, True, False) for i in range(5)])
        assert m._treat_steps(df) == []


class TestConditioningModel3Success:
    def test_cat_in_window_is_success(self) -> None:
        m = ConditioningModel3(horizon=3)
        df = make_df([(0, True, True), (1, True, False), (2, True, False), (3, True, False)])
        assert m._success(df, 0) is True

    def test_no_cat_in_window_is_failure(self) -> None:
        m = ConditioningModel3(horizon=3)
        df = make_df([(0, True, True), (1, False, False), (2, False, False), (3, False, False)])
        assert m._success(df, 0) is False

    def test_partial_cat_in_window_is_success(self) -> None:
        m = ConditioningModel3(horizon=3)
        df = make_df([(0, True, True), (1, False, False), (2, True, False), (3, False, False)])
        assert m._success(df, 0) is True


class TestConditioningModel3CheckGraduation:
    def _df_with_k_successes(self, k: int, horizon: int) -> pd.DataFrame:
        rows = []
        for base in range(0, k * (horizon + 1), horizon + 1):
            rows.append((base, True, True))
            rows += [(base + j, True, False) for j in range(1, horizon + 1)]
        return make_df(rows)

    def test_graduates_after_k_successes(self) -> None:
        m = ConditioningModel3(horizon=2, k=3, p_graduate=0.67, p_explore=0.0)
        df = self._df_with_k_successes(3, 2)
        m._check_graduation(df, current_step=50)
        assert m.level == 1

    def test_does_not_graduate_on_failures(self) -> None:
        m = ConditioningModel3(horizon=2, k=3, p_graduate=0.67, p_explore=0.0)
        rows = []
        for base in range(0, 9, 3):
            rows.append((base, True, True))
            rows += [(base + j, False, False) for j in range(1, 3)]
        m._check_graduation(make_df(rows), current_step=50)
        assert m.level == 0

    def test_resets_level_start_treat_idx_after_graduation(self) -> None:
        m = ConditioningModel3(horizon=2, k=3, p_graduate=0.67, p_explore=0.0)
        df = self._df_with_k_successes(3, 2)
        m._check_graduation(df, current_step=50)
        assert m._level_start_treat_idx == 3

    def test_does_not_graduate_without_enough_future_data(self) -> None:
        m = ConditioningModel3(horizon=3, k=3, p_graduate=0.67, p_explore=0.0)
        # treat at step 0, but current_step=1 so 0 + 3 = 3 is not < 1
        df = make_df([(0, True, True), (1, True, False)])
        m._check_graduation(df, current_step=1)
        assert m.level == 0

    def test_second_graduation_requires_new_treats(self) -> None:
        m = ConditioningModel3(horizon=2, k=3, p_graduate=0.67, p_explore=0.0)
        df = self._df_with_k_successes(3, 2)
        m._check_graduation(df, current_step=50)
        assert m.level == 1
        # calling again without new treats should not graduate further
        m._check_graduation(df, current_step=50)
        assert m.level == 1


class TestConditioningModel3Fit:
    def test_no_eligible_treats_returns_uniform_prior(self) -> None:
        m = ConditioningModel3(horizon=3, p_explore=0.0)
        m.fit(make_df([]), (0, 0), 0)
        assert m.alpha_ == pytest.approx(1.0)
        assert m.beta_ == pytest.approx(1.0)

    def test_all_successes_makes_alpha_dominant(self) -> None:
        m = ConditioningModel3(horizon=2, k=3, p_explore=0.0)
        rows = []
        for base in range(0, 9, 3):
            rows.append((base, True, True))
            rows += [(base + j, True, False) for j in range(1, 3)]
        m.fit(make_df(rows), (0, 0), 20)
        assert m.alpha_ > m.beta_

    def test_all_failures_makes_beta_dominant(self) -> None:
        m = ConditioningModel3(horizon=2, k=3, p_explore=0.0)
        rows = []
        for base in range(0, 9, 3):
            rows.append((base, True, True))
            rows += [(base + j, False, False) for j in range(1, 3)]
        m.fit(make_df(rows), (0, 0), 20)
        assert m.beta_ > m.alpha_

    def test_fit_returns_self(self) -> None:
        m = ConditioningModel3(p_explore=0.0)
        assert m.fit(make_df([]), (0, 0), 0) is m


class TestConditioningModel3Predict:
    def test_predict_returns_alpha_over_total(self) -> None:
        m = ConditioningModel3()
        m.alpha_ = 4.0
        m.beta_ = 2.0
        assert m.predict() == pytest.approx(4.0 / 6.0)


class TestConditioningModel3ShouldTreat:
    def test_level0_always_treats(self) -> None:
        m = ConditioningModel3(p_explore=0.0)
        assert m.level == 0
        assert m.should_treat(True, make_df([]), 0) is True
        assert m.should_treat(False, make_df([]), 0) is True

    def test_level1_treats_when_horizon_elapsed(self) -> None:
        m = ConditioningModel3(horizon=3, p_explore=0.0)
        m.level = 1
        df = make_df([(0, True, True)])
        assert m.should_treat(True, df, 2) is False  # 2 < 1*3
        assert m.should_treat(True, df, 3) is True  # 3 >= 1*3

    def test_level2_requires_double_horizon(self) -> None:
        m = ConditioningModel3(horizon=3, p_explore=0.0)
        m.level = 2
        df = make_df([(0, True, True)])
        assert m.should_treat(True, df, 5) is False  # 5 < 2*3
        assert m.should_treat(True, df, 6) is True  # 6 >= 2*3

    def test_no_prior_treats_at_higher_level_treats_immediately(self) -> None:
        m = ConditioningModel3(horizon=3, p_explore=0.0)
        m.level = 2
        # No treats in df → last_treat defaults to current_step - level*horizon → gap = level*horizon
        assert m.should_treat(True, make_df([]), 10) is True

    def test_exploration_is_approximately_fair_coin(self) -> None:
        m = ConditioningModel3(p_explore=1.0)
        results = [m.should_treat(True, make_df([]), 0) for _ in range(200)]
        assert 60 < sum(results) < 140
