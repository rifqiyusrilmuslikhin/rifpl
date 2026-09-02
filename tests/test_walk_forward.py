"""Acceptance tests for the Sprint 6 walk-forward evaluation harness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpl_model.evaluation import (
    EvaluationConfigError,
    EvaluationPlan,
    EvaluationWindow,
    ExpandingWindowSplitter,
    MetricInputError,
    PairingError,
    WalkForwardHarness,
    WindowRole,
    assert_exact_same_rows,
    bootstrap_gameweek_deltas,
    compare_ranking_predictions,
    diagnostic_cohorts,
    load_evaluation_plan,
    probability_metrics_by_gameweek,
    ranking_metrics_by_gameweek,
    regenerate_ranking_report,
    summarize_seed_metrics,
)


def _window(
    name: str,
    role: WindowRole,
    start: tuple[str, int],
    end: tuple[str, int],
    *,
    calibration: str | None = None,
) -> EvaluationWindow:
    return EvaluationWindow(
        name,
        role,
        start[0],
        start[1],
        end[0],
        end[1],
        selection_allowed=role is WindowRole.DISCOVERY,
        calibration_window=calibration,
    )


def _plan() -> EvaluationPlan:
    return EvaluationPlan(
        "test",
        ("season", "gameweek", "player_key"),
        (
            _window("warmup", WindowRole.WARMUP, ("2022-23", 1), ("2022-23", 38)),
            _window("calibration", WindowRole.CALIBRATION, ("2023-24", 1), ("2023-24", 1)),
            _window(
                "discovery",
                WindowRole.DISCOVERY,
                ("2023-24", 2),
                ("2023-24", 3),
                calibration="calibration",
            ),
        ),
    )


def _evaluation_frame() -> pd.DataFrame:
    rows = []
    periods = [("2022-23", 38), ("2023-24", 1), ("2023-24", 2), ("2023-24", 3)]
    start = datetime(2023, 5, 20, 10, tzinfo=UTC)
    for period_index, (season, gameweek) in enumerate(periods):
        for player_index, position in enumerate(("GKP", "DEF", "MID", "FWD")):
            actual = player_index + period_index
            rows.append(
                {
                    "season": season,
                    "gameweek": gameweek,
                    "player_key": f"player-{player_index}",
                    "deadline_utc": start + timedelta(days=period_index * 7),
                    "signal": float(actual) + 0.25,
                    "actual_points_gw": actual,
                    "actual_minutes_gw": 90 if player_index != 0 else 0,
                    "position_at_deadline": position,
                    "fixture_count": 2 if player_index == 3 else 1,
                    "status": "d" if player_index == 0 else "a",
                    "eligibility": True,
                    "baseline_points": 0.0,
                }
            )
    # This locally available later season must never enter an earlier fold.
    rows.append(
        {
            "season": "2025-26",
            "gameweek": 1,
            "player_key": "future-player",
            "deadline_utc": start + timedelta(days=800),
            "signal": 999.0,
            "actual_points_gw": 99,
            "actual_minutes_gw": 90,
            "position_at_deadline": "MID",
            "fixture_count": 1,
            "status": "a",
            "eligibility": True,
            "baseline_points": 0.0,
        }
    )
    return pd.DataFrame(rows)


class _DummyPredictor:
    def __init__(self, observed_training_seasons: list[set[str]]) -> None:
        self.observed_training_seasons = observed_training_seasons
        self.fitted_row_keys: list[tuple[object, ...]] = []

    def fit(
        self,
        train_frame: pd.DataFrame,
        calibration_frame: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        target_columns: tuple[str, ...],
    ) -> None:
        assert feature_columns == ("signal",)
        assert target_columns == ("actual_points_gw",)
        assert set(calibration_frame["season"]) == {"2023-24"}
        self.observed_training_seasons.append(set(train_frame["season"]))
        self.fitted_row_keys = list(
            train_frame[["season", "gameweek", "player_key"]].itertuples(index=False, name=None)
        )

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert set(frame) == {"season", "gameweek", "player_key", "signal"}
        result = frame[["season", "gameweek", "player_key"]].copy()
        result["predicted_points"] = frame["signal"].to_numpy()
        return result


def test_dummy_model_runs_end_to_end_and_retained_predictions_regenerate(tmp_path: Path) -> None:
    observed: list[set[str]] = []
    harness = WalkForwardHarness(ExpandingWindowSplitter(_plan()))
    artifact = harness.run(
        _evaluation_frame(),
        lambda seed, fold: _DummyPredictor(observed),
        feature_columns=("signal",),
        target_columns=("actual_points_gw",),
        prediction_columns=("predicted_points",),
        baseline_columns=("baseline_points",),
        context_columns=(
            "actual_minutes_gw",
            "position_at_deadline",
            "fixture_count",
            "status",
        ),
    )

    assert observed == [{"2022-23"}]
    assert set(artifact.frame["season"]) == {"2023-24"}
    before = regenerate_ranking_report(artifact)[("discovery", 42)].summary
    path, _ = artifact.write(tmp_path / "dummy_oof.parquet", metadata={"model": "dummy"})
    loaded = type(artifact).read(path)
    after = regenerate_ranking_report(loaded)[("discovery", 42)].summary
    assert before == after


def test_metrics_are_calculated_per_gw_not_as_pooled_primary() -> None:
    frame = pd.DataFrame(
        {
            "season": ["2024-25"] * 4,
            "gameweek": [1, 1, 1, 2],
            "player_key": ["a", "b", "c", "a"],
            "actual_points_gw": [0.0, 0.0, 0.0, 0.0],
            "predicted_points": [0.0, 0.0, 0.0, 8.0],
        }
    )
    report = ranking_metrics_by_gameweek(frame)

    assert report.summary["mae"] == pytest.approx(4.0)
    assert np.mean(np.abs(frame["predicted_points"] - frame["actual_points_gw"])) == 2.0
    assert report.aggregation == "mean_of_gameweeks"


def test_one_canonical_dgw_row_is_counted_once_and_fixture_duplicates_are_rejected() -> None:
    frame = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "gameweek": [10, 10],
            "player_key": ["dgw-player", "sgw-player"],
            "actual_points_gw": [12, 3],
            "predicted_points": [9.0, 4.0],
        }
    )
    report = ranking_metrics_by_gameweek(frame)
    assert report.by_gameweek.loc[0, "row_count"] == 2

    duplicate_fixture_rows = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(MetricInputError, match="aggregate DGW fixtures"):
        ranking_metrics_by_gameweek(duplicate_fixture_rows)


def test_paired_comparison_rejects_any_row_mismatch() -> None:
    candidate = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "gameweek": [1, 1],
            "player_key": ["a", "b"],
            "actual_points_gw": [1, 2],
        }
    )
    baseline = candidate.iloc[[0]].copy()
    with pytest.raises(PairingError, match="exact same rows"):
        assert_exact_same_rows(candidate, baseline)


def test_paired_comparison_and_seed_noise_keep_gw_and_fold_evidence() -> None:
    candidate = pd.DataFrame(
        {
            "season": ["2024-25"] * 4,
            "gameweek": [1, 1, 2, 2],
            "player_key": ["a", "b", "a", "b"],
            "actual_points_gw": [5, 1, 2, 7],
            "predicted_points": [4.0, 2.0, 3.0, 6.0],
        }
    )
    baseline = candidate.copy()
    baseline["predicted_points"] = 0.0
    comparison = compare_ranking_predictions(
        candidate, baseline, metric="mae", bootstrap_samples=100, seed=7
    )
    assert comparison.by_gameweek[["season", "gameweek"]].to_dict("records") == [
        {"season": "2024-25", "gameweek": 1},
        {"season": "2024-25", "gameweek": 2},
    ]

    reports = {
        ("fold-1", 7): ranking_metrics_by_gameweek(candidate),
        ("fold-1", 42): ranking_metrics_by_gameweek(baseline),
    }
    seed_summary = summarize_seed_metrics(reports, metric="mae", higher_is_better=False)
    assert seed_summary.loc[0, "seed_count"] == 2
    assert seed_summary.loc[0, "best_seed"] == 7


def test_probability_metrics_include_calibration_and_handle_single_class_gw() -> None:
    frame = pd.DataFrame(
        {
            "season": ["2024-25"] * 4,
            "gameweek": [1, 1, 2, 2],
            "player_key": ["a", "b", "a", "b"],
            "y_play_any": [0, 1, 1, 1],
            "p_play_any": [0.1, 0.8, 0.7, 0.9],
        }
    )
    report = probability_metrics_by_gameweek(
        frame, target_column="y_play_any", probability_column="p_play_any", bins=5
    )
    assert report.summary["brier"] == pytest.approx(0.0375)
    assert np.isnan(report.by_gameweek.loc[1, "roc_auc"])
    assert report.calibration["count"].sum() == 4
    assert np.isfinite(report.expected_calibration_error)


def test_required_cohorts_mark_outcome_filters_diagnostic_only() -> None:
    frame = _evaluation_frame().iloc[:4]
    cohorts = {cohort.name: cohort for cohort in diagnostic_cohorts(frame)}
    assert cohorts["played"].diagnostic_only
    assert cohorts["minutes_60_plus"].diagnostic_only
    assert not cohorts["all"].diagnostic_only
    assert cohorts["dgw"].mask.sum() == 1
    assert cohorts["early_season"].mask.sum() == 0


def test_bootstrap_resamples_unique_gameweeks_and_is_deterministic() -> None:
    per_gw = pd.DataFrame(
        {
            "season": ["2024-25"] * 3,
            "gameweek": [1, 2, 3],
            "delta": [0.1, 0.2, -0.1],
        }
    )
    first = bootstrap_gameweek_deltas(per_gw, samples=100, seed=7)
    second = bootstrap_gameweek_deltas(per_gw, samples=100, seed=7)
    assert first == second
    duplicated = pd.concat([per_gw, per_gw.iloc[[0]]])
    with pytest.raises(PairingError, match="exactly one row per GW"):
        bootstrap_gameweek_deltas(duplicated)


def test_frozen_plan_prevents_confirmation_selection() -> None:
    plan = load_evaluation_plan()
    plan.assert_selection_allowed(("discovery_1", "discovery_2"))
    with pytest.raises(EvaluationConfigError, match="cannot be used for selection"):
        plan.assert_selection_allowed(("confirmation",))
