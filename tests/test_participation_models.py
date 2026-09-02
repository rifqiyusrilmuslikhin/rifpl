"""Acceptance tests for chronological participation-aware candidates."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.evaluation import (
    EvaluationPlan,
    EvaluationWindow,
    ExpandingWindowSplitter,
    WalkForwardHarness,
    WindowRole,
)
from fpl_model.features.contract import BASELINE_FEATURE_NAMES
from fpl_model.models import (
    PARTICIPATION_SEEDS,
    ParticipationAwarePredictor,
    regenerate_participation_report,
    run_participation_models,
)


def _window(
    name: str,
    role: WindowRole,
    start: int,
    end: int,
    *,
    calibration: str | None = None,
) -> EvaluationWindow:
    return EvaluationWindow(
        name,
        role,
        "2024-25",
        start,
        "2024-25",
        end,
        selection_allowed=role is WindowRole.DISCOVERY,
        calibration_window=calibration,
    )


def _harness() -> WalkForwardHarness:
    plan = EvaluationPlan(
        "participation-test",
        ("season", "gameweek", "player_key"),
        (
            _window("warmup", WindowRole.WARMUP, 1, 1),
            _window("discovery_calibration", WindowRole.CALIBRATION, 2, 2),
            _window("discovery", WindowRole.DISCOVERY, 3, 4, calibration="discovery_calibration"),
            _window("confirmation_calibration", WindowRole.CALIBRATION, 5, 5),
            _window(
                "confirmation",
                WindowRole.CONFIRMATION,
                6,
                7,
                calibration="confirmation_calibration",
            ),
        ),
    )
    return WalkForwardHarness(ExpandingWindowSplitter(plan))


def _frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start = datetime(2024, 8, 1, 10, tzinfo=UTC)
    positions = ("GKP", "DEF", "MID", "FWD")
    for gameweek in range(1, 8):
        deadline = start + timedelta(days=7 * gameweek)
        for player in range(12):
            fixture_count = 0 if player == 0 else (2 if player == 11 else 1)
            plays = fixture_count > 0 and (player + gameweek) % 4 != 0
            minutes = 0 if not plays else (30 if player % 3 == 0 else 90 * fixture_count)
            points = 0 if not plays else int(player % 6 + gameweek % 3)
            row: dict[str, Any] = {
                "season": "2024-25",
                "gameweek": gameweek,
                "player_key": f"player-{player:02d}",
                "deadline_utc": deadline,
                "eligibility": True,
                "position": positions[player % 4],
                "status_risk_ordinal": float(player % 5 == 0),
                "y_points": float(points),
                "y_minutes": float(minutes),
                "y_play_any": int(minutes > 0),
                "y_minutes_60": int(minutes >= 60),
                "y_points_if_play": float(points) if minutes > 0 else np.nan,
            }
            for feature_index, feature in enumerate(BASELINE_FEATURE_NAMES):
                if feature == "position":
                    row[feature] = positions[player % 4]
                elif feature == "fixture_count":
                    row[feature] = fixture_count
                elif feature == "status_risk_ordinal":
                    row[feature] = float(player % 5 == 0)
                elif feature in {"play_any_rate_last5", "minutes_60_rate_last5"}:
                    row[feature] = np.nan if gameweek == 1 else float(plays)
                else:
                    row[feature] = player * 0.05 + gameweek * 0.02 + feature_index / 1_000
            rows.append(row)
    return pd.DataFrame(rows)


class _RecordingClassifier:
    fitted_rows: list[int] = []

    def __init__(self, **parameters: Any) -> None:
        assert parameters["objective"] == "binary:logistic"

    def fit(self, features: Any, target: np.ndarray, *, verbose: bool) -> _RecordingClassifier:
        assert verbose is False
        self.__class__.fitted_rows.append(features.shape[0])
        self.probability = float(np.mean(target))
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        probability = np.full(features.shape[0], self.probability)
        return np.column_stack((1.0 - probability, probability))


class _RecordingRegressor:
    fitted_rows: list[int] = []

    def __init__(self, **parameters: Any) -> None:
        assert parameters["objective"] == "reg:squarederror"

    def fit(
        self,
        features: Any,
        target: np.ndarray,
        *,
        verbose: bool,
        eval_set: list[tuple[Any, np.ndarray]],
    ) -> _RecordingRegressor:
        assert verbose is False
        assert eval_set[0][0].shape[0] == len(eval_set[0][1])
        self.__class__.fitted_rows.append(features.shape[0])
        self.value = float(np.mean(target))
        return self

    def predict(self, features: Any) -> np.ndarray:
        return np.full(features.shape[0], self.value)


def test_heads_are_fold_local_conditional_and_coherent() -> None:
    frame = _frame()
    train = frame.loc[frame["gameweek"].eq(1)]
    calibration = frame.loc[frame["gameweek"].eq(2)]
    test = frame.loc[frame["gameweek"].eq(3)]
    _RecordingClassifier.fitted_rows.clear()
    _RecordingRegressor.fitted_rows.clear()
    predictor = ParticipationAwarePredictor(
        seed=42,
        classifier_factory=_RecordingClassifier,
        regressor_factory=_RecordingRegressor,
    )
    predictor.fit(
        train,
        calibration,
        feature_columns=BASELINE_FEATURE_NAMES,
        target_columns=("y_points", "y_minutes", "y_play_any", "y_minutes_60", "y_points_if_play"),
    )
    output = predictor.predict(test)

    played_rows = int(train["y_play_any"].sum())
    assert _RecordingClassifier.fitted_rows == [len(train), len(train)]
    assert _RecordingRegressor.fitted_rows == [played_rows, played_rows, len(train)]
    assert predictor.fitted_row_keys == list(
        train[["season", "gameweek", "player_key"]].itertuples(index=False, name=None)
    )
    assert output["p_minutes_60"].le(output["p_play_any"] + 1e-12).all()
    assert output["expected_minutes"].ge(60 * output["p_minutes_60"] - 1e-9).all()
    assert (
        output["expected_minutes"].le(output["gw_max_minutes"] * output["p_play_any"] + 1e-9).all()
    )
    assert output.loc[output["gw_max_minutes"].eq(0), "p_play_any"].eq(0).all()
    assert 0 <= predictor.blend_direct_weight <= 1


def test_walk_forward_ensembles_and_report_are_regenerable() -> None:
    artifact = run_participation_models(_frame(), _harness())
    output = artifact.frame
    assert set(output["seed"]) == set(PARTICIPATION_SEEDS)
    assert output["p_minutes_60_ensemble"].le(output["p_play_any_ensemble"] + 1e-12).all()
    assert output["expected_minutes_ensemble"].ge(60 * output["p_minutes_60_ensemble"] - 1e-9).all()
    identity = ["season", "gameweek", "player_key", "fold"]
    expected = output.groupby(identity)["xpts_blend"].transform("mean")
    assert np.allclose(output["xpts_blend_ensemble"], expected)

    report = regenerate_participation_report(
        artifact,
        window_roles={"discovery": "discovery", "confirmation": "confirmation"},
        bootstrap_samples=100,
    )
    assert report.coherence["final_coherence_passed"]
    assert report.coherence["probability_violations_after"] == 0
    assert set(report.probability_gate["decisions"]) == {"play_any", "minutes_60"}
    assert set(report.xpts_metrics["confirmation"]) == set(
        ("xpts_direct_ensemble", "xpts_conditional_ensemble", "xpts_blend_ensemble")
    )
    assert "position:GKP" in report.cohort_reports["confirmation"]
    assert report.decision.startswith(("KEEP_DIRECT", "PROMOTE_"))
