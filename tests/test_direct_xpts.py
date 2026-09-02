"""Acceptance tests for the fixed Sprint 8 direct-xPts experiment."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fpl_model.evaluation import (
    EvaluationPlan,
    EvaluationWindow,
    ExpandingWindowSplitter,
    WalkForwardHarness,
    WindowRole,
)
from fpl_model.features.contract import BASELINE_FEATURE_NAMES
from fpl_model.models import (
    BASE_ARM,
    DIRECT_XPTS_SEEDS,
    DirectXptsPredictor,
    FrozenDirectXptsReport,
    regenerate_direct_xpts_report,
    run_direct_xpts,
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
        "direct-xpts-test",
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
        for player in range(24):
            row: dict[str, Any] = {
                "season": "2024-25",
                "gameweek": gameweek,
                "player_key": f"player-{player:02d}",
                "deadline_utc": deadline,
                "snapshot_captured_at_utc": deadline - timedelta(hours=2),
                "eligibility": True,
            }
            for feature_index, feature in enumerate(BASELINE_FEATURE_NAMES):
                if feature == "position":
                    row[feature] = positions[player % len(positions)]
                else:
                    value = player * (0.08 + feature_index / 2_000) + gameweek * 0.04
                    row[feature] = np.nan if (player + feature_index) % 29 == 0 else value
            row["price"] = 4.0 + player * 0.2
            row["points_mean_last5"] = player * 0.11 + (gameweek - 1) * 0.08
            target = player * 0.15 + np.sin(player / 3) + gameweek * 0.1
            row["y_points"] = float(target)
            has_ep = player % 5 != 0
            row["ep_next"] = target + 0.2 if has_ep else np.nan
            row["ep_next_value_state"] = "value" if has_ep else "source_unavailable"
            rows.append(row)
    return pd.DataFrame(rows)


class _RecordingRegressor:
    instances: list[_RecordingRegressor] = []

    def __init__(self, **parameters: Any) -> None:
        self.parameters = parameters
        self.best_iteration = 3
        self.best_score = 0.25
        self.train_rows = 0
        self.validation_rows = 0
        self.__class__.instances.append(self)

    def fit(
        self,
        features: Any,
        target: np.ndarray,
        *,
        eval_set: list[tuple[Any, np.ndarray]],
        verbose: bool,
    ) -> _RecordingRegressor:
        assert verbose is False
        self.train_rows = features.shape[0]
        self.validation_rows = eval_set[0][0].shape[0]
        assert len(target) == self.train_rows
        assert len(eval_set[0][1]) == self.validation_rows
        return self

    def predict(self, features: Any) -> np.ndarray:
        return np.full(features.shape[0], -0.5)


def test_preprocessing_and_early_stopping_are_fold_local() -> None:
    frame = _frame()
    train = frame.loc[frame["gameweek"].eq(1)].copy()
    calibration = frame.loc[frame["gameweek"].eq(2)].copy()
    _RecordingRegressor.instances.clear()
    predictor = DirectXptsPredictor(
        seed=42,
        include_ep_next_arm=False,
        model_factory=_RecordingRegressor,
    )
    predictor.fit(
        train,
        calibration,
        feature_columns=BASELINE_FEATURE_NAMES,
        target_columns=("y_points",),
    )

    model = _RecordingRegressor.instances[0]
    assert model.train_rows == len(train)
    assert model.validation_rows == len(calibration)
    assert model.parameters["objective"] == "reg:squarederror"
    assert model.parameters["eval_metric"] == "mae"
    assert model.parameters["tree_method"] == "hist"
    assert predictor.fitted_row_keys == list(
        train[["season", "gameweek", "player_key"]].itertuples(index=False, name=None)
    )
    numeric_pipeline = predictor.preprocessors[BASE_ARM].named_transformers_["numeric"]
    numeric_columns = predictor.preprocessors[BASE_ARM].transformers_[0][2]
    price_index = numeric_columns.index("price")
    expected_train_median = pd.to_numeric(train["price"]).median()
    assert numeric_pipeline.named_steps["imputer"].statistics_[price_index] == pytest.approx(
        expected_train_median
    )

    predicted = predictor.predict(calibration)
    assert predicted["xpts_direct_raw"].eq(-0.5).all()
    assert predicted["xpts_direct"].eq(0.0).all()
    assert predicted["xpts_direct_best_iteration"].eq(3.0).all()


def test_fixed_seed_xgboost_predictions_are_reproducible() -> None:
    frame = _frame()
    train = frame.loc[frame["gameweek"].eq(1)].copy()
    calibration = frame.loc[frame["gameweek"].eq(2)].copy()
    test = frame.loc[frame["gameweek"].eq(3)].copy()
    predictions = []
    iterations = []
    for _ in range(2):
        predictor = DirectXptsPredictor(seed=42, include_ep_next_arm=False)
        predictor.fit(
            train,
            calibration,
            feature_columns=BASELINE_FEATURE_NAMES,
            target_columns=("y_points",),
        )
        predictions.append(predictor.predict(test)["xpts_direct_raw"].to_numpy())
        iterations.append(predictor.models[BASE_ARM].best_iteration)

    assert np.array_equal(predictions[0], predictions[1])
    assert iterations[0] == iterations[1]


def test_three_seed_oof_ensemble_optional_arm_and_report(tmp_path: Path) -> None:
    artifact = run_direct_xpts(_frame(), _harness())
    output = artifact.frame

    assert set(output["seed"]) == set(DIRECT_XPTS_SEEDS)
    assert "xpts_direct_with_ep_next" in output
    assert output["xpts_direct"].ge(0).all()
    assert output["xpts_direct_with_ep_next"].ge(0).all()
    assert output["xpts_direct_best_iteration"].between(0, 599).all()
    identity = ["season", "gameweek", "player_key", "fold"]
    expected_raw = output.groupby(identity)["xpts_direct_raw"].transform("mean")
    assert np.allclose(output["xpts_direct_ensemble_raw"], expected_raw)
    assert np.allclose(output["xpts_direct_ensemble"], np.maximum(expected_raw, 0.0))
    assert output.groupby(identity)["xpts_direct_ensemble"].nunique().eq(1).all()

    artifact_path, _ = artifact.write(
        tmp_path / "direct_xpts.parquet", metadata={"experiment_id": "test"}
    )
    loaded = type(artifact).read(artifact_path)
    report = regenerate_direct_xpts_report(
        loaded,
        data_manifest_hash="fixture-hash",
        window_roles={"discovery": "discovery", "confirmation": "confirmation"},
        bootstrap_samples=100,
    )
    assert isinstance(report, FrozenDirectXptsReport)
    assert report.experiment["seeds"] == list(DIRECT_XPTS_SEEDS)
    assert report.experiment["optional_arm"] == "ep_next"
    assert {row["metric"] for row in report.seed_noise} >= {
        "ndcg_at_10",
        "spearman",
        "mae",
        "mean_bias",
    }
    comparison = report.paired_comparisons["discovery"]["xpts_direct_ensemble"]
    assert set(comparison) == {"pred_last5_points", "pred_points_ridge", "rank_price", "ep_next"}
    assert len(comparison["pred_last5_points"]["ndcg_at_10"]["by_gameweek"]) == 2
    assert report.promotion_gate["confirmation_result"] in {"passed", "failed"}
    assert report.decision in {"KEEP", "REJECT"}
    report.write_json(tmp_path / "direct_xpts_report.json")
