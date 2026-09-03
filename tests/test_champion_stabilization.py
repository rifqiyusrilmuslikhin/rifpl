"""Acceptance tests for discovery-only selection and untouched champion confirmation."""

from __future__ import annotations

import hashlib
import json
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
    CHAMPION_SEEDS,
    ChampionStabilizationError,
    FinalistDeclaration,
    PromotionAuditInputs,
    SelectionFreeze,
    build_stabilization_arms,
    freeze_champion_selection,
    load_champion_stabilization_plan,
    pin_champion_bundle,
    regenerate_champion_discovery_report,
    regenerate_champion_promotion_report,
    run_champion_discovery,
    run_untouched_confirmation,
    write_negative_results_markdown,
)
from fpl_model.models.participation import PREDICTION_COLUMNS

DATA_MANIFEST_HASHES = {"fixture": hashlib.sha256(b"fixture").hexdigest()}


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
        "1.0.0",
        ("season", "gameweek", "player_key"),
        (
            _window("warmup", WindowRole.WARMUP, 1, 1),
            _window("calibration_1", WindowRole.CALIBRATION, 2, 2),
            _window(
                "discovery_1",
                WindowRole.DISCOVERY,
                3,
                4,
                calibration="calibration_1",
            ),
            _window("calibration_2", WindowRole.CALIBRATION, 5, 5),
            _window(
                "discovery_2",
                WindowRole.DISCOVERY,
                6,
                7,
                calibration="calibration_2",
            ),
            _window("confirmation_calibration", WindowRole.CALIBRATION, 8, 8),
            _window(
                "confirmation",
                WindowRole.CONFIRMATION,
                9,
                10,
                calibration="confirmation_calibration",
            ),
        ),
    )
    return WalkForwardHarness(ExpandingWindowSplitter(plan))


def _frame() -> pd.DataFrame:
    policy = load_champion_stabilization_plan()
    rows: list[dict[str, Any]] = []
    start = datetime(2024, 8, 1, 10, tzinfo=UTC)
    positions = ("GKP", "DEF", "MID", "FWD")
    for gameweek in range(1, 11):
        deadline = start + timedelta(days=7 * gameweek)
        for player in range(16):
            played = (player + gameweek) % 4 != 0
            sixty = played and player % 3 != 0
            minutes = 90.0 if sixty else (30.0 if played else 0.0)
            points = float(player % 8 + gameweek % 3)
            row: dict[str, Any] = {
                "season": "2024-25",
                "gameweek": gameweek,
                "player_key": f"player-{player:02d}",
                "deadline_utc": deadline,
                "eligibility": True,
                "y_points": points,
                "y_minutes": minutes,
                "y_play_any": int(played),
                "y_minutes_60": int(sixty),
                "y_points_if_play": points if played else np.nan,
            }
            for index, feature in enumerate(BASELINE_FEATURE_NAMES):
                if feature == "position":
                    row[feature] = positions[player % 4]
                elif feature == "fixture_count":
                    row[feature] = 1
                elif feature in {"play_any_rate_last5", "minutes_60_rate_last5"}:
                    row[feature] = 0.5
                else:
                    row[feature] = player * 0.1 + gameweek * 0.01 + index / 1_000
            row["points_mean_last5"] = 10.0 - points
            if player % 2:
                for feature in policy.understat_features:
                    row[feature] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


class _SyntheticPredictor:
    def __init__(self, arm_id: str, seed: int) -> None:
        self.arm_id = arm_id
        self.seed = seed
        self.fitted_row_keys: list[tuple[object, ...]] = []

    def fit(
        self,
        train_frame: pd.DataFrame,
        calibration_frame: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        target_columns: tuple[str, ...],
    ) -> None:
        del calibration_frame, feature_columns, target_columns
        self.fitted_row_keys = list(
            train_frame[["season", "gameweek", "player_key"]].itertuples(index=False, name=None)
        )

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[["season", "gameweek", "player_key"]].copy()
        player = frame["player_key"].str.removeprefix("player-").astype(int).to_numpy()
        gameweek = frame["gameweek"].to_numpy(dtype=int)
        actual = (player % 8 + gameweek % 3).astype(float)
        selected = self.arm_id == "full/baseline"
        points = actual if selected else 10.0 - actual
        played = (player + gameweek) % 4 != 0
        sixty = played & (player % 3 != 0)
        p_play = np.where(played, 0.9, 0.1)
        p_sixty = np.where(sixty, 0.8, 0.05)
        tiny_offset = CHAMPION_SEEDS.index(self.seed) * 1e-6
        values: dict[str, np.ndarray] = {
            "p_play_any_raw": p_play,
            "p_play_any_calibrated": p_play,
            "p_minutes_60_raw": p_sixty,
            "p_minutes_60_calibrated": p_sixty,
            "p_play_any": p_play,
            "p_minutes_60": p_sixty,
            "p_play_any_historical": np.full(len(frame), 0.5),
            "p_minutes_60_historical": np.full(len(frame), 0.5),
            "conditional_minutes_raw": np.full(len(frame), 75.0),
            "conditional_minutes": np.full(len(frame), 75.0),
            "expected_minutes_unconstrained": p_play * 75.0,
            "expected_minutes": np.maximum(p_play * 75.0, 60.0 * p_sixty),
            "conditional_points_raw": points,
            "conditional_points": points,
            "xpts_direct_raw": points + tiny_offset,
            "xpts_direct": points + tiny_offset,
            "xpts_conditional_raw": points + tiny_offset,
            "xpts_conditional": points + tiny_offset,
            "xpts_blend": points + tiny_offset,
            "blend_direct_weight": np.full(len(frame), 0.5),
            "gw_max_minutes": np.full(len(frame), 90.0),
        }
        for column in PREDICTION_COLUMNS:
            result[column] = values[column]
        return result


def _prior_report(path: Path) -> FinalistDeclaration:
    path.write_text(
        json.dumps(
            {
                "promotion_gate": {"selected_on_discovery": "xpts_blend_ensemble"},
                "decision": "PROMOTE_XPTS_BLEND_ENSEMBLE",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return FinalistDeclaration.from_report(
        path,
        discovery_windows=("discovery_1", "discovery_2"),
    )


def test_plan_predeclares_five_seeds_source_arm_ablations_and_small_grid() -> None:
    plan = load_champion_stabilization_plan()
    arms = build_stabilization_arms(plan)
    assert plan.seeds == CHAMPION_SEEDS
    assert len(plan.fpl_only_features) == 38
    assert {arm.category for arm in arms} == {
        "finalist",
        "source_comparison",
        "grouped_ablation",
        "hyperparameter_grid",
    }
    assert 2 <= len(plan.hyperparameter_grid) <= 6


def test_discovery_selects_without_confirmation_then_freezes_and_runs_one_arm(
    tmp_path: Path,
) -> None:
    finalist = _prior_report(tmp_path / "prior.json")
    harness = _harness()
    study = run_champion_discovery(
        _frame(),
        harness,
        finalist,
        data_manifest_hashes=DATA_MANIFEST_HASHES,
        predictor_factory=lambda arm, seed, fold: _SyntheticPredictor(arm.id, seed),
    )
    report = regenerate_champion_discovery_report(study)

    assert report.selection["selected_arm"] == "full/baseline"
    assert report.selection["confirmation_observed"] is False
    assert set(report.source_comparison["cohorts"]) == {
        "operational_full_coverage",
        "exact_understat_eligible_rows",
    }
    assert report.source_comparison["cohorts"]["operational_full_coverage"]["row_count"] == 64
    assert report.source_comparison["cohorts"]["exact_understat_eligible_rows"]["row_count"] == 32
    assert report.negative_results

    selection = freeze_champion_selection(
        report,
        study,
        evaluation_plan_version=harness.splitter.plan.version,
    )
    loaded_path = selection.write_json(tmp_path / "selection.json")
    loaded = SelectionFreeze.read_json(loaded_path)
    assert loaded.selection_sha256 == selection.selection_sha256
    confirmation_path = tmp_path / "confirmation.parquet"
    confirmation = run_untouched_confirmation(
        _frame(),
        harness,
        selection,
        artifact_path=confirmation_path,
        predictor_factory=lambda arm, seed, fold: _SyntheticPredictor(arm.id, seed),
    )
    assert set(confirmation.frame["fold"]) == {"confirmation"}
    assert set(confirmation.frame["seed"]) == set(CHAMPION_SEEDS)
    with pytest.raises(FileExistsError, match="already exists"):
        run_untouched_confirmation(
            _frame(),
            harness,
            selection,
            artifact_path=confirmation_path,
            predictor_factory=lambda arm, seed, fold: _SyntheticPredictor(arm.id, seed),
        )


def test_tampered_selection_is_rejected(tmp_path: Path) -> None:
    finalist = _prior_report(tmp_path / "prior.json")
    study = run_champion_discovery(
        _frame(),
        _harness(),
        finalist,
        data_manifest_hashes=DATA_MANIFEST_HASHES,
        predictor_factory=lambda arm, seed, fold: _SyntheticPredictor(arm.id, seed),
    )
    report = regenerate_champion_discovery_report(study)
    selection = freeze_champion_selection(
        report,
        study,
        evaluation_plan_version="1.0.0",
    )
    payload = selection.to_dict()
    payload["selected_arm"] = "grid/lower_variance"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ChampionStabilizationError, match="checksum"):
        SelectionFreeze.read_json(path)


def test_promotion_report_and_pinned_bundle_retain_complete_audit(tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.json"
    finalist = _prior_report(prior_path)
    harness = _harness()
    study = run_champion_discovery(
        _frame(),
        harness,
        finalist,
        data_manifest_hashes=DATA_MANIFEST_HASHES,
        predictor_factory=lambda arm, seed, fold: _SyntheticPredictor(arm.id, seed),
    )
    discovery = regenerate_champion_discovery_report(study)
    selection = freeze_champion_selection(
        discovery,
        study,
        evaluation_plan_version=harness.splitter.plan.version,
    )
    confirmation_path = tmp_path / "confirmation.parquet"
    run_untouched_confirmation(
        _frame(),
        harness,
        selection,
        artifact_path=confirmation_path,
        predictor_factory=lambda arm, seed, fold: _SyntheticPredictor(arm.id, seed),
    )
    model_path = tmp_path / "champion.model"
    model_path.write_bytes(b"synthetic reproducible model")
    inputs = PromotionAuditInputs(
        model_artifact_path=model_path,
        confirmation_artifact_path=confirmation_path,
        finalist_report_path=prior_path,
        data_manifest_hashes=DATA_MANIFEST_HASHES,
        git_commit="0" * 40,
    )
    promotion = regenerate_champion_promotion_report(discovery, selection, inputs)

    assert promotion.audit["passed"]
    assert promotion.decision == "PROMOTE"
    assert promotion.promotion_gate["passed"]
    bundle = pin_champion_bundle(promotion, selection, inputs, tmp_path / "champion")
    assert (bundle / "champion.manifest.json").is_file()
    assert (bundle / "environment.lock.json").is_file()
    negative_path = write_negative_results_markdown(promotion, tmp_path / "negative_results.md")
    assert "Decision:" in negative_path.read_text(encoding="utf-8")
