"""Acceptance tests for Sprint 7 simple baselines and frozen OOF reporting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

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
from fpl_model.models import (
    BaselineInputError,
    FrozenBaselineReport,
    regenerate_baseline_report,
    run_simple_baselines,
    validate_ep_next_provenance,
)

FEATURES = (
    "position",
    "price",
    "minutes_last_appearance",
    "points_last_appearance",
    "points_mean_last5",
    "play_any_rate_last5",
    "minutes_60_rate_last5",
    "minutes_mean_last5",
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


def _harness() -> WalkForwardHarness:
    plan = EvaluationPlan(
        "baseline-test",
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
    return WalkForwardHarness(ExpandingWindowSplitter(plan))


def _frame() -> pd.DataFrame:
    rows = []
    periods = [("2022-23", 38), ("2023-24", 1), ("2023-24", 2), ("2023-24", 3)]
    start = datetime(2023, 5, 20, 10, tzinfo=UTC)
    positions = ("GKP", "DEF", "MID", "FWD")
    for period_index, (season, gameweek) in enumerate(periods):
        deadline = start + timedelta(days=period_index * 7)
        for player_index, position in enumerate(positions):
            minutes = (0, 30, 70, 90)[player_index]
            points = player_index * 2 + period_index
            has_ep = player_index != 0
            rows.append(
                {
                    "season": season,
                    "gameweek": gameweek,
                    "player_key": f"player-{player_index}",
                    "deadline_utc": deadline,
                    "snapshot_captured_at_utc": deadline - timedelta(hours=2),
                    "eligibility": True,
                    "position": position,
                    "price": 4.5 + player_index,
                    "minutes_last_appearance": np.nan if period_index == 0 else minutes,
                    "points_last_appearance": np.nan if period_index == 0 else points - 1,
                    "points_mean_last5": np.nan if period_index == 0 else points - 0.5,
                    "play_any_rate_last5": np.nan if period_index == 0 else float(minutes > 0),
                    "minutes_60_rate_last5": (
                        np.nan if period_index == 0 else float(minutes >= 60)
                    ),
                    "minutes_mean_last5": np.nan if period_index == 0 else minutes,
                    "ep_next": float(points) + 0.25 if has_ep else np.nan,
                    "ep_next_value_state": "value" if has_ep else "source_unavailable",
                    "y_points": float(points),
                    "y_minutes": float(minutes),
                    "y_play_any": int(minutes > 0),
                    "y_minutes_60": int(minutes >= 60),
                }
            )
    return pd.DataFrame(rows)


def test_all_baselines_share_walk_forward_rows_and_report_regenerates(tmp_path: Path) -> None:
    artifact = run_simple_baselines(_frame(), _harness(), feature_columns=FEATURES)
    output = artifact.frame

    assert len(output) == 8
    assert output["pred_zero_points"].eq(0).all()
    assert output["pred_points_ridge"].ge(0).all()
    assert output["pred_minutes_ridge"].between(0, 180).all()
    assert output["ep_next"].isna().sum() == 2
    probability_columns = [column for column in output if column.startswith("p_")]
    assert output[probability_columns].apply(lambda values: values.between(0, 1).all()).all()

    artifact_path, _ = artifact.write(tmp_path / "simple_baselines.parquet")
    loaded = type(artifact).read(artifact_path)
    report = regenerate_baseline_report(loaded)
    assert isinstance(report, FrozenBaselineReport)
    ep_coverage = next(row for row in report.coverage if row["baseline"] == "ep_next")
    assert ep_coverage["coverage"] == pytest.approx(0.75)
    price = report.ranking_only_metrics["discovery/seed-42"]["rank_price"]
    assert "mae" not in price
    assert "calibration" in report.probability_metrics["discovery/seed-42"]["p_play_any_logistic"]
    report.write_json(tmp_path / "simple_baselines_report.json")


def test_ep_next_rejects_postdeadline_and_never_fills_missing() -> None:
    frame = _frame().iloc[:2].copy()
    values = validate_ep_next_provenance(frame)
    assert np.isnan(values.iloc[0])
    frame.loc[frame.index[1], "snapshot_captured_at_utc"] = frame.loc[
        frame.index[1], "deadline_utc"
    ]
    with pytest.raises(BaselineInputError, match="strictly pre-deadline"):
        validate_ep_next_provenance(frame)
