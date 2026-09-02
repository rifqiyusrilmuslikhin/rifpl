"""Regenerate and freeze Sprint 7 reports exclusively from retained OOF predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.evaluation.artifacts import RetainedPredictions
from fpl_model.evaluation.metrics import (
    ProbabilityReport,
    probability_metrics_by_gameweek,
    ranking_metrics_by_gameweek,
)

POINT_BASELINES = (
    "pred_zero_points",
    "pred_last_appearance_points",
    "pred_last5_points",
    "pred_position_minutes_points",
    "pred_points_ridge",
)
PROBABILITY_BASELINES = {
    "p_play_any_base_rate": "y_play_any",
    "p_play_any_historical": "y_play_any",
    "p_play_any_logistic": "y_play_any",
    "p_minutes_60_base_rate": "y_minutes_60",
    "p_minutes_60_historical": "y_minutes_60",
    "p_minutes_60_logistic": "y_minutes_60",
}
RANKING_ONLY_METRICS = (
    "ndcg_at_10",
    "spearman",
    "top_10_overlap",
    "top_1_in_actual_top_10",
    "gameweek_count",
    "row_count",
)


class BaselineReportError(ValueError):
    """Raised when retained predictions cannot regenerate an honest baseline report."""


@dataclass(frozen=True, slots=True)
class FrozenBaselineReport:
    schema_version: str
    point_metrics: dict[str, dict[str, Any]]
    ranking_only_metrics: dict[str, dict[str, dict[str, float]]]
    probability_metrics: dict[str, dict[str, dict[str, Any]]]
    coverage: list[dict[str, Any]]
    sanity_checks: dict[str, Any]
    recommendation: str
    recommendation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def write_json(self, path: str | Path) -> Path:
        """Freeze a report without silently replacing a previously reviewed result."""
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"frozen baseline report already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination.resolve()


def regenerate_baseline_report(
    artifact: RetainedPredictions,
    *,
    points_target: str = "y_points",
    minutes_target: str = "y_minutes",
    bins: int = 10,
) -> FrozenBaselineReport:
    """Build all metrics and calibration evidence without fitting any model."""
    frame = artifact.frame
    required = {
        points_target,
        minutes_target,
        "ep_next",
        "rank_price",
        "pred_minutes_historical",
        "pred_minutes_ridge",
        "has_last_appearance_history",
        "has_last5_points_history",
        "has_play_rate_history",
        "has_minutes_60_rate_history",
        "has_minutes_mean_history",
        *POINT_BASELINES,
        *PROBABILITY_BASELINES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BaselineReportError(f"retained baseline artifact is missing columns {missing!r}")
    scored = frame.loc[frame["eligibility"]].copy()
    if scored.empty:
        raise BaselineReportError("baseline report has no eligible OOF rows")

    point_metrics: dict[str, dict[str, Any]] = {}
    ranking_only: dict[str, dict[str, dict[str, float]]] = {}
    probability: dict[str, dict[str, dict[str, Any]]] = {}
    coverage: list[dict[str, Any]] = []

    for (fold, seed), group in scored.groupby(["fold", "seed"], sort=True):
        label = _fold_seed_label(str(fold), int(seed))
        point_metrics[label] = {
            name: ranking_metrics_by_gameweek(
                group, actual_column=points_target, prediction_column=name
            ).summary
            for name in POINT_BASELINES
        }
        price_report = ranking_metrics_by_gameweek(
            group, actual_column=points_target, prediction_column="rank_price"
        )
        ranking_only[label] = {
            "rank_price": {metric: price_report.summary[metric] for metric in RANKING_ONLY_METRICS}
        }

        ep_available = group["ep_next"].notna()
        coverage.extend(
            _coverage_rows(
                group,
                ep_available,
                fold=str(fold),
                seed=int(seed),
                baseline="ep_next",
            )
        )
        history_availability = {
            "last_appearance": "has_last_appearance_history",
            "last5_points": "has_last5_points_history",
            "historical_play_rate": "has_play_rate_history",
            "historical_60_rate": "has_minutes_60_rate_history",
            "historical_mean_minutes": "has_minutes_mean_history",
        }
        for baseline, indicator in history_availability.items():
            coverage.extend(
                _coverage_rows(
                    group,
                    group[indicator].eq(1.0),
                    fold=str(fold),
                    seed=int(seed),
                    baseline=baseline,
                )
            )
        if ep_available.any():
            ep_group = group.loc[ep_available]
            ep_report = ranking_metrics_by_gameweek(
                ep_group, actual_column=points_target, prediction_column="ep_next"
            )
            ranking_only[label]["ep_next"] = ep_report.summary
            # Exact-row comparator makes the reduced official-xPts coverage explicit and paired.
            ranking_only[label]["pred_last5_points_on_ep_next_rows"] = ranking_metrics_by_gameweek(
                ep_group,
                actual_column=points_target,
                prediction_column="pred_last5_points",
            ).summary

        minutes = {}
        for name in ("pred_minutes_historical", "pred_minutes_ridge"):
            report = ranking_metrics_by_gameweek(
                group, actual_column=minutes_target, prediction_column=name
            )
            minutes[name] = {
                metric: report.summary[metric]
                for metric in ("mae", "rmse", "mean_bias", "gameweek_count", "row_count")
            }
        point_metrics[label]["minutes"] = minutes

        probability[label] = {}
        for probability_column, target_column in PROBABILITY_BASELINES.items():
            report = probability_metrics_by_gameweek(
                group,
                target_column=target_column,
                probability_column=probability_column,
                bins=bins,
            )
            probability[label][probability_column] = _probability_payload(report)

    sanity = _sanity_checks(scored, points_target)
    recommendation, reason = _recommendation(point_metrics, sanity)
    return FrozenBaselineReport(
        schema_version="1.0.0",
        point_metrics=point_metrics,
        ranking_only_metrics=ranking_only,
        probability_metrics=probability,
        coverage=coverage,
        sanity_checks=sanity,
        recommendation=recommendation,
        recommendation_reason=reason,
    )


def _coverage_rows(
    frame: pd.DataFrame,
    available: pd.Series,
    *,
    fold: str,
    seed: int,
    baseline: str,
) -> list[dict[str, Any]]:
    rows = []
    for season, indices in frame.groupby("season", sort=True).groups.items():
        season_available = available.loc[indices]
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "season": str(season),
                "baseline": baseline,
                "row_count": int(len(indices)),
                "available_count": int(season_available.sum()),
                "missing_count": int((~season_available).sum()),
                "coverage": float(season_available.mean()),
            }
        )
    return rows


def _probability_payload(report: ProbabilityReport) -> dict[str, Any]:
    return {
        "summary": report.summary,
        "calibration_intercept": report.calibration_intercept,
        "calibration_slope": report.calibration_slope,
        "expected_calibration_error": report.expected_calibration_error,
        "calibration": report.calibration.to_dict(orient="records"),
    }


def _sanity_checks(frame: pd.DataFrame, points_target: str) -> dict[str, Any]:
    actual = pd.to_numeric(frame[points_target], errors="coerce").to_numpy(dtype=float)
    zero = pd.to_numeric(frame["pred_zero_points"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(actual).all() and np.isfinite(zero).all()
    exactly_zero = bool(np.equal(zero, 0.0).all())
    expected_mae = float(np.mean(np.abs(actual)))
    observed_mae = float(np.mean(np.abs(zero - actual)))
    return {
        "always_zero_exactly_zero": exactly_zero,
        "always_zero_mae_matches_mean_absolute_target": bool(
            finite and np.isclose(expected_mae, observed_mae)
        ),
        "row_identity_unique": bool(
            ~frame[["season", "gameweek", "player_key", "fold", "seed"]].duplicated().any()
        ),
        "ep_next_missing_count": int(frame["ep_next"].isna().sum()),
    }


def _recommendation(
    point_metrics: dict[str, dict[str, Any]], sanity: dict[str, Any]
) -> tuple[str, str]:
    integrity_checks = (
        sanity["always_zero_exactly_zero"],
        sanity["always_zero_mae_matches_mean_absolute_target"],
        sanity["row_identity_unique"],
    )
    if not all(bool(value) for value in integrity_checks):
        return "STOP", "Always-zero or retained-row integrity checks failed; diagnose data first."
    reasonable = []
    for reports in point_metrics.values():
        zero = reports["pred_zero_points"]
        last5 = reports["pred_last5_points"]
        reasonable.append(last5["ndcg_at_10"] >= zero["ndcg_at_10"] or last5["mae"] <= zero["mae"])
    if reasonable and all(reasonable):
        return "GO", "Last-5 is plausible relative to always-zero in every evaluated fold."
    return "STOP", "Last-5 underperforms always-zero on both ranking and MAE in at least one fold."


def _fold_seed_label(fold: str, seed: int) -> str:
    return f"{fold}/seed-{seed}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
