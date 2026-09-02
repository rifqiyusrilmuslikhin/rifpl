"""Exact-row paired model comparisons and GW-level bootstrap uncertainty."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.evaluation.metrics import GW_COLUMNS, MetricReport, ranking_metrics_by_gameweek

ROW_KEY_COLUMNS = ("season", "gameweek", "player_key")


class PairingError(ValueError):
    """Raised when two candidates were not evaluated on precisely the same observations."""


@dataclass(frozen=True, slots=True)
class PairedMetricComparison:
    metric: str
    by_gameweek: pd.DataFrame
    mean_delta: float
    bootstrap_std: float
    interval_low: float
    interval_high: float
    confidence: float
    bootstrap_samples: int
    seed: int


def summarize_seed_metrics(
    reports: Mapping[tuple[str, int], Any],
    *,
    metric: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    """Report seed noise, worst/best seed, and every test block."""
    records = []
    for (fold, seed), report in reports.items():
        if metric not in report.summary:
            raise PairingError(f"metric {metric!r} is absent from report {(fold, seed)!r}")
        records.append({"fold": fold, "seed": seed, "value": float(report.summary[metric])})
    if not records:
        raise PairingError("seed summary requires at least one report")

    rows = []
    for fold, group in pd.DataFrame(records).groupby("fold", sort=True):
        values = group["value"]
        if not np.isfinite(values).all():
            raise PairingError(f"seed metric {metric!r} must be finite in fold {fold!r}")
        best_index = values.idxmax() if higher_is_better else values.idxmin()
        worst_index = values.idxmin() if higher_is_better else values.idxmax()
        rows.append(
            {
                "fold": fold,
                "metric": metric,
                "seed_count": len(group),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(group) > 1 else 0.0,
                "worst_seed": int(group.loc[worst_index, "seed"]),
                "worst_value": float(group.loc[worst_index, "value"]),
                "best_seed": int(group.loc[best_index, "seed"]),
                "best_value": float(group.loc[best_index, "value"]),
                "max_pairwise_change": float(values.max() - values.min()),
            }
        )
    return pd.DataFrame(rows)


def assert_exact_same_rows(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    key_columns: tuple[str, ...] = ROW_KEY_COLUMNS,
    target_columns: tuple[str, ...] = ("actual_points_gw",),
) -> None:
    """Reject key, target, duplication, or eligibility mismatch before comparison."""
    comparison_columns = (*key_columns, *target_columns)
    _require_columns(candidate, comparison_columns, "candidate")
    _require_columns(baseline, comparison_columns, "baseline")
    for label, frame in (("candidate", candidate), ("baseline", baseline)):
        if frame[list(key_columns)].isna().any(axis=None):
            raise PairingError(f"{label} predictions contain missing row keys")
        if frame[list(key_columns)].duplicated().any():
            raise PairingError(f"{label} predictions contain duplicate row keys")
    candidate_index = pd.MultiIndex.from_frame(candidate[list(key_columns)])
    baseline_index = pd.MultiIndex.from_frame(baseline[list(key_columns)])
    if set(candidate_index) != set(baseline_index) or len(candidate_index) != len(baseline_index):
        missing = sorted(set(baseline_index).difference(candidate_index), key=repr)[:5]
        extra = sorted(set(candidate_index).difference(baseline_index), key=repr)[:5]
        raise PairingError(
            f"paired comparison requires exact same rows; missing={missing!r}, extra={extra!r}"
        )

    left = candidate.set_index(list(key_columns)).sort_index()
    right = baseline.set_index(list(key_columns)).sort_index()
    for column in target_columns:
        if not left[column].equals(right[column]):
            raise PairingError(f"paired rows disagree on target column {column!r}")
    if "eligibility" in candidate.columns or "eligibility" in baseline.columns:
        if "eligibility" not in candidate.columns or "eligibility" not in baseline.columns:
            raise PairingError("both paired frames must retain eligibility when either one does")
        if not left["eligibility"].equals(right["eligibility"]):
            raise PairingError("paired rows disagree on eligibility")


def compare_ranking_predictions(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    metric: str = "ndcg_at_10",
    candidate_prediction: str = "predicted_points",
    baseline_prediction: str = "predicted_points",
    actual_column: str = "actual_points_gw",
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PairedMetricComparison:
    assert_exact_same_rows(candidate, baseline, target_columns=(actual_column,))
    candidate_report = ranking_metrics_by_gameweek(
        candidate,
        actual_column=actual_column,
        prediction_column=candidate_prediction,
    )
    baseline_report = ranking_metrics_by_gameweek(
        baseline,
        actual_column=actual_column,
        prediction_column=baseline_prediction,
    )
    return compare_metric_reports(
        candidate_report,
        baseline_report,
        metric=metric,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )


def compare_metric_reports(
    candidate: MetricReport,
    baseline: MetricReport,
    *,
    metric: str,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PairedMetricComparison:
    if metric not in candidate.by_gameweek or metric not in baseline.by_gameweek:
        raise PairingError(f"metric {metric!r} is absent from a per-GW report")
    left = candidate.by_gameweek.set_index(list(GW_COLUMNS))
    right = baseline.by_gameweek.set_index(list(GW_COLUMNS))
    if not left.index.is_unique or not right.index.is_unique or set(left.index) != set(right.index):
        raise PairingError("metric reports must contain identical unique gameweeks")
    for evidence_column in ("row_count", "row_key_hash"):
        if evidence_column not in left or evidence_column not in right:
            raise PairingError(
                f"paired metric reports must retain exact-row evidence {evidence_column!r}"
            )
        if not left[evidence_column].equals(right[evidence_column]):
            raise PairingError(f"paired gameweeks disagree on {evidence_column}")
    paired = pd.DataFrame(
        {
            "candidate": left.loc[right.index, metric].astype(float),
            "baseline": right[metric].astype(float),
        },
        index=right.index,
    ).reset_index()
    paired["delta"] = paired["candidate"] - paired["baseline"]
    bootstrap = bootstrap_gameweek_deltas(
        paired,
        delta_column="delta",
        samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    return PairedMetricComparison(
        metric=metric,
        by_gameweek=paired,
        mean_delta=bootstrap["mean_delta"],
        bootstrap_std=bootstrap["bootstrap_std"],
        interval_low=bootstrap["interval_low"],
        interval_high=bootstrap["interval_high"],
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def bootstrap_gameweek_deltas(
    by_gameweek: pd.DataFrame,
    *,
    delta_column: str = "delta",
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Resample whole (season, GW) units, never correlated player rows."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    _require_columns(by_gameweek, (*GW_COLUMNS, delta_column), "per-GW delta")
    if by_gameweek[list(GW_COLUMNS)].duplicated().any():
        raise PairingError("bootstrap input must have exactly one row per GW")
    deltas = pd.to_numeric(by_gameweek[delta_column], errors="coerce").to_numpy(dtype=float)
    if not len(deltas) or not np.isfinite(deltas).all():
        raise PairingError("bootstrap deltas must be non-empty finite per-GW values")
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = deltas[sampled_indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "mean_delta": float(deltas.mean()),
        "bootstrap_std": float(means.std(ddof=1)) if samples > 1 else 0.0,
        "interval_low": float(np.quantile(means, alpha)),
        "interval_high": float(np.quantile(means, 1 - alpha)),
    }


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise PairingError(f"{label} frame is missing columns {missing!r}")
