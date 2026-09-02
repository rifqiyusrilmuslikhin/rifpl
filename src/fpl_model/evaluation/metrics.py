"""GW-first ranking, error, and probability metrics.

Every primary aggregate in this module is an arithmetic mean of per-GW values. The API does not
offer a pooled-season primary metric, making the intended evaluation unit difficult to bypass.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

GW_COLUMNS = ("season", "gameweek")
RANKING_METRICS = (
    "ndcg_at_10",
    "spearman",
    "mae",
    "rmse",
    "mean_bias",
    "top_10_overlap",
    "top_1_in_actual_top_10",
)
PROBABILITY_METRICS = (
    "brier",
    "log_loss",
    "roc_auc",
    "pr_auc",
    "precision_at_5",
    "precision_at_10",
    "precision_at_20",
    "lift_at_5",
    "lift_at_10",
    "lift_at_20",
    "recall_at_10",
    "recall_at_20",
)


class MetricInputError(ValueError):
    """Raised when predictions cannot be scored without ambiguity."""


@dataclass(frozen=True, slots=True)
class MetricReport:
    """Regenerable metrics with explicit GW-level evidence."""

    by_gameweek: pd.DataFrame
    summary: dict[str, float]
    cohort: str = "all"
    aggregation: str = "mean_of_gameweeks"

    def __post_init__(self) -> None:
        if self.aggregation != "mean_of_gameweeks":
            raise MetricInputError("primary metrics must be aggregated as mean_of_gameweeks")


@dataclass(frozen=True, slots=True)
class ProbabilityReport:
    by_gameweek: pd.DataFrame
    summary: dict[str, float]
    calibration: pd.DataFrame
    calibration_intercept: float
    calibration_slope: float
    expected_calibration_error: float
    cohort: str = "all"
    aggregation: str = "mean_of_gameweeks"

    def __post_init__(self) -> None:
        if self.aggregation != "mean_of_gameweeks":
            raise MetricInputError("primary metrics must be aggregated as mean_of_gameweeks")


def ranking_metrics_by_gameweek(
    frame: pd.DataFrame,
    *,
    actual_column: str = "actual_points_gw",
    prediction_column: str = "predicted_points",
    key_column: str = "player_key",
    cohort: str = "all",
) -> MetricReport:
    """Compute ranking and error values independently within each canonical GW."""
    _require_columns(frame, (*GW_COLUMNS, key_column, actual_column, prediction_column))
    _validate_scoring_rows(frame, key_column, (actual_column, prediction_column))
    rows: list[dict[str, Any]] = []
    for (season, gameweek), group in frame.groupby(list(GW_COLUMNS), sort=True, observed=True):
        actual = group[actual_column].to_numpy(dtype=float)
        predicted = group[prediction_column].to_numpy(dtype=float)
        keys = group[key_column].astype(str).to_numpy()
        cutoff = min(10, len(group))
        predicted_order = _descending_order(predicted, keys)
        actual_order = _descending_order(actual, keys)
        predicted_top = set(predicted_order[:cutoff])
        actual_top = set(actual_order[:cutoff])
        error = predicted - actual
        rows.append(
            {
                "season": season,
                "gameweek": int(gameweek),
                "row_count": len(group),
                "row_key_hash": _row_key_hash(keys),
                "ndcg_at_10": _ndcg_at_k(actual, predicted_order, actual_order, cutoff),
                "spearman": _spearman(actual, predicted),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mean_bias": float(np.mean(error)),
                "top_10_overlap": len(predicted_top.intersection(actual_top)) / cutoff,
                "top_1_in_actual_top_10": float(predicted_order[0] in actual_top),
            }
        )
    by_gameweek = pd.DataFrame(rows)
    summary = _mean_summary(by_gameweek, RANKING_METRICS)
    summary["gameweek_count"] = float(len(by_gameweek))
    summary["row_count"] = float(by_gameweek["row_count"].sum())
    return MetricReport(by_gameweek, summary, cohort=cohort)


def probability_metrics_by_gameweek(
    frame: pd.DataFrame,
    *,
    target_column: str,
    probability_column: str,
    key_column: str = "player_key",
    bins: int = 10,
    cohort: str = "all",
) -> ProbabilityReport:
    """Score probabilities GW-first and retain pooled reliability-bin data as a diagnostic."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    _require_columns(frame, (*GW_COLUMNS, key_column, target_column, probability_column))
    _validate_scoring_rows(frame, key_column, (target_column, probability_column))
    targets = frame[target_column].to_numpy(dtype=float)
    probabilities = frame[probability_column].to_numpy(dtype=float)
    if not np.isin(targets, (0.0, 1.0)).all():
        raise MetricInputError(f"binary target {target_column!r} must contain only 0 and 1")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise MetricInputError(f"probability {probability_column!r} must be in [0, 1]")

    rows: list[dict[str, Any]] = []
    for (season, gameweek), group in frame.groupby(list(GW_COLUMNS), sort=True, observed=True):
        y = group[target_column].to_numpy(dtype=int)
        probability = group[probability_column].to_numpy(dtype=float)
        keys = group[key_column].astype(str).to_numpy()
        row = {
            "season": season,
            "gameweek": int(gameweek),
            "row_count": len(group),
            "row_key_hash": _row_key_hash(keys),
            "brier": float(np.mean(np.square(probability - y))),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "roc_auc": _binary_metric_or_nan(roc_auc_score, y, probability),
            "pr_auc": _binary_metric_or_nan(average_precision_score, y, probability),
        }
        row.update(_top_probability_metrics(y, probability, keys))
        rows.append(row)
    by_gameweek = pd.DataFrame(rows)
    summary = _mean_summary(by_gameweek, PROBABILITY_METRICS)
    summary["gameweek_count"] = float(len(by_gameweek))
    summary["row_count"] = float(by_gameweek["row_count"].sum())
    calibration = calibration_table(targets, probabilities, bins=bins)
    intercept, slope = calibration_intercept_slope(targets, probabilities)
    ece = float(
        (
            calibration["count"]
            * (calibration["observed_rate"] - calibration["mean_probability"]).abs()
        ).sum()
        / calibration["count"].sum()
    )
    return ProbabilityReport(
        by_gameweek=by_gameweek,
        summary=summary,
        calibration=calibration,
        calibration_intercept=intercept,
        calibration_slope=slope,
        expected_calibration_error=ece,
        cohort=cohort,
    )


def calibration_table(
    targets: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    y, probability = _validate_probability_vectors(targets, probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    assignments = np.maximum(assignments, 0)
    records = []
    for bin_index in range(bins):
        selected = assignments == bin_index
        if not selected.any():
            continue
        records.append(
            {
                "bin": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "count": int(selected.sum()),
                "mean_probability": float(probability[selected].mean()),
                "observed_rate": float(y[selected].mean()),
            }
        )
    return pd.DataFrame(records)


def calibration_intercept_slope(
    targets: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Fit the conventional logistic calibration intercept and slope."""
    from sklearn.linear_model import LogisticRegression

    y_values, probability = _validate_probability_vectors(targets, probabilities)
    y = y_values.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    epsilon = np.finfo(float).eps
    clipped = np.clip(probability, epsilon, 1 - epsilon)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs").fit(logits, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def monotonicity_violation_rate(
    lower_threshold_probability: np.ndarray | pd.Series,
    higher_threshold_probability: np.ndarray | pd.Series,
    *,
    tolerance: float = 1e-12,
) -> float:
    """Return the share where P(higher threshold) incorrectly exceeds P(lower threshold)."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    lower = np.asarray(lower_threshold_probability, dtype=float)
    higher = np.asarray(higher_threshold_probability, dtype=float)
    if lower.shape != higher.shape or lower.size == 0:
        raise MetricInputError("paired probability vectors must have the same non-empty shape")
    if not np.isfinite(lower).all() or not np.isfinite(higher).all():
        raise MetricInputError("paired probability vectors must be finite")
    if ((lower < 0) | (lower > 1) | (higher < 0) | (higher > 1)).any():
        raise MetricInputError("paired probability vectors must be in [0, 1]")
    return float(np.mean(higher > lower + tolerance))


def _ndcg_at_k(
    actual: np.ndarray,
    predicted_order: np.ndarray,
    actual_order: np.ndarray,
    cutoff: int,
) -> float:
    # Negative FPL scores are valid outcomes but not valid relevance. They carry zero gain.
    relevance = np.maximum(actual, 0.0)
    discounts = np.log2(np.arange(2, cutoff + 2))
    dcg = float(np.sum(relevance[predicted_order[:cutoff]] / discounts))
    ideal = float(np.sum(relevance[actual_order[:cutoff]] / discounts))
    return dcg / ideal if ideal > 0 else 0.0


def _descending_order(values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    # lexsort's final key is primary; canonical player key makes ties reproducible.
    return np.lexsort((keys, -values))


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2 or np.unique(actual).size < 2 or np.unique(predicted).size < 2:
        return float("nan")
    actual_ranks = pd.Series(actual).rank(method="average")
    predicted_ranks = pd.Series(predicted).rank(method="average")
    return float(actual_ranks.corr(predicted_ranks))


def _binary_metric_or_nan(metric: Any, target: np.ndarray, probability: np.ndarray) -> float:
    if np.unique(target).size < 2:
        return float("nan")
    return float(metric(target, probability))


def _top_probability_metrics(
    target: np.ndarray, probability: np.ndarray, keys: np.ndarray
) -> dict[str, float]:
    order = _descending_order(probability, keys)
    positives = int(target.sum())
    prevalence = float(target.mean())
    output: dict[str, float] = {}
    for cutoff in (5, 10, 20):
        selected = order[: min(cutoff, len(order))]
        precision = float(target[selected].mean())
        output[f"precision_at_{cutoff}"] = precision
        output[f"lift_at_{cutoff}"] = precision / prevalence if prevalence > 0 else float("nan")
        if cutoff in (10, 20):
            output[f"recall_at_{cutoff}"] = (
                float(target[selected].sum() / positives) if positives else float("nan")
            )
    return output


def _mean_summary(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, float]:
    return {column: float(frame[column].mean(skipna=True)) for column in columns}


def _row_key_hash(keys: np.ndarray) -> str:
    payload = "\n".join(sorted(str(key) for key in keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_probability_vectors(
    targets: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(targets, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or probability.ndim != 1 or y.shape != probability.shape or y.size == 0:
        raise MetricInputError("targets and probabilities must be same-length non-empty vectors")
    if not np.isfinite(y).all() or not np.isfinite(probability).all():
        raise MetricInputError("targets and probabilities must be finite")
    if not np.isin(y, (0.0, 1.0)).all():
        raise MetricInputError("probability targets must contain only 0 and 1")
    if ((probability < 0) | (probability > 1)).any():
        raise MetricInputError("probabilities must be in [0, 1]")
    return y, probability


def _validate_scoring_rows(
    frame: pd.DataFrame, key_column: str, numeric_columns: tuple[str, ...]
) -> None:
    if frame.empty:
        raise MetricInputError("cannot calculate metrics for an empty frame")
    if frame[[*GW_COLUMNS, key_column]].isna().any(axis=None):
        raise MetricInputError("metric row keys must not be missing")
    if frame[[*GW_COLUMNS, key_column]].duplicated().any():
        raise MetricInputError(
            "duplicate player-GW keys detected; aggregate DGW fixtures to one canonical row"
        )
    values = frame[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any(axis=None) or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise MetricInputError(f"metric columns must contain finite numbers: {numeric_columns!r}")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise MetricInputError(f"metric frame is missing required columns {missing!r}")
