"""Fail-fast point-in-time and fold-scope leakage checks.

This module contains invariants only.  It deliberately does not create evaluation folds, fit
models, or calculate metrics; those responsibilities belong to the walk-forward harness.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from fpl_model.data.canonical import DGWAnchorError, validate_dgw_anchors

CANONICAL_KEY_COLUMNS = ("season", "gameweek", "player_key")
TARGET_COLUMNS = frozenset(
    {
        "actual_points_gw",
        "actual_minutes_gw",
        "y_play_any",
        "y_minutes_60",
        "y_minutes",
        "y_points",
        "y_points_if_play",
        "y_haul_5",
        "y_haul_10",
    }
)

# These are raw match outcomes when they occur directly in a model matrix.  Historical versions
# must be explicitly aggregated and named as such (for example ``minutes_mean_last5``).
CURRENT_GAMEWEEK_RAW_COLUMNS = frozenset(
    {
        "minutes",
        "total_points",
        "starts",
        "bps",
        "bonus",
        "goals_scored",
        "assists",
        "yellow_cards",
        "red_cards",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "xg",
        "xa",
        "shots",
        "key_passes",
        "lineup",
        "is_starter",
    }
)

_OUTCOME_TOKENS = frozenset(
    {
        "minutes",
        "points",
        "starts",
        "bps",
        "bonus",
        "goals",
        "assists",
        "cards",
        "saves",
        "xg",
        "xa",
        "shots",
        "lineup",
        "starter",
    }
)
_CURRENT_MARKERS = frozenset({"actual", "target", "current", "gw", "event", "live"})
_PROVENANCE_TIMESTAMP_SUFFIXES = (
    "available_at_utc",
    "captured_at_utc",
    "retrieved_at_utc",
    "published_at_utc",
    "feature_cutoff_utc",
)
_REQUIRED_PROVENANCE_TIMESTAMPS = ("feature_cutoff_utc", "snapshot_captured_at_utc")
_SEASON_PATTERN = re.compile(r"^(?P<start>\d{4})-(?P<end>\d{2}|\d{4})$")


class LeakageError(ValueError):
    """Raised when data or fit provenance violates a causal boundary."""


@dataclass(frozen=True, slots=True)
class SuspiciousCorrelation:
    """One raw feature/target pair whose direct correlation crosses the audit threshold."""

    feature: str
    target: str
    correlation: float
    paired_rows: int


def assert_feature_columns_safe(feature_columns: Iterable[str]) -> None:
    """Reject target labels and unaggregated current-GW outcomes from a model feature list."""
    columns = list(feature_columns)
    if len(columns) != len(set(columns)):
        raise LeakageError("model feature list contains duplicate columns")

    forbidden = sorted(column for column in columns if _is_forbidden_feature_name(column))
    if forbidden:
        raise LeakageError(
            f"model feature list contains target/current-GW outcome columns {forbidden!r}"
        )


def assert_feature_timestamps_predeadline(
    frame: pd.DataFrame,
    *,
    timestamp_columns: Iterable[str] | None = None,
    deadline_column: str = "deadline_utc",
) -> None:
    """Reject non-null feature provenance at or after each row's canonical deadline.

    When ``timestamp_columns`` is omitted, provenance-shaped columns are discovered by suffix.
    Event timestamps such as a known future kickoff are intentionally not inferred as availability
    timestamps; callers can include additional source-specific columns explicitly.
    """
    _require_columns(frame, (deadline_column, *_REQUIRED_PROVENANCE_TIMESTAMPS))
    deadlines = _utc_series(frame[deadline_column], deadline_column, allow_missing=False)
    if timestamp_columns is None:
        timestamps = [
            column
            for column in frame.columns
            if column != deadline_column
            and column.casefold().endswith(_PROVENANCE_TIMESTAMP_SUFFIXES)
        ]
    else:
        timestamps = list(dict.fromkeys((*_REQUIRED_PROVENANCE_TIMESTAMPS, *timestamp_columns)))
    _require_columns(frame, timestamps)

    for column in timestamps:
        values = _utc_series(
            frame[column],
            column,
            allow_missing=column not in _REQUIRED_PROVENANCE_TIMESTAMPS,
        )
        leaked = values.notna() & values.ge(deadlines)
        if leaked.any():
            row_labels = frame.index[leaked].tolist()[:5]
            raise LeakageError(
                f"feature timestamp {column!r} must be strictly before {deadline_column!r}; "
                f"violations at rows {row_labels!r}"
            )


def assert_chronological_fold(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    key_columns: Sequence[str] = CANONICAL_KEY_COLUMNS,
    deadline_column: str = "deadline_utc",
) -> None:
    """Reject overlapping, non-chronological, or later-season-contaminated fold rows."""
    if train_frame.empty or test_frame.empty:
        raise LeakageError("chronological fold requires non-empty train and test rows")
    required = (*key_columns, deadline_column)
    _require_columns(train_frame, required)
    _require_columns(test_frame, required)

    train_keys = _frame_keys(train_frame, key_columns, "training")
    test_keys = _frame_keys(test_frame, key_columns, "test")
    overlap = train_keys.intersection(test_keys)
    if overlap:
        raise LeakageError(f"training and test rows overlap at keys {_preview(overlap)!r}")

    train_seasons = {_season_start(value) for value in train_frame["season"]}
    test_seasons = {_season_start(value) for value in test_frame["season"]}
    if max(train_seasons) > min(test_seasons):
        raise LeakageError("training fold contains a season later than the test fold")

    train_deadlines = _utc_series(
        train_frame[deadline_column], deadline_column, allow_missing=False
    )
    test_deadlines = _utc_series(test_frame[deadline_column], deadline_column, allow_missing=False)
    if train_deadlines.max() >= test_deadlines.min():
        raise LeakageError("all training deadlines must be strictly earlier than test deadlines")


def assert_fold_local_fit(
    training_row_keys: Iterable[Hashable],
    fitted_row_keys: Iterable[Hashable],
    *,
    held_out_row_keys: Iterable[Hashable] = (),
    require_all_training_rows: bool = True,
) -> None:
    """Prove that a learned transformation was fitted on training rows only.

    Preprocessors should retain the row keys supplied to ``fit`` and call this guard immediately.
    The exact-set default also catches accidental fitting on only a different training subset;
    callers with an intentional training-only subset may disable that part of the invariant.
    """
    training = _unique_key_set(training_row_keys, "training")
    fitted = _unique_key_set(fitted_row_keys, "fitted")
    held_out = _unique_key_set(held_out_row_keys, "held-out")
    if not training:
        raise LeakageError("fold-local fit requires at least one training row")
    if not fitted:
        raise LeakageError("transformation was not fitted on any rows")

    held_out_used = fitted.intersection(held_out)
    if held_out_used:
        raise LeakageError(f"transformation fit includes held-out rows {_preview(held_out_used)!r}")
    outside_training = fitted.difference(training)
    if outside_training:
        raise LeakageError(
            "transformation fit includes rows outside the training fold "
            f"{_preview(outside_training)!r}"
        )
    if require_all_training_rows:
        missing = training.difference(fitted)
        if missing:
            raise LeakageError(f"transformation fit omitted training rows {_preview(missing)!r}")


def assert_dgw_deadline_anchoring(
    fixture_contexts: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    history_feature_columns: Sequence[str] = (),
) -> None:
    """Reject a DGW whose second fixture can see a different history/deadline state."""
    records = (
        fixture_contexts.to_dict(orient="records")
        if isinstance(fixture_contexts, pd.DataFrame)
        else fixture_contexts
    )
    try:
        validate_dgw_anchors(records, history_fields=history_feature_columns)
    except (DGWAnchorError, KeyError, TypeError) as error:
        raise LeakageError(f"DGW deadline anchoring failed: {error}") from error


def find_suspicious_raw_features(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    target_columns: Iterable[str] = ("actual_minutes_gw", "actual_points_gw"),
    correlation_threshold: float = 0.995,
    minimum_paired_rows: int = 8,
) -> tuple[SuspiciousCorrelation, ...]:
    """Return numeric features with implausibly direct target correlation.

    This heuristic is a tripwire, not feature selection.  A flagged feature must be audited for
    provenance; its predictive usefulness is never a reason to weaken this check.
    """
    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1]")
    if minimum_paired_rows < 3:
        raise ValueError("minimum_paired_rows must be at least 3")
    features = list(feature_columns)
    targets = list(target_columns)
    _require_columns(frame, (*features, *targets))

    findings: list[SuspiciousCorrelation] = []
    for feature in features:
        if not is_numeric_dtype(frame[feature]):
            continue
        for target in targets:
            if not is_numeric_dtype(frame[target]):
                continue
            paired = frame[[feature, target]].dropna()
            if len(paired) < minimum_paired_rows:
                continue
            if paired[feature].nunique() < 2 or paired[target].nunique() < 2:
                continue
            correlation = float(paired[feature].corr(paired[target]))
            if pd.notna(correlation) and abs(correlation) >= correlation_threshold:
                findings.append(SuspiciousCorrelation(feature, target, correlation, len(paired)))
    return tuple(sorted(findings, key=lambda item: (item.feature, item.target, item.correlation)))


def assert_no_suspicious_raw_features(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    target_columns: Iterable[str] = ("actual_minutes_gw", "actual_points_gw"),
    correlation_threshold: float = 0.995,
    minimum_paired_rows: int = 8,
) -> None:
    """Fail if denylisted names or direct target proxies occur in a model matrix."""
    features = tuple(feature_columns)
    assert_feature_columns_safe(features)
    findings = find_suspicious_raw_features(
        frame,
        features,
        target_columns=target_columns,
        correlation_threshold=correlation_threshold,
        minimum_paired_rows=minimum_paired_rows,
    )
    if findings:
        details = [
            f"{item.feature}->{item.target} (r={item.correlation:.6f}, n={item.paired_rows})"
            for item in findings
        ]
        raise LeakageError(f"suspicious direct target correlation detected: {details!r}")


def audit_feature_frame(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    timestamp_columns: Iterable[str] | None = None,
    target_columns: Iterable[str] = ("actual_minutes_gw", "actual_points_gw"),
    correlation_threshold: float = 0.995,
    minimum_paired_rows: int = 8,
) -> None:
    """Run canonical-key, timestamp, denylist, and raw-proxy checks as one release gate."""
    features = tuple(feature_columns)
    _frame_keys(frame, CANONICAL_KEY_COLUMNS, "feature")
    assert_feature_timestamps_predeadline(frame, timestamp_columns=timestamp_columns)
    assert_feature_columns_safe(features)
    targets = tuple(target_columns)
    available_targets = tuple(column for column in targets if column in frame.columns)
    if available_targets:
        assert_no_suspicious_raw_features(
            frame,
            features,
            target_columns=available_targets,
            correlation_threshold=correlation_threshold,
            minimum_paired_rows=minimum_paired_rows,
        )


def _is_forbidden_feature_name(column: Any) -> bool:
    if not isinstance(column, str) or not column.strip():
        raise LeakageError("model feature names must be non-empty strings")
    normalized = re.sub(r"[^a-z0-9]+", "_", column.casefold()).strip("_")
    if normalized in TARGET_COLUMNS or normalized in CURRENT_GAMEWEEK_RAW_COLUMNS:
        return True
    tokens = set(normalized.split("_"))
    return bool(tokens.intersection(_OUTCOME_TOKENS)) and bool(
        tokens.intersection(_CURRENT_MARKERS)
    )


def _utc_series(series: pd.Series, name: str, *, allow_missing: bool) -> pd.Series:
    converted = pd.to_datetime(series, utc=True, errors="coerce")
    invalid = converted.isna() & series.notna()
    if invalid.any():
        raise LeakageError(f"timestamp column {name!r} contains invalid values")
    if not allow_missing and converted.isna().any():
        raise LeakageError(f"timestamp column {name!r} must not be missing")
    return converted


def _frame_keys(
    frame: pd.DataFrame, key_columns: Sequence[str], label: str
) -> set[tuple[Any, ...]]:
    _require_columns(frame, key_columns)
    if frame[list(key_columns)].isna().any(axis=None):
        raise LeakageError(f"{label} canonical keys must not be missing")
    keys = [tuple(row) for row in frame.loc[:, key_columns].itertuples(index=False, name=None)]
    unique = set(keys)
    if len(unique) != len(keys):
        raise LeakageError(f"{label} frame contains duplicate canonical keys")
    return unique


def _unique_key_set(values: Iterable[Hashable], label: str) -> set[Hashable]:
    items = list(values)
    try:
        unique = set(items)
    except TypeError as error:
        raise LeakageError(f"{label} row keys must be hashable") from error
    if len(unique) != len(items):
        raise LeakageError(f"{label} row keys contain duplicates")
    return unique


def _season_start(value: Any) -> int:
    if not isinstance(value, str):
        raise LeakageError(f"season must be a string, got {value!r}")
    match = _SEASON_PATTERN.fullmatch(value.strip())
    if match is None:
        raise LeakageError(f"season must use YYYY-YY or YYYY-YYYY format, got {value!r}")
    return int(match.group("start"))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise LeakageError(f"frame is missing required columns {missing!r}")


def _preview(values: set[Any]) -> list[Any]:
    return sorted(values, key=repr)[:5]
