"""Post-prediction diagnostic cohorts for walk-forward evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


class CohortError(ValueError):
    """Raised when required cohort provenance is absent or invalid."""


@dataclass(frozen=True, slots=True)
class Cohort:
    name: str
    mask: pd.Series
    diagnostic_only: bool


def diagnostic_cohorts(
    frame: pd.DataFrame,
    *,
    eligibility_column: str = "eligibility",
    minutes_column: str = "actual_minutes_gw",
    position_column: str | None = None,
    fixture_count_column: str = "fixture_count",
    status_column: str | None = None,
) -> tuple[Cohort, ...]:
    """Return required cohorts; outcome-derived masks are visibly diagnostic-only.

    Every mask starts from pre-deadline eligibility. The played and 60+ masks are created only
    after predictions exist and must never be supplied to a predictor as an eligibility filter.
    """
    position_column = _resolve_column(frame, position_column, ("position_at_deadline", "position"))
    status_column = _resolve_column(frame, status_column, ("status", "status_risk_ordinal"))
    required = {
        eligibility_column,
        minutes_column,
        position_column,
        fixture_count_column,
        status_column,
        "gameweek",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise CohortError(f"cohort frame is missing required columns {missing!r}")
    eligibility = _boolean_series(frame[eligibility_column], eligibility_column)
    minutes = pd.to_numeric(frame[minutes_column], errors="coerce")
    fixtures = pd.to_numeric(frame[fixture_count_column], errors="coerce")
    gameweeks = pd.to_numeric(frame["gameweek"], errors="coerce")
    if minutes.isna().any() or fixtures.isna().any() or gameweeks.isna().any():
        raise CohortError("minutes, fixture_count, and gameweek must be numeric and non-missing")

    cohorts: list[Cohort] = [
        Cohort("all", eligibility, False),
        Cohort("played", eligibility & minutes.gt(0), True),
        Cohort("minutes_60_plus", eligibility & minutes.ge(60), True),
    ]
    positions = frame[position_column].astype("string")
    for position in ("GKP", "DEF", "MID", "FWD"):
        cohorts.append(Cohort(f"position:{position}", eligibility & positions.eq(position), False))
    cohorts.extend(
        (
            Cohort("sgw", eligibility & fixtures.eq(1), False),
            Cohort("dgw", eligibility & fixtures.gt(1), False),
            Cohort("early_season", eligibility & gameweeks.le(5), False),
            Cohort("established_season", eligibility & gameweeks.gt(5), False),
        )
    )
    if status_column == "status_risk_ordinal":
        risk = pd.to_numeric(frame[status_column], errors="coerce")
        available = risk.eq(0)
        flagged = risk.gt(0)
    else:
        status = frame[status_column].astype("string").str.casefold()
        available = status.eq("a")
        flagged = status.ne("a") & status.notna()
    cohorts.extend(
        (
            Cohort("available", eligibility & available, False),
            Cohort("flagged", eligibility & flagged, False),
        )
    )
    return tuple(cohorts)


def cohort_frame(frame: pd.DataFrame, cohort: Cohort) -> pd.DataFrame:
    if not cohort.mask.index.equals(frame.index):
        raise CohortError("cohort mask index does not match prediction frame")
    return frame.loc[cohort.mask].copy()


def evaluate_ranking_cohorts(frame: pd.DataFrame, **metric_kwargs: Any) -> dict[str, Any]:
    """Score every populated required cohort after a common prediction pass."""
    from fpl_model.evaluation.metrics import ranking_metrics_by_gameweek

    reports: dict[str, Any] = {}
    for cohort in diagnostic_cohorts(frame):
        selected = cohort_frame(frame, cohort)
        if not selected.empty:
            reports[cohort.name] = ranking_metrics_by_gameweek(
                selected, cohort=cohort.name, **metric_kwargs
            )
    return reports


def evaluate_probability_cohorts(frame: pd.DataFrame, **metric_kwargs: Any) -> dict[str, Any]:
    """Score probability outputs on every populated required post-prediction cohort."""
    from fpl_model.evaluation.metrics import probability_metrics_by_gameweek

    reports: dict[str, Any] = {}
    for cohort in diagnostic_cohorts(frame):
        selected = cohort_frame(frame, cohort)
        if not selected.empty:
            reports[cohort.name] = probability_metrics_by_gameweek(
                selected, cohort=cohort.name, **metric_kwargs
            )
    return reports


def _boolean_series(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any() or not series.isin((True, False)).all():
        raise CohortError(f"{name!r} must be non-missing boolean")
    return series.astype(bool)


def _resolve_column(frame: pd.DataFrame, requested: str | None, candidates: tuple[str, ...]) -> str:
    if requested is not None:
        return requested
    try:
        return next(column for column in candidates if column in frame.columns)
    except StopIteration as error:
        raise CohortError(f"cohort frame requires one of columns {list(candidates)!r}") from error
