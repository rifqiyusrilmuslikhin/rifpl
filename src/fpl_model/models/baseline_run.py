"""Orchestration boundary for running all simple baselines through walk-forward."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from fpl_model.evaluation.artifacts import RetainedPredictions
from fpl_model.evaluation.harness import WalkForwardHarness
from fpl_model.features.contract import BASELINE_FEATURE_NAMES
from fpl_model.models.baselines import (
    PREDICTION_COLUMNS,
    SimpleBaselinePredictor,
    validate_ep_next_provenance,
)

DEFAULT_BASELINE_FEATURES = (*BASELINE_FEATURE_NAMES, "points_last_appearance")
DEFAULT_TARGETS = ("y_points", "y_minutes", "y_play_any", "y_minutes_60")


def run_simple_baselines(
    frame: pd.DataFrame,
    harness: WalkForwardHarness,
    *,
    feature_columns: Sequence[str] = DEFAULT_BASELINE_FEATURES,
    target_columns: Sequence[str] = DEFAULT_TARGETS,
    context_columns: Sequence[str] = (),
    eligibility_column: str = "eligibility",
    window_names: Sequence[str] | None = None,
    ridge_alpha: float = 10.0,
    seed: int = 42,
) -> RetainedPredictions:
    """Validate official xPts, fit fold-local baselines, and retain every OOF output."""
    prepared = frame.copy()
    prepared["ep_next"] = validate_ep_next_provenance(prepared)

    def factory(factory_seed: int, fold: object) -> SimpleBaselinePredictor:
        del fold
        return SimpleBaselinePredictor(seed=factory_seed, ridge_alpha=ridge_alpha)

    return harness.run(
        prepared,
        factory,
        feature_columns=tuple(feature_columns),
        target_columns=tuple(target_columns),
        prediction_columns=PREDICTION_COLUMNS,
        seeds=(seed,),
        baseline_columns=("ep_next",),
        eligibility_column=eligibility_column,
        context_columns=tuple(context_columns),
        window_names=window_names,
    )
