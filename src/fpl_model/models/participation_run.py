"""Walk-forward orchestration for participation-aware candidates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fpl_model.evaluation.artifacts import RetainedPredictions, retained_predictions
from fpl_model.evaluation.harness import WalkForwardHarness
from fpl_model.features.contract import BASELINE_FEATURE_NAMES
from fpl_model.models.participation import (
    ENSEMBLE_COLUMNS,
    ENSEMBLE_SOURCE_COLUMNS,
    PARTICIPATION_SEEDS,
    PARTICIPATION_TARGETS,
    PREDICTION_COLUMNS,
    ParticipationAwarePredictor,
    ParticipationConfig,
    ParticipationInputError,
)

DEFAULT_PARTICIPATION_FEATURES = BASELINE_FEATURE_NAMES
DEFAULT_PARTICIPATION_TARGETS = PARTICIPATION_TARGETS
DEFAULT_COHORT_CONTEXT = ("position", "fixture_count", "status_risk_ordinal")


def run_participation_models(
    frame: pd.DataFrame,
    harness: WalkForwardHarness,
    *,
    feature_columns: Sequence[str] = DEFAULT_PARTICIPATION_FEATURES,
    target_columns: Sequence[str] = DEFAULT_PARTICIPATION_TARGETS,
    context_columns: Sequence[str] = (),
    eligibility_column: str = "eligibility",
    window_names: Sequence[str] | None = None,
    seeds: Sequence[int] = PARTICIPATION_SEEDS,
    config: ParticipationConfig | None = None,
) -> RetainedPredictions:
    """Fit all heads, retain seed predictions, and append arithmetic-mean ensembles."""
    features = tuple(feature_columns)
    if features != BASELINE_FEATURE_NAMES:
        raise ParticipationInputError(
            "runner requires the ordered frozen 46-feature contract without selection"
        )
    if tuple(seeds) != PARTICIPATION_SEEDS:
        raise ParticipationInputError(
            f"participation experiment requires seeds {PARTICIPATION_SEEDS!r} in that order"
        )
    if config is not None and config != ParticipationConfig():
        raise ParticipationInputError("the runner does not expose a hyperparameter tuning surface")

    retained_context = tuple(
        dict.fromkeys(
            [
                *context_columns,
                *(column for column in DEFAULT_COHORT_CONTEXT if column in frame.columns),
            ]
        )
    )

    def factory(seed: int, fold: object) -> ParticipationAwarePredictor:
        del fold
        return ParticipationAwarePredictor(seed=seed, config=config)

    individual = harness.run(
        frame,
        factory,
        feature_columns=features,
        target_columns=tuple(target_columns),
        prediction_columns=PREDICTION_COLUMNS,
        seeds=tuple(seeds),
        eligibility_column=eligibility_column,
        context_columns=retained_context,
        window_names=window_names,
    )
    return _with_participation_ensembles(individual)


def _with_participation_ensembles(artifact: RetainedPredictions) -> RetainedPredictions:
    frame = artifact.frame
    identity = [*artifact.key_columns, "fold"]
    expected_seeds = set(PARTICIPATION_SEEDS)
    observed = frame.groupby(identity, sort=False, dropna=False)["seed"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    if not observed.map(lambda values: values == expected_seeds).all():
        raise ParticipationInputError("every fold row must contain exactly the declared seeds")

    for source, destination in zip(ENSEMBLE_SOURCE_COLUMNS, ENSEMBLE_COLUMNS, strict=True):
        mean = frame.groupby(identity, sort=False, dropna=False)[source].transform("mean")
        values = mean.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ParticipationInputError(f"ensemble {destination!r} is non-finite")
        frame[destination] = values

    # Linear reconciliation constraints survive arithmetic averaging; assert rather than repair.
    if (frame["p_minutes_60_ensemble"] > frame["p_play_any_ensemble"] + 1e-12).any():
        raise ParticipationInputError("probability coherence was lost during ensembling")
    lower = 60.0 * frame["p_minutes_60_ensemble"]
    upper = frame["gw_max_minutes"] * frame["p_play_any_ensemble"]
    if (frame["expected_minutes_ensemble"] < lower - 1e-9).any() or (
        frame["expected_minutes_ensemble"] > upper + 1e-9
    ).any():
        raise ParticipationInputError("minutes coherence was lost during ensembling")

    return retained_predictions(
        frame,
        prediction_columns=(*artifact.prediction_columns, *ENSEMBLE_COLUMNS),
        target_columns=artifact.target_columns,
        baseline_columns=artifact.baseline_columns,
        key_columns=artifact.key_columns,
    )
