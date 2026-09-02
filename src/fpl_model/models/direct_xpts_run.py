"""Walk-forward orchestration for the fixed Sprint 8 direct-xPts experiment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fpl_model.evaluation.artifacts import RetainedPredictions, retained_predictions
from fpl_model.evaluation.harness import WalkForwardHarness
from fpl_model.features.contract import BASELINE_FEATURE_NAMES
from fpl_model.models.baselines import validate_ep_next_provenance
from fpl_model.models.direct_xpts import (
    BASE_ENSEMBLE_COLUMNS,
    DIRECT_XPTS_SEEDS,
    EP_NEXT_ENSEMBLE_COLUMNS,
    DirectXptsConfig,
    DirectXptsInputError,
    DirectXptsPredictor,
    direct_prediction_columns,
    ensemble_prediction_columns,
)

DEFAULT_DIRECT_XPTS_FEATURES = BASELINE_FEATURE_NAMES
DEFAULT_DIRECT_XPTS_TARGETS = ("y_points",)


def run_direct_xpts(
    frame: pd.DataFrame,
    harness: WalkForwardHarness,
    *,
    feature_columns: Sequence[str] = DEFAULT_DIRECT_XPTS_FEATURES,
    target_columns: Sequence[str] = DEFAULT_DIRECT_XPTS_TARGETS,
    context_columns: Sequence[str] = (),
    eligibility_column: str = "eligibility",
    window_names: Sequence[str] | None = None,
    seeds: Sequence[int] = DIRECT_XPTS_SEEDS,
    config: DirectXptsConfig | None = None,
) -> RetainedPredictions:
    """Run three fixed seeds and retain seed-level plus arithmetic-mean predictions.

    The primary arm always uses exactly the 46 baseline features. When at least one valid
    pre-deadline ``ep_next`` value exists, a named optional arm is trained alongside it. The
    no-``ep_next`` primary arm is therefore always present and directly auditable.
    """
    features = tuple(feature_columns)
    if features != BASELINE_FEATURE_NAMES:
        raise DirectXptsInputError(
            "run_direct_xpts requires the ordered frozen 46-feature contract without selection"
        )
    if tuple(seeds) != DIRECT_XPTS_SEEDS:
        raise DirectXptsInputError(f"Sprint 8 requires seeds {DIRECT_XPTS_SEEDS!r} in that order")
    if config is not None and config != DirectXptsConfig():
        raise DirectXptsInputError(
            "Sprint 8 uses the predeclared DirectXptsConfig; tuning is not allowed in this runner"
        )

    prepared = frame.copy()
    prepared["ep_next"] = validate_ep_next_provenance(prepared)
    include_ep_next_arm = bool(prepared["ep_next"].notna().any())
    harness_features = features + (("ep_next",) if include_ep_next_arm else ())
    predictions = direct_prediction_columns(include_ep_next_arm=include_ep_next_arm)

    def factory(seed: int, fold: object) -> DirectXptsPredictor:
        del fold
        return DirectXptsPredictor(
            seed=seed,
            include_ep_next_arm=include_ep_next_arm,
            config=config,
        )

    individual = harness.run(
        prepared,
        factory,
        feature_columns=harness_features,
        target_columns=tuple(target_columns),
        prediction_columns=predictions,
        seeds=tuple(seeds),
        baseline_columns=("ep_next",),
        eligibility_column=eligibility_column,
        context_columns=tuple(context_columns),
        window_names=window_names,
    )
    return _with_mean_ensemble(individual, include_ep_next_arm=include_ep_next_arm)


def _with_mean_ensemble(
    artifact: RetainedPredictions,
    *,
    include_ep_next_arm: bool,
) -> RetainedPredictions:
    frame = artifact.frame
    identity = [*artifact.key_columns, "fold"]
    expected_seeds = set(DIRECT_XPTS_SEEDS)
    observed = frame.groupby(identity, sort=False, dropna=False)["seed"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    if not observed.map(lambda values: values == expected_seeds).all():
        raise DirectXptsInputError("every fold row must contain exactly the three declared seeds")

    mappings = [
        ("xpts_direct_raw", BASE_ENSEMBLE_COLUMNS[0], BASE_ENSEMBLE_COLUMNS[1]),
    ]
    if include_ep_next_arm:
        mappings.append(
            (
                "xpts_direct_with_ep_next_raw",
                EP_NEXT_ENSEMBLE_COLUMNS[0],
                EP_NEXT_ENSEMBLE_COLUMNS[1],
            )
        )
    for raw_column, ensemble_raw, ensemble_clipped in mappings:
        mean_raw = frame.groupby(identity, sort=False, dropna=False)[raw_column].transform("mean")
        if not np.isfinite(mean_raw.to_numpy(dtype=float)).all():
            raise DirectXptsInputError("mean ensemble produced non-finite predictions")
        frame[ensemble_raw] = mean_raw
        # Inference clipping is applied after arithmetic averaging of the raw seed predictions.
        frame[ensemble_clipped] = np.maximum(mean_raw, 0.0)

    all_predictions = (
        *artifact.prediction_columns,
        *ensemble_prediction_columns(include_ep_next_arm=include_ep_next_arm),
    )
    return retained_predictions(
        frame,
        prediction_columns=all_predictions,
        target_columns=artifact.target_columns,
        baseline_columns=artifact.baseline_columns,
        key_columns=artifact.key_columns,
    )
