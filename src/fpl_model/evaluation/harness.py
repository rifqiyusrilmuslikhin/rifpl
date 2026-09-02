"""Model-agnostic walk-forward runner that emits retained OOF predictions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import pandas as pd

from fpl_model.evaluation.artifacts import RetainedPredictions, retained_predictions
from fpl_model.evaluation.comparison import PairingError, assert_exact_same_rows
from fpl_model.evaluation.leakage_check import (
    LeakageError,
    assert_feature_columns_safe,
    assert_fold_local_fit,
)
from fpl_model.evaluation.windows import ExpandingWindowSplitter, WalkForwardFold


class FoldPredictor(Protocol):
    """Minimal causal predictor contract used by the evaluation harness."""

    fitted_row_keys: Sequence[object]

    def fit(
        self,
        train_frame: pd.DataFrame,
        calibration_frame: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        target_columns: tuple[str, ...],
    ) -> None: ...

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame: ...


PredictorFactory = Callable[[int, WalkForwardFold], FoldPredictor]


class WalkForwardHarness:
    """Fit once per fold/seed and preserve exact test rows for later rescoring."""

    def __init__(self, splitter: ExpandingWindowSplitter) -> None:
        self.splitter = splitter

    def run(
        self,
        frame: pd.DataFrame,
        predictor_factory: PredictorFactory,
        *,
        feature_columns: Sequence[str],
        target_columns: Sequence[str],
        prediction_columns: Sequence[str],
        seeds: Sequence[int] = (42,),
        baseline_columns: Sequence[str] = (),
        eligibility_column: str = "eligibility",
        context_columns: Sequence[str] = (),
        window_names: Sequence[str] | None = None,
    ) -> RetainedPredictions:
        features = tuple(feature_columns)
        targets = tuple(target_columns)
        predictions = tuple(prediction_columns)
        baselines = tuple(baseline_columns)
        keys = self.splitter.plan.key_columns
        _require_unique_names("feature", features, allow_empty=True)
        _require_unique_names("target", targets)
        _require_unique_names("prediction", predictions)
        _require_unique_names("baseline", baselines, allow_empty=True)
        assert_feature_columns_safe(features)
        feature_target_overlap = set(features).intersection(targets)
        if feature_target_overlap:
            raise LeakageError(
                f"target columns cannot be model features: {sorted(feature_target_overlap)!r}"
            )
        prediction_overlap = set(predictions).intersection(
            (*keys, *features, *targets, *baselines, *context_columns, eligibility_column)
        )
        if prediction_overlap:
            raise ValueError(
                f"prediction columns must be new output fields: {sorted(prediction_overlap)!r}"
            )
        required = {
            *keys,
            "deadline_utc",
            *features,
            *targets,
            *baselines,
            *context_columns,
            eligibility_column,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise LeakageError(f"walk-forward frame is missing required columns {missing!r}")
        if (
            not seeds
            or len(seeds) != len(set(seeds))
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        ):
            raise ValueError("seeds must be a non-empty unique sequence of integers")

        outputs: list[pd.DataFrame] = []
        for fold in self.splitter.split(frame, window_names=window_names):
            train, calibration, test = fold.frames(frame)
            training_keys = _row_keys(train, keys)
            held_out_keys = [*_row_keys(calibration, keys), *_row_keys(test, keys)]
            # Retained context can include post-GW diagnostics such as actual minutes. It is never
            # exposed to predict; only audited feature columns and row identity cross that boundary.
            predictor_input_columns = [*keys, *features]
            predictor_input = test.loc[:, list(dict.fromkeys(predictor_input_columns))].copy()
            for seed in seeds:
                predictor = predictor_factory(seed, fold)
                predictor.fit(
                    train.copy(),
                    calibration.copy(),
                    feature_columns=features,
                    target_columns=targets,
                )
                if not hasattr(predictor, "fitted_row_keys"):
                    raise LeakageError("predictor must retain fitted_row_keys after fit")
                assert_fold_local_fit(
                    training_keys,
                    predictor.fitted_row_keys,
                    held_out_row_keys=held_out_keys,
                )
                predicted = predictor.predict(predictor_input.copy())
                if not isinstance(predicted, pd.DataFrame):
                    raise TypeError("predictor.predict must return a pandas DataFrame")
                _validate_prediction_frame(predicted, test, keys, predictions)
                retained_columns = list(
                    dict.fromkeys(
                        [*keys, eligibility_column, *targets, *baselines, *context_columns]
                    )
                )
                retained = test.loc[:, retained_columns].copy()
                retained = retained.merge(
                    predicted.loc[:, [*keys, *predictions]],
                    on=list(keys),
                    how="left",
                    validate="one_to_one",
                )
                retained.insert(len(keys), "fold", fold.name)
                retained.insert(len(keys) + 1, "seed", seed)
                if eligibility_column != "eligibility":
                    retained = retained.rename(columns={eligibility_column: "eligibility"})
                outputs.append(retained)
        if not outputs:
            raise LeakageError("walk-forward splitter produced no test predictions")
        result = pd.concat(outputs, ignore_index=True)
        return retained_predictions(
            result,
            prediction_columns=predictions,
            target_columns=targets,
            baseline_columns=baselines,
            key_columns=keys,
        )


def regenerate_ranking_report(
    artifact: RetainedPredictions,
    *,
    actual_column: str = "actual_points_gw",
    prediction_column: str = "predicted_points",
    eligibility_only: bool = True,
) -> Mapping[tuple[str, int], Any]:
    """Regenerate independent reports for every fold/seed without fitting a model."""
    from fpl_model.evaluation.metrics import ranking_metrics_by_gameweek

    reports: dict[tuple[str, int], Any] = {}
    for (fold, seed), group in artifact.frame.groupby(["fold", "seed"], sort=True):
        if eligibility_only:
            group = group.loc[group["eligibility"]]
        reports[(str(fold), int(seed))] = ranking_metrics_by_gameweek(
            group,
            actual_column=actual_column,
            prediction_column=prediction_column,
        )
    return reports


def regenerate_probability_report(
    artifact: RetainedPredictions,
    *,
    target_column: str,
    probability_column: str,
    eligibility_only: bool = True,
    bins: int = 10,
) -> Mapping[tuple[str, int], Any]:
    """Regenerate probability and calibration reports without fitting a model."""
    from fpl_model.evaluation.metrics import probability_metrics_by_gameweek

    reports: dict[tuple[str, int], Any] = {}
    for (fold, seed), group in artifact.frame.groupby(["fold", "seed"], sort=True):
        if eligibility_only:
            group = group.loc[group["eligibility"]]
        reports[(str(fold), int(seed))] = probability_metrics_by_gameweek(
            group,
            target_column=target_column,
            probability_column=probability_column,
            bins=bins,
        )
    return reports


def _validate_prediction_frame(
    predicted: pd.DataFrame,
    test: pd.DataFrame,
    key_columns: tuple[str, ...],
    prediction_columns: tuple[str, ...],
) -> None:
    missing = sorted(set((*key_columns, *prediction_columns)).difference(predicted.columns))
    if missing:
        raise PairingError(f"prediction output is missing columns {missing!r}")
    expected = test.loc[:, list(key_columns)].copy()
    for column in prediction_columns:
        expected[column] = 0.0
    candidate = predicted.loc[:, [*key_columns, *prediction_columns]].copy()
    numeric = candidate.loc[:, list(prediction_columns)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any(axis=None) or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise PairingError("prediction output must contain finite numeric values")
    # Targets here are harmless sentinels; this reuses exact key-set and duplicate validation.
    assert_exact_same_rows(
        candidate.rename(columns={prediction_columns[0]: "_sentinel"}),
        expected.rename(columns={prediction_columns[0]: "_sentinel"}),
        key_columns=key_columns,
        target_columns=(),
    )


def _row_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[tuple[Any, ...]]:
    return list(frame.loc[:, list(columns)].itertuples(index=False, name=None))


def _require_unique_names(label: str, names: tuple[str, ...], *, allow_empty: bool = False) -> None:
    if (not names and not allow_empty) or len(names) != len(set(names)):
        qualifier = "possibly empty " if allow_empty else "non-empty "
        raise ValueError(f"{label} columns must be a {qualifier}unique sequence")
