"""Fold-local participation-aware XGBoost heads with chronological calibration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from fpl_model.evaluation.metrics import ranking_metrics_by_gameweek
from fpl_model.features.contract import BASELINE_FEATURE_NAMES
from fpl_model.models.direct_xpts import KEY_COLUMNS, build_fold_preprocessor

PARTICIPATION_SEEDS = (42, 7, 2026)
PARTICIPATION_TARGETS = (
    "y_points",
    "y_minutes",
    "y_play_any",
    "y_minutes_60",
    "y_points_if_play",
)

PREDICTION_COLUMNS = (
    "p_play_any_raw",
    "p_play_any_calibrated",
    "p_minutes_60_raw",
    "p_minutes_60_calibrated",
    "p_play_any",
    "p_minutes_60",
    "p_play_any_historical",
    "p_minutes_60_historical",
    "conditional_minutes_raw",
    "conditional_minutes",
    "expected_minutes_unconstrained",
    "expected_minutes",
    "conditional_points_raw",
    "conditional_points",
    "xpts_direct_raw",
    "xpts_direct",
    "xpts_conditional_raw",
    "xpts_conditional",
    "xpts_blend",
    "blend_direct_weight",
    "gw_max_minutes",
)

ENSEMBLE_SOURCE_COLUMNS = (
    "p_play_any_raw",
    "p_play_any_calibrated",
    "p_minutes_60_raw",
    "p_minutes_60_calibrated",
    "p_play_any",
    "p_minutes_60",
    "conditional_minutes_raw",
    "conditional_minutes",
    "expected_minutes_unconstrained",
    "expected_minutes",
    "conditional_points_raw",
    "conditional_points",
    "xpts_direct_raw",
    "xpts_direct",
    "xpts_conditional_raw",
    "xpts_conditional",
    "xpts_blend",
    "blend_direct_weight",
)
ENSEMBLE_COLUMNS = tuple(f"{column}_ensemble" for column in ENSEMBLE_SOURCE_COLUMNS)
PARTICIPATION_PREDICTION_COLUMNS = PREDICTION_COLUMNS
PARTICIPATION_ENSEMBLE_COLUMNS = ENSEMBLE_COLUMNS


class ParticipationInputError(ValueError):
    """Raised when a participation experiment would violate its causal contract."""


@dataclass(frozen=True, slots=True)
class ParticipationConfig:
    """Predeclared conservative configuration shared by all five boosted heads."""

    n_estimators: int = 600
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: float = 10.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 5.0
    reg_alpha: float = 0.1
    early_stopping_rounds: int = 50
    tree_method: str = "hist"
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.n_estimators != 600 or self.learning_rate <= 0:
            raise ValueError("participation experiments fix n_estimators at 600")
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        if self.tree_method != "hist":
            raise ValueError("participation heads require tree_method='hist'")

    def classifier_parameters(self, seed: int) -> dict[str, Any]:
        parameters = asdict(self)
        # The complete calibration block is reserved for post-fit Platt calibration.
        parameters.pop("early_stopping_rounds")
        return parameters | {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": seed,
            "verbosity": 0,
        }

    def regressor_parameters(self, seed: int) -> dict[str, Any]:
        return asdict(self) | {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "random_state": seed,
            "verbosity": 0,
        }


@dataclass(frozen=True, slots=True)
class _ConstantClassifier:
    probability: float

    def predict_proba(self, features: Any) -> np.ndarray:
        positive = np.full(features.shape[0], self.probability, dtype=float)
        return np.column_stack((1.0 - positive, positive))


@dataclass(frozen=True, slots=True)
class _ConstantCalibrator:
    probability: float

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return np.full(len(probability), self.probability, dtype=float)


class _PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2_000)

    def fit(self, probability: np.ndarray, target: np.ndarray) -> _PlattCalibrator:
        self.model.fit(_logit_feature(probability), target)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(_logit_feature(probability))[:, 1]


class ParticipationAwarePredictor:
    """Fit independent heads on training rows and Platt calibrators on a trailing block.

    Head predictions are combined only after fitting; no head prediction is an in-sample feature
    of another head. Conditional regressors see only training rows with positive minutes.
    """

    def __init__(
        self,
        *,
        seed: int,
        config: ParticipationConfig | None = None,
        expected_feature_columns: Sequence[str] = BASELINE_FEATURE_NAMES,
        classifier_factory: Callable[..., Any] = XGBClassifier,
        regressor_factory: Callable[..., Any] = XGBRegressor,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.config = config or ParticipationConfig()
        self.expected_feature_columns = tuple(expected_feature_columns)
        if (
            not self.expected_feature_columns
            or len(self.expected_feature_columns) != len(set(self.expected_feature_columns))
            or not set(self.expected_feature_columns).issubset(BASELINE_FEATURE_NAMES)
        ):
            raise ParticipationInputError(
                "expected features must be a non-empty unique subset of the baseline contract"
            )
        self.classifier_factory = classifier_factory
        self.regressor_factory = regressor_factory
        self.fitted_row_keys: list[tuple[object, ...]] = []
        self.feature_columns: tuple[str, ...] = ()
        self.targets: dict[str, str] = {}
        self.preprocessors: dict[str, Any] = {}
        self.models: dict[str, Any] = {}
        self.calibrators: dict[str, Any] = {}
        self.base_rates: dict[str, float] = {}
        self.blend_direct_weight = 1.0

    def fit(
        self,
        train_frame: pd.DataFrame,
        calibration_frame: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        target_columns: tuple[str, ...],
    ) -> None:
        self.feature_columns = tuple(feature_columns)
        if self.feature_columns != self.expected_feature_columns:
            raise ParticipationInputError(
                "participation model features differ from the predeclared arm contract"
            )
        self.targets = _resolve_targets(target_columns)
        required = (*KEY_COLUMNS, *self.feature_columns, *self.targets.values())
        _require_columns(train_frame, required)
        _require_columns(calibration_frame, required)
        if train_frame.empty or calibration_frame.empty:
            raise ParticipationInputError("training and calibration blocks must be non-empty")
        _validate_target_coherence(train_frame, self.targets)
        _validate_target_coherence(calibration_frame, self.targets)

        self.fitted_row_keys = list(
            train_frame.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None)
        )
        y_play = _binary_target(train_frame, self.targets["play"])
        y_sixty = _binary_target(train_frame, self.targets["sixty"])
        if np.any(y_sixty > y_play):
            raise ParticipationInputError("y_minutes_60 cannot be positive when y_play_any is zero")
        self.base_rates = {"play": float(y_play.mean()), "sixty": float(y_sixty.mean())}

        self._fit_classifier("play", train_frame, y_play)
        self._fit_classifier("sixty", train_frame, y_sixty)
        self._fit_calibrator(
            "play", calibration_frame, _binary_target(calibration_frame, self.targets["play"])
        )
        self._fit_calibrator(
            "sixty", calibration_frame, _binary_target(calibration_frame, self.targets["sixty"])
        )

        played = y_play.astype(bool)
        calibration_played = _binary_target(calibration_frame, self.targets["play"]).astype(bool)
        if not played.any():
            raise ParticipationInputError("conditional regressors require played training rows")
        self._fit_regressor(
            "minutes",
            train_frame.loc[played],
            calibration_frame.loc[calibration_played],
            self.targets["minutes"],
        )
        self._fit_regressor(
            "points",
            train_frame.loc[played],
            calibration_frame.loc[calibration_played],
            self.targets["points_if_play"],
        )
        self._fit_regressor("direct", train_frame, calibration_frame, self.targets["points"])
        self.blend_direct_weight = self._select_blend_weight(calibration_frame)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.models or not self.calibrators:
            raise ParticipationInputError("predictor must be fitted before predict")
        _require_columns(frame, (*KEY_COLUMNS, *self.feature_columns))
        values = self._prediction_values(frame)
        result = frame.loc[:, list(KEY_COLUMNS)].copy()
        for column in PREDICTION_COLUMNS:
            result[column] = values[column]
        return result

    def _fit_classifier(self, name: str, frame: pd.DataFrame, target: np.ndarray) -> None:
        features = frame.loc[:, list(self.feature_columns)]
        preprocessor = build_fold_preprocessor(features)
        transformed = preprocessor.fit_transform(features)
        if np.unique(target).size == 1:
            model: Any = _ConstantClassifier(float(target[0]))
        else:
            model = self.classifier_factory(**self.config.classifier_parameters(self.seed))
            model.fit(transformed, target, verbose=False)
        self.preprocessors[name] = preprocessor
        self.models[name] = model

    def _fit_calibrator(self, name: str, frame: pd.DataFrame, target: np.ndarray) -> None:
        raw = self._raw_probability(name, frame)
        if np.unique(target).size == 1 or np.ptp(raw) <= 1e-12:
            calibrator: Any = _ConstantCalibrator(float(target.mean()))
        else:
            calibrator = _PlattCalibrator().fit(raw, target)
        self.calibrators[name] = calibrator

    def _fit_regressor(
        self,
        name: str,
        frame: pd.DataFrame,
        calibration_frame: pd.DataFrame,
        target_column: str,
    ) -> None:
        features = frame.loc[:, list(self.feature_columns)]
        target = _finite_target(frame, target_column)
        preprocessor = build_fold_preprocessor(features)
        transformed = preprocessor.fit_transform(features)
        parameters = self.config.regressor_parameters(self.seed)
        if calibration_frame.empty:
            parameters.pop("early_stopping_rounds")
        model = self.regressor_factory(**parameters)
        fit_arguments: dict[str, Any] = {"verbose": False}
        if not calibration_frame.empty:
            calibration_features = calibration_frame.loc[:, list(self.feature_columns)]
            transformed_calibration = preprocessor.transform(calibration_features)
            calibration_target = _finite_target(calibration_frame, target_column)
            fit_arguments["eval_set"] = [(transformed_calibration, calibration_target)]
        model.fit(transformed, target, **fit_arguments)
        self.preprocessors[name] = preprocessor
        self.models[name] = model

    def _raw_probability(self, name: str, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessors[name].transform(frame.loc[:, list(self.feature_columns)])
        probability = np.asarray(self.models[name].predict_proba(transformed), dtype=float)[:, 1]
        return np.clip(probability, 0.0, 1.0)

    def _regression(self, name: str, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessors[name].transform(frame.loc[:, list(self.feature_columns)])
        values = np.asarray(self.models[name].predict(transformed), dtype=float)
        if not np.isfinite(values).all():
            raise ParticipationInputError(f"{name} head produced non-finite predictions")
        return values

    def _prediction_values(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        raw_play = self._raw_probability("play", frame)
        raw_sixty = self._raw_probability("sixty", frame)
        calibrated_play = np.clip(self.calibrators["play"].predict(raw_play), 0.0, 1.0)
        calibrated_sixty = np.clip(self.calibrators["sixty"].predict(raw_sixty), 0.0, 1.0)
        fixture_count = _finite_feature(frame, "fixture_count")
        if (fixture_count < 0).any():
            raise ParticipationInputError("fixture_count must be non-negative")
        if not np.equal(fixture_count, np.floor(fixture_count)).all():
            raise ParticipationInputError("fixture_count must contain whole numbers")
        gw_max = 90.0 * fixture_count

        # A confirmed blank cannot have participation; otherwise reconcile the threshold heads.
        play = np.where(gw_max > 0, calibrated_play, 0.0)
        sixty = np.where(gw_max >= 60, np.minimum(calibrated_sixty, play), 0.0)

        conditional_minutes_raw = self._regression("minutes", frame)
        conditional_minutes = np.clip(conditional_minutes_raw, 0.0, gw_max)
        expected_unconstrained = play * conditional_minutes
        lower = 60.0 * sixty
        upper = gw_max * play
        expected_minutes = np.minimum(np.maximum(expected_unconstrained, lower), upper)

        conditional_points_raw = self._regression("points", frame)
        conditional_points = np.maximum(conditional_points_raw, 0.0)
        direct_raw = self._regression("direct", frame)
        direct = np.maximum(direct_raw, 0.0)
        xpts_conditional_raw = play * conditional_points_raw
        xpts_conditional = play * conditional_points
        blend = (
            self.blend_direct_weight * direct + (1.0 - self.blend_direct_weight) * xpts_conditional
        )
        return {
            "p_play_any_raw": raw_play,
            "p_play_any_calibrated": calibrated_play,
            "p_minutes_60_raw": raw_sixty,
            "p_minutes_60_calibrated": calibrated_sixty,
            "p_play_any": play,
            "p_minutes_60": sixty,
            "p_play_any_historical": _rate_or_base(
                frame, "play_any_rate_last5", self.base_rates["play"]
            ),
            "p_minutes_60_historical": _rate_or_base(
                frame, "minutes_60_rate_last5", self.base_rates["sixty"]
            ),
            "conditional_minutes_raw": conditional_minutes_raw,
            "conditional_minutes": conditional_minutes,
            "expected_minutes_unconstrained": expected_unconstrained,
            "expected_minutes": expected_minutes,
            "conditional_points_raw": conditional_points_raw,
            "conditional_points": conditional_points,
            "xpts_direct_raw": direct_raw,
            "xpts_direct": direct,
            "xpts_conditional_raw": xpts_conditional_raw,
            "xpts_conditional": xpts_conditional,
            "xpts_blend": blend,
            "blend_direct_weight": np.full(len(frame), self.blend_direct_weight),
            "gw_max_minutes": gw_max,
        }

    def _select_blend_weight(self, calibration_frame: pd.DataFrame) -> float:
        # Calibrators are already fitted on this explicitly designated selection block.
        self.blend_direct_weight = 1.0
        values = self._prediction_values(calibration_frame)
        actual = self.targets["points"]
        candidates: list[tuple[float, float, float]] = []
        for weight in np.linspace(0.0, 1.0, 21):
            prediction = (
                weight * values["xpts_direct"] + (1.0 - weight) * values["xpts_conditional"]
            )
            scored = calibration_frame.loc[:, [*KEY_COLUMNS, actual]].copy()
            scored["_blend"] = prediction
            report = ranking_metrics_by_gameweek(
                scored, actual_column=actual, prediction_column="_blend"
            )
            # NDCG is primary; MAE breaks ties; then prefer the simpler direct head.
            candidates.append((float(report.summary["ndcg_at_10"]), -report.summary["mae"], weight))
        return float(max(candidates)[2])


def _resolve_targets(columns: Sequence[str]) -> dict[str, str]:
    available = set(columns)

    def choose(candidates: Sequence[str], label: str) -> str:
        try:
            return next(column for column in candidates if column in available)
        except StopIteration as error:
            raise ParticipationInputError(f"target_columns must include {label}") from error

    return {
        "points": choose(("y_points", "actual_points_gw"), "a points target"),
        "minutes": choose(("y_minutes", "actual_minutes_gw"), "a minutes target"),
        "play": choose(("y_play_any",), "y_play_any"),
        "sixty": choose(("y_minutes_60",), "y_minutes_60"),
        "points_if_play": choose(
            ("y_points_if_play", "y_points", "actual_points_gw"), "points-if-play"
        ),
    }


def _finite_target(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ParticipationInputError(f"target {column!r} must be finite on fitted rows")
    return values


def _validate_target_coherence(frame: pd.DataFrame, targets: dict[str, str]) -> None:
    minutes = _finite_target(frame, targets["minutes"])
    play = _binary_target(frame, targets["play"])
    sixty = _binary_target(frame, targets["sixty"])
    fixture_count = _finite_feature(frame, "fixture_count")
    if (minutes < 0).any() or not np.array_equal(play, (minutes > 0).astype(int)):
        raise ParticipationInputError("y_play_any must equal 1(y_minutes > 0)")
    if not np.array_equal(sixty, (minutes >= 60).astype(int)):
        raise ParticipationInputError("y_minutes_60 must equal 1(y_minutes >= 60)")
    if (minutes > 90.0 * fixture_count + 1e-9).any():
        raise ParticipationInputError("y_minutes exceeds the rule-consistent GW maximum")

    points_if_play = targets["points_if_play"]
    if points_if_play not in {targets["points"]}:
        conditional = pd.to_numeric(frame[points_if_play], errors="coerce").to_numpy(dtype=float)
        points = _finite_target(frame, targets["points"])
        played = play.astype(bool)
        if not np.isfinite(conditional[played]).all() or not np.allclose(
            conditional[played], points[played]
        ):
            raise ParticipationInputError("y_points_if_play must equal y_points on played rows")
        if np.isfinite(conditional[~played]).any():
            raise ParticipationInputError("y_points_if_play must be missing on non-played rows")


def _binary_target(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _finite_target(frame, column)
    if not np.isin(values, (0.0, 1.0)).all():
        raise ParticipationInputError(f"target {column!r} must contain only 0 and 1")
    return values.astype(int)


def _finite_feature(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ParticipationInputError(f"feature {column!r} must be finite")
    return values


def _rate_or_base(frame: pd.DataFrame, column: str, base_rate: float) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").fillna(base_rate).to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ParticipationInputError(f"historical rate {column!r} must be in [0, 1]")
    return values


def _logit_feature(probability: np.ndarray) -> np.ndarray:
    epsilon = np.finfo(float).eps
    clipped = np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ParticipationInputError(f"participation frame is missing columns {missing!r}")
