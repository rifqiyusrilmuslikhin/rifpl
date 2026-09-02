"""Deterministic and regularized simple baselines for the shared walk-forward harness."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fpl_model.evaluation.windows import WalkForwardFold

KEY_COLUMNS = ("season", "gameweek", "player_key")

POINT_TARGETS = ("y_points", "actual_points_gw")
MINUTES_TARGETS = ("y_minutes", "actual_minutes_gw")
PLAY_TARGETS = ("y_play_any",)
SIXTY_TARGETS = ("y_minutes_60",)

PREDICTION_COLUMNS = (
    "pred_zero_points",
    "pred_last_appearance_points",
    "pred_last5_points",
    "pred_position_minutes_points",
    "rank_price",
    "has_last_appearance_history",
    "has_last5_points_history",
    "has_play_rate_history",
    "has_minutes_60_rate_history",
    "has_minutes_mean_history",
    "p_play_any_base_rate",
    "p_minutes_60_base_rate",
    "p_play_any_historical",
    "p_minutes_60_historical",
    "pred_minutes_historical",
    "p_play_any_logistic",
    "p_minutes_60_logistic",
    "pred_points_ridge_raw",
    "pred_points_ridge",
    "pred_minutes_ridge_raw",
    "pred_minutes_ridge",
)


class BaselineInputError(ValueError):
    """Raised when a baseline input would be ambiguous or non-causal."""


@dataclass(frozen=True, slots=True)
class BaselineTargets:
    points: str
    minutes: str
    play_any: str
    minutes_60: str


class SimpleBaselinePredictor:
    """Fit all Sprint 7 baselines together on one fold and one exact test row set.

    The sklearn pipelines own their imputers, encoders, and scalers, and are newly constructed for
    each predictor instance. Historical cold starts use training-fold base rates; point-history
    cold starts use zero because the player has no prior EPL points. Official ``ep_next`` remains
    outside this predictor so missing values can remain missing in the retained artifact.
    """

    def __init__(self, *, seed: int = 42, ridge_alpha: float = 10.0) -> None:
        if ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        self.seed = seed
        self.ridge_alpha = float(ridge_alpha)
        self.fitted_row_keys: list[tuple[object, ...]] = []
        self.targets: BaselineTargets | None = None
        self.feature_columns: tuple[str, ...] = ()
        self.model_columns: tuple[str, ...] = ()
        self.base_rates: dict[str, float] = {}
        self.position_minutes_means: dict[tuple[str, str], float] = {}
        self.global_points_mean = 0.0
        self.models: dict[str, Any] = {}

    def fit(
        self,
        train_frame: pd.DataFrame,
        calibration_frame: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        target_columns: tuple[str, ...],
    ) -> None:
        del calibration_frame  # Logistic and Ridge need no calibration/early-stopping stage.
        self.targets = _resolve_targets(target_columns)
        _require_columns(
            train_frame,
            (
                *KEY_COLUMNS,
                *feature_columns,
                self.targets.points,
                self.targets.minutes,
                self.targets.play_any,
                self.targets.minutes_60,
            ),
        )
        self.fitted_row_keys = list(
            train_frame.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None)
        )
        self.feature_columns = feature_columns
        self.model_columns = tuple(
            column
            for column in feature_columns
            if column not in {"points_last_appearance", "ep_next", "ep_next_value_state"}
        )
        if not self.model_columns:
            raise BaselineInputError("at least one model feature is required")

        y_points = _finite_target(train_frame, self.targets.points)
        y_minutes = _finite_target(train_frame, self.targets.minutes)
        y_play = _binary_target(train_frame, self.targets.play_any)
        y_sixty = _binary_target(train_frame, self.targets.minutes_60)
        self.base_rates = {
            "play": float(y_play.mean()),
            "sixty": float(y_sixty.mean()),
            "minutes": float(y_minutes.mean()),
        }
        self.global_points_mean = float(y_points.mean())
        self._fit_position_minutes_means(train_frame, y_points)

        x = train_frame.loc[:, list(self.model_columns)]
        self.models = {
            "play": _fit_classifier(x, y_play, seed=self.seed),
            "sixty": _fit_classifier(x, y_sixty, seed=self.seed),
            "points": _fit_regressor(x, y_points, alpha=self.ridge_alpha),
            "minutes": _fit_regressor(x, y_minutes, alpha=self.ridge_alpha),
        }

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.targets is None or not self.models:
            raise BaselineInputError("predictor must be fitted before predict")
        _require_columns(frame, (*KEY_COLUMNS, *self.feature_columns))
        result = frame.loc[:, list(KEY_COLUMNS)].copy()
        result["pred_zero_points"] = 0.0
        result["pred_last_appearance_points"] = _history_or_zero(frame, "points_last_appearance")
        result["pred_last5_points"] = _history_or_zero(frame, "points_mean_last5")
        result["pred_position_minutes_points"] = self._position_minutes_prediction(frame)
        result["rank_price"] = _finite_feature(frame, "price")
        result["has_last_appearance_history"] = _availability(frame, "points_last_appearance")
        result["has_last5_points_history"] = _availability(frame, "points_mean_last5")
        result["has_play_rate_history"] = _availability(frame, "play_any_rate_last5")
        result["has_minutes_60_rate_history"] = _availability(frame, "minutes_60_rate_last5")
        result["has_minutes_mean_history"] = _availability(frame, "minutes_mean_last5")
        result["p_play_any_base_rate"] = self.base_rates["play"]
        result["p_minutes_60_base_rate"] = self.base_rates["sixty"]
        result["p_play_any_historical"] = _rate_or_base(
            frame, "play_any_rate_last5", self.base_rates["play"]
        )
        result["p_minutes_60_historical"] = _rate_or_base(
            frame, "minutes_60_rate_last5", self.base_rates["sixty"]
        )
        result["pred_minutes_historical"] = _value_or_base(
            frame, "minutes_mean_last5", self.base_rates["minutes"]
        )

        x = frame.loc[:, list(self.model_columns)]
        result["p_play_any_logistic"] = _classifier_probability(self.models["play"], x)
        result["p_minutes_60_logistic"] = _classifier_probability(self.models["sixty"], x)
        raw_points = np.asarray(self.models["points"].predict(x), dtype=float)
        raw_minutes = np.asarray(self.models["minutes"].predict(x), dtype=float)
        result["pred_points_ridge_raw"] = raw_points
        result["pred_points_ridge"] = np.maximum(raw_points, 0.0)
        result["pred_minutes_ridge_raw"] = raw_minutes
        # A GW can contain a double header, so 180 rather than 90 is the conservative upper bound.
        result["pred_minutes_ridge"] = np.clip(raw_minutes, 0.0, 180.0)
        return result

    def _fit_position_minutes_means(self, frame: pd.DataFrame, points: np.ndarray) -> None:
        _require_columns(frame, ("position", "minutes_last_appearance"))
        statuses = _minutes_status(frame["minutes_last_appearance"])
        grouped = (
            pd.DataFrame(
                {
                    "position": frame["position"].astype("string").fillna("missing").to_numpy(),
                    "minutes_status": statuses.to_numpy(),
                    "points": points,
                }
            )
            .groupby(["position", "minutes_status"], observed=True)["points"]
            .mean()
        )
        self.position_minutes_means = {
            (str(position), str(status)): float(value)
            for (position, status), value in grouped.items()
        }

    def _position_minutes_prediction(self, frame: pd.DataFrame) -> np.ndarray:
        positions = frame["position"].astype("string").fillna("missing")
        statuses = _minutes_status(frame["minutes_last_appearance"])
        return np.asarray(
            [
                self.position_minutes_means.get(
                    (str(position), str(status)), self.global_points_mean
                )
                for position, status in zip(positions, statuses, strict=True)
            ],
            dtype=float,
        )


def baseline_predictor_factory(
    seed: int, fold: WalkForwardFold, *, ridge_alpha: float = 10.0
) -> SimpleBaselinePredictor:
    """Factory compatible with :class:`WalkForwardHarness`. The fold is audit context only."""
    del fold
    return SimpleBaselinePredictor(seed=seed, ridge_alpha=ridge_alpha)


def validate_ep_next_provenance(
    frame: pd.DataFrame,
    *,
    value_column: str = "ep_next",
    state_column: str = "ep_next_value_state",
    captured_column: str = "snapshot_captured_at_utc",
    deadline_column: str = "deadline_utc",
) -> pd.Series:
    """Return valid official xPts without ever replacing unavailable values with zero."""
    _require_columns(frame, (value_column, state_column, captured_column, deadline_column))
    captured = pd.to_datetime(frame[captured_column], utc=True, errors="coerce")
    deadlines = pd.to_datetime(frame[deadline_column], utc=True, errors="coerce")
    if captured.isna().any() or deadlines.isna().any() or captured.ge(deadlines).any():
        raise BaselineInputError("ep_next requires valid strictly pre-deadline snapshot provenance")
    states = frame[state_column].astype("string")
    allowed = {"value", "genuine_zero", "source_unavailable", "acquisition_failure"}
    if not states.isin(allowed).all():
        raise BaselineInputError("ep_next contains an unknown value state")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    available = states.isin({"value", "genuine_zero"})
    if values[available].isna().any() or values[~available].notna().any():
        raise BaselineInputError("ep_next value and value-state provenance disagree")
    if (states.eq("value") & values.eq(0)).any() or (
        states.eq("genuine_zero") & values.ne(0)
    ).any():
        raise BaselineInputError("ep_next zero must be distinguished as genuine_zero")
    return values.astype(float)


def _resolve_targets(columns: Sequence[str]) -> BaselineTargets:
    values = set(columns)

    def choose(candidates: Sequence[str], label: str) -> str:
        matches = [column for column in candidates if column in values]
        if not matches:
            raise BaselineInputError(f"target_columns must include a {label} target")
        return matches[0]

    return BaselineTargets(
        points=choose(POINT_TARGETS, "points"),
        minutes=choose(MINUTES_TARGETS, "minutes"),
        play_any=choose(PLAY_TARGETS, "play-any"),
        minutes_60=choose(SIXTY_TARGETS, "60-plus"),
    )


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in frame if is_numeric_dtype(frame[column])]
    categorical = [column for column in frame if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def _fit_classifier(frame: pd.DataFrame, target: np.ndarray, *, seed: int) -> Any:
    if np.unique(target).size == 1:
        return _ConstantClassifier(float(target[0]))
    model = Pipeline(
        [
            ("preprocess", _preprocessor(frame)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=2_000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return model.fit(frame, target)


def _fit_regressor(frame: pd.DataFrame, target: np.ndarray, *, alpha: float) -> Any:
    model = Pipeline([("preprocess", _preprocessor(frame)), ("model", Ridge(alpha=alpha))])
    return model.fit(frame, target)


@dataclass(frozen=True, slots=True)
class _ConstantClassifier:
    probability: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(frame), self.probability, dtype=float)
        return np.column_stack((1.0 - positive, positive))


def _classifier_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model.predict_proba(frame), dtype=float)[:, 1], 0.0, 1.0)


def _finite_target(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise BaselineInputError(f"target {column!r} must contain finite numbers")
    return values


def _binary_target(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = _finite_target(frame, column)
    if not np.isin(values, (0.0, 1.0)).all():
        raise BaselineInputError(f"target {column!r} must contain only 0 and 1")
    return values.astype(int)


def _finite_feature(frame: pd.DataFrame, column: str) -> np.ndarray:
    _require_columns(frame, (column,))
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise BaselineInputError(f"baseline feature {column!r} must be finite")
    return values


def _history_or_zero(frame: pd.DataFrame, column: str) -> np.ndarray:
    _require_columns(frame, (column,))
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _availability(frame: pd.DataFrame, column: str) -> np.ndarray:
    _require_columns(frame, (column,))
    return pd.to_numeric(frame[column], errors="coerce").notna().to_numpy(dtype=float)


def _rate_or_base(frame: pd.DataFrame, column: str, base: float) -> np.ndarray:
    values = _value_or_base(frame, column, base)
    if ((values < 0) | (values > 1)).any():
        raise BaselineInputError(f"historical probability {column!r} must be in [0, 1]")
    return values


def _value_or_base(frame: pd.DataFrame, column: str, base: float) -> np.ndarray:
    _require_columns(frame, (column,))
    values = pd.to_numeric(frame[column], errors="coerce").fillna(base).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise BaselineInputError(f"baseline feature {column!r} must be numeric when present")
    return values


def _minutes_status(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [numeric.isna(), numeric.eq(0), numeric.ge(60)],
            ["no_history", "zero", "60_plus"],
            default="cameo",
        ),
        index=values.index,
        dtype="string",
    )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise BaselineInputError(f"baseline frame is missing required columns {missing!r}")
