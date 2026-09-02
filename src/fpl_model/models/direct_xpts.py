"""Fold-local direct xPts models with chronological XGBoost early stopping."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from fpl_model.features.contract import BASELINE_FEATURE_NAMES

KEY_COLUMNS = ("season", "gameweek", "player_key")
DIRECT_XPTS_SEEDS = (42, 7, 2026)
BASE_ARM = "baseline_46"
EP_NEXT_ARM = "with_ep_next"

BASE_PREDICTION_COLUMNS = (
    "xpts_direct_raw",
    "xpts_direct",
    "xpts_direct_best_iteration",
    "xpts_direct_validation_mae",
    "pred_last5_points",
    "pred_points_ridge_raw",
    "pred_points_ridge",
    "rank_price",
)
EP_NEXT_PREDICTION_COLUMNS = (
    "xpts_direct_with_ep_next_raw",
    "xpts_direct_with_ep_next",
    "xpts_direct_with_ep_next_best_iteration",
    "xpts_direct_with_ep_next_validation_mae",
)
BASE_ENSEMBLE_COLUMNS = (
    "xpts_direct_ensemble_raw",
    "xpts_direct_ensemble",
)
EP_NEXT_ENSEMBLE_COLUMNS = (
    "xpts_direct_with_ep_next_ensemble_raw",
    "xpts_direct_with_ep_next_ensemble",
)


class DirectXptsInputError(ValueError):
    """Raised when direct-xPts training would violate the frozen experiment contract."""


@dataclass(frozen=True, slots=True)
class DirectXptsConfig:
    """Conservative predeclared XGBoost configuration; this is not a tuning surface."""

    n_estimators: int = 600
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: float = 10.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 5.0
    reg_alpha: float = 0.1
    early_stopping_rounds: int = 50
    objective: str = "reg:squarederror"
    eval_metric: str = "mae"
    tree_method: str = "hist"
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.n_estimators != 600:
            raise ValueError(
                "Sprint 8 fixes n_estimators at 600; hyperparameter tuning is deferred"
            )
        if self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive")
        if self.objective != "reg:squarederror" or self.eval_metric != "mae":
            raise ValueError("Sprint 8 requires square-error training with MAE early stopping")
        if self.tree_method != "hist":
            raise ValueError("Sprint 8 requires tree_method='hist'")

    def model_parameters(self, seed: int) -> dict[str, Any]:
        parameters = asdict(self)
        parameters["random_state"] = seed
        parameters["verbosity"] = 0
        return parameters


def direct_prediction_columns(*, include_ep_next_arm: bool) -> tuple[str, ...]:
    columns = BASE_PREDICTION_COLUMNS
    if include_ep_next_arm:
        columns += EP_NEXT_PREDICTION_COLUMNS
    return columns


def ensemble_prediction_columns(*, include_ep_next_arm: bool) -> tuple[str, ...]:
    columns = BASE_ENSEMBLE_COLUMNS
    if include_ep_next_arm:
        columns += EP_NEXT_ENSEMBLE_COLUMNS
    return columns


def build_fold_preprocessor(
    frame: pd.DataFrame,
    *,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Create an unfitted transformer whose statistics can only come from ``fit`` input."""
    if frame.shape[1] == 0:
        raise DirectXptsInputError("direct xPts requires at least one feature")
    numeric = [column for column in frame if is_numeric_dtype(frame[column])]
    categorical = [column for column in frame if column not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
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
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


class DirectXptsPredictor:
    """Fit the fixed 46-feature candidate and optional ``ep_next`` arm in one fold.

    Preprocessing is fitted on the training partition only. The already fitted transformer is
    then used for the disjoint chronological calibration block passed to XGBoost as its sole
    early-stopping evaluation set. Neither calibration nor test targets enter model fitting.
    """

    def __init__(
        self,
        *,
        seed: int,
        include_ep_next_arm: bool,
        config: DirectXptsConfig | None = None,
        model_factory: Callable[..., Any] = XGBRegressor,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.include_ep_next_arm = include_ep_next_arm
        self.config = config or DirectXptsConfig()
        self.model_factory = model_factory
        self.fitted_row_keys: list[tuple[object, ...]] = []
        self.preprocessors: dict[str, ColumnTransformer] = {}
        self.models: dict[str, Any] = {}
        self.ridge_pipeline: Pipeline | None = None
        self.feature_columns: tuple[str, ...] = ()
        self.target_column = ""

    def fit(
        self,
        train_frame: pd.DataFrame,
        calibration_frame: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...],
        target_columns: tuple[str, ...],
    ) -> None:
        self.target_column = _resolve_points_target(target_columns)
        self.feature_columns = tuple(feature_columns)
        _validate_feature_contract(self.feature_columns, self.include_ep_next_arm)
        required = (*KEY_COLUMNS, *self.feature_columns, self.target_column)
        _require_columns(train_frame, required)
        _require_columns(calibration_frame, required)
        if train_frame.empty or calibration_frame.empty:
            raise DirectXptsInputError(
                "training and chronological calibration blocks must be non-empty"
            )

        self.fitted_row_keys = list(
            train_frame.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None)
        )
        y_train = _finite_target(train_frame, self.target_column)
        y_calibration = _finite_target(calibration_frame, self.target_column)

        arms: dict[str, tuple[str, ...]] = {BASE_ARM: BASELINE_FEATURE_NAMES}
        if self.include_ep_next_arm:
            arms[EP_NEXT_ARM] = (*BASELINE_FEATURE_NAMES, "ep_next")
        self.preprocessors = {}
        self.models = {}
        for arm, columns in arms.items():
            train_features = train_frame.loc[:, list(columns)]
            calibration_features = calibration_frame.loc[:, list(columns)]
            preprocessor = build_fold_preprocessor(train_features)
            transformed_train = preprocessor.fit_transform(train_features)
            transformed_calibration = preprocessor.transform(calibration_features)
            model = self.model_factory(**self.config.model_parameters(self.seed))
            model.fit(
                transformed_train,
                y_train,
                eval_set=[(transformed_calibration, y_calibration)],
                verbose=False,
            )
            if not hasattr(model, "best_iteration") or not hasattr(model, "best_score"):
                raise DirectXptsInputError(
                    "XGBoost did not expose early-stopping evidence from the calibration block"
                )
            self.preprocessors[arm] = preprocessor
            self.models[arm] = model

        ridge_features = train_frame.loc[:, list(BASELINE_FEATURE_NAMES)]
        self.ridge_pipeline = Pipeline(
            [
                ("preprocess", build_fold_preprocessor(ridge_features, scale_numeric=True)),
                ("model", Ridge(alpha=10.0)),
            ]
        ).fit(ridge_features, y_train)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.models or self.ridge_pipeline is None:
            raise DirectXptsInputError("predictor must be fitted before predict")
        _require_columns(frame, (*KEY_COLUMNS, *self.feature_columns))
        result = frame.loc[:, list(KEY_COLUMNS)].copy()
        self._predict_arm(
            frame,
            result,
            arm=BASE_ARM,
            columns=BASELINE_FEATURE_NAMES,
            prefix="xpts_direct",
        )
        if self.include_ep_next_arm:
            self._predict_arm(
                frame,
                result,
                arm=EP_NEXT_ARM,
                columns=(*BASELINE_FEATURE_NAMES, "ep_next"),
                prefix="xpts_direct_with_ep_next",
            )

        result["pred_last5_points"] = _numeric_or_zero(frame, "points_mean_last5")
        ridge_raw = np.asarray(
            self.ridge_pipeline.predict(frame.loc[:, list(BASELINE_FEATURE_NAMES)]), dtype=float
        )
        result["pred_points_ridge_raw"] = ridge_raw
        result["pred_points_ridge"] = np.maximum(ridge_raw, 0.0)
        result["rank_price"] = _finite_feature(frame, "price")
        return result

    def _predict_arm(
        self,
        frame: pd.DataFrame,
        result: pd.DataFrame,
        *,
        arm: str,
        columns: Sequence[str],
        prefix: str,
    ) -> None:
        transformed = self.preprocessors[arm].transform(frame.loc[:, list(columns)])
        raw = np.asarray(self.models[arm].predict(transformed), dtype=float)
        if not np.isfinite(raw).all():
            raise DirectXptsInputError(f"{arm} produced non-finite predictions")
        result[f"{prefix}_raw"] = raw
        result[prefix] = np.maximum(raw, 0.0)
        result[f"{prefix}_best_iteration"] = float(self.models[arm].best_iteration)
        result[f"{prefix}_validation_mae"] = float(self.models[arm].best_score)


def _validate_feature_contract(features: tuple[str, ...], include_ep_next_arm: bool) -> None:
    required = set(BASELINE_FEATURE_NAMES)
    missing = sorted(required.difference(features))
    permitted = required | ({"ep_next"} if include_ep_next_arm else set())
    extra = sorted(set(features).difference(permitted))
    if missing or extra or len(features) != len(set(features)):
        raise DirectXptsInputError(
            "direct xPts features must be exactly the frozen 46-feature contract"
            + (" plus ep_next" if include_ep_next_arm else "")
            + f"; missing={missing!r}, extra={extra!r}"
        )
    if include_ep_next_arm != ("ep_next" in features):
        raise DirectXptsInputError("ep_next arm flag and feature columns disagree")


def _resolve_points_target(target_columns: Sequence[str]) -> str:
    for column in ("y_points", "actual_points_gw"):
        if column in target_columns:
            return column
    raise DirectXptsInputError("target_columns must contain y_points or actual_points_gw")


def _finite_target(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DirectXptsInputError(f"target {column!r} must contain finite numbers")
    return values


def _numeric_or_zero(frame: pd.DataFrame, column: str) -> np.ndarray:
    _require_columns(frame, (column,))
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _finite_feature(frame: pd.DataFrame, column: str) -> np.ndarray:
    _require_columns(frame, (column,))
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DirectXptsInputError(f"feature {column!r} must contain finite numbers")
    return values


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DirectXptsInputError(f"direct xPts frame is missing columns {missing!r}")
