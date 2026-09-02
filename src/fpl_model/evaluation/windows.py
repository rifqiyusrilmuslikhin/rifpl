"""Frozen evaluation-window policy and chronological expanding folds."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from fpl_model.evaluation.leakage_check import LeakageError, assert_chronological_fold

DEFAULT_EVALUATION_CONFIG = Path("config/evaluation_windows.toml")
_SEASON_PATTERN = re.compile(r"^(?P<start>\d{4})-(?:\d{2}|\d{4})$")


class WindowRole(StrEnum):
    WARMUP = "warmup"
    CALIBRATION = "calibration"
    DISCOVERY = "discovery"
    CONFIRMATION = "confirmation"
    PROSPECTIVE = "prospective"


class EvaluationConfigError(ValueError):
    """Raised when frozen evaluation policy is invalid or misused."""


@dataclass(frozen=True, slots=True, order=True)
class Period:
    season_start: int
    gameweek: int

    @classmethod
    def from_values(cls, season: str, gameweek: int) -> Period:
        match = _SEASON_PATTERN.fullmatch(str(season).strip())
        if match is None:
            raise EvaluationConfigError(f"season must use YYYY-YY or YYYY-YYYY: {season!r}")
        if isinstance(gameweek, bool) or not isinstance(gameweek, int) or not 1 <= gameweek <= 38:
            raise EvaluationConfigError(f"gameweek must be an integer in [1, 38]: {gameweek!r}")
        return cls(int(match.group("start")), gameweek)


@dataclass(frozen=True, slots=True)
class EvaluationWindow:
    name: str
    role: WindowRole
    start_season: str
    start_gameweek: int
    end_season: str
    end_gameweek: int
    selection_allowed: bool = False
    calibration_window: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise EvaluationConfigError("window name must not be empty")
        if self.start > self.end:
            raise EvaluationConfigError(f"window {self.name!r} has inverted boundaries")
        expected_selection = self.role is WindowRole.DISCOVERY
        if self.selection_allowed != expected_selection:
            raise EvaluationConfigError(
                "only discovery windows may be marked selection_allowed; "
                f"invalid window {self.name!r}"
            )
        evaluated = self.role in {
            WindowRole.DISCOVERY,
            WindowRole.CONFIRMATION,
            WindowRole.PROSPECTIVE,
        }
        if evaluated != (self.calibration_window is not None):
            raise EvaluationConfigError(
                f"evaluated window {self.name!r} must name one calibration window"
            )

    @property
    def start(self) -> Period:
        return Period.from_values(self.start_season, self.start_gameweek)

    @property
    def end(self) -> Period:
        return Period.from_values(self.end_season, self.end_gameweek)

    def contains(self, season: str, gameweek: int) -> bool:
        period = Period.from_values(season, gameweek)
        return self.start <= period <= self.end


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    version: str
    key_columns: tuple[str, ...]
    windows: tuple[EvaluationWindow, ...]

    def __post_init__(self) -> None:
        if not self.version or not self.windows:
            raise EvaluationConfigError("evaluation plan requires a version and windows")
        canonical_keys = {"season", "gameweek", "player_key"}
        if not canonical_keys.issubset(self.key_columns) or len(self.key_columns) != len(
            set(self.key_columns)
        ):
            raise EvaluationConfigError(
                "evaluation key columns must uniquely include season, gameweek, and player_key"
            )
        names = [window.name for window in self.windows]
        if len(names) != len(set(names)):
            raise EvaluationConfigError("evaluation window names must be unique")
        ordered = sorted(self.windows, key=lambda window: window.start)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.end >= current.start:
                raise EvaluationConfigError(
                    f"evaluation windows overlap: {previous.name!r}, {current.name!r}"
                )
        by_name = {window.name: window for window in self.windows}
        for window in self.evaluation_windows:
            calibration = by_name.get(window.calibration_window)
            if calibration is None:
                raise EvaluationConfigError(
                    f"window {window.name!r} references an unknown calibration window"
                )
            if calibration.role is not WindowRole.CALIBRATION:
                raise EvaluationConfigError(
                    f"window {window.name!r} must reference a role=calibration window"
                )
            if calibration.end >= window.start:
                raise EvaluationConfigError(
                    f"calibration for {window.name!r} must end before its test window"
                )
            if _next_period(calibration.end) != window.start:
                raise EvaluationConfigError(
                    f"calibration for {window.name!r} must be the immediately preceding block"
                )

    @property
    def evaluation_windows(self) -> tuple[EvaluationWindow, ...]:
        roles = {WindowRole.DISCOVERY, WindowRole.CONFIRMATION, WindowRole.PROSPECTIVE}
        return tuple(window for window in self.windows if window.role in roles)

    @property
    def selection_windows(self) -> tuple[EvaluationWindow, ...]:
        return tuple(window for window in self.windows if window.selection_allowed)

    def window(self, name: str) -> EvaluationWindow:
        try:
            return next(window for window in self.windows if window.name == name)
        except StopIteration as error:
            raise EvaluationConfigError(f"unknown evaluation window {name!r}") from error

    def assert_selection_allowed(self, window_names: Iterable[str]) -> None:
        forbidden = [name for name in window_names if not self.window(name).selection_allowed]
        if forbidden:
            raise EvaluationConfigError(
                "confirmation/prospective/calibration windows cannot be used for selection: "
                f"{sorted(forbidden)!r}"
            )


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    name: str
    role: WindowRole
    train_indices: tuple[object, ...]
    calibration_indices: tuple[object, ...]
    test_indices: tuple[object, ...]

    def frames(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            frame.loc[list(self.train_indices)].copy(),
            frame.loc[list(self.calibration_indices)].copy(),
            frame.loc[list(self.test_indices)].copy(),
        )


def load_evaluation_plan(path: str | Path = DEFAULT_EVALUATION_CONFIG) -> EvaluationPlan:
    with Path(path).open("rb") as config_file:
        payload = tomllib.load(config_file)
    windows = tuple(
        EvaluationWindow(
            name=item["name"],
            role=WindowRole(item["role"]),
            start_season=item["start_season"],
            start_gameweek=item["start_gameweek"],
            end_season=item["end_season"],
            end_gameweek=item["end_gameweek"],
            selection_allowed=item.get("selection_allowed", False),
            calibration_window=item.get("calibration_window"),
        )
        for item in payload["windows"]
    )
    return EvaluationPlan(
        version=payload["version"],
        key_columns=tuple(payload.get("key_columns", ("season", "gameweek", "player_key"))),
        windows=windows,
    )


class ExpandingWindowSplitter:
    """Create out-of-time folds while physically excluding all future rows."""

    def __init__(self, plan: EvaluationPlan) -> None:
        self.plan = plan

    def split(
        self,
        frame: pd.DataFrame,
        *,
        window_names: Sequence[str] | None = None,
        require_non_empty: bool = True,
    ) -> Iterator[WalkForwardFold]:
        required = {*self.plan.key_columns, "deadline_utc"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise LeakageError(f"evaluation frame is missing required columns {missing!r}")
        if not frame.index.is_unique:
            raise LeakageError("evaluation frame index must be unique")
        if frame[list(self.plan.key_columns)].duplicated().any():
            raise LeakageError("evaluation frame contains duplicate canonical row keys")

        periods = [
            Period.from_values(season, int(gameweek))
            for season, gameweek in frame[["season", "gameweek"]].itertuples(index=False, name=None)
        ]
        selected = (
            self.plan.evaluation_windows
            if window_names is None
            else tuple(self.plan.window(name) for name in window_names)
        )
        invalid_roles = [
            window.name
            for window in selected
            if window.role
            not in {
                WindowRole.DISCOVERY,
                WindowRole.CONFIRMATION,
                WindowRole.PROSPECTIVE,
            }
        ]
        if invalid_roles:
            raise EvaluationConfigError(f"cannot evaluate non-test windows {invalid_roles!r}")

        for test_window in selected:
            calibration = self.plan.window(test_window.calibration_window or "")
            training_windows = tuple(
                window for window in self.plan.windows if window.end < calibration.start
            )
            train_mask = [
                any(window.start <= period <= window.end for window in training_windows)
                for period in periods
            ]
            calibration_mask = [
                calibration.start <= period <= calibration.end for period in periods
            ]
            test_mask = [test_window.start <= period <= test_window.end for period in periods]
            train = frame.loc[train_mask]
            calibration_frame = frame.loc[calibration_mask]
            test = frame.loc[test_mask]
            if train.empty or calibration_frame.empty or test.empty:
                if require_non_empty:
                    raise LeakageError(
                        f"fold {test_window.name!r} has empty train/calibration/test partition: "
                        f"{len(train)}/{len(calibration_frame)}/{len(test)}"
                    )
                continue

            assert_chronological_fold(train, calibration_frame, key_columns=self.plan.key_columns)
            assert_chronological_fold(train, test, key_columns=self.plan.key_columns)
            assert_chronological_fold(calibration_frame, test, key_columns=self.plan.key_columns)
            yield WalkForwardFold(
                name=test_window.name,
                role=test_window.role,
                train_indices=tuple(train.index),
                calibration_indices=tuple(calibration_frame.index),
                test_indices=tuple(test.index),
            )


def _next_period(period: Period) -> Period:
    if period.gameweek < 38:
        return Period(period.season_start, period.gameweek + 1)
    return Period(period.season_start + 1, 1)
