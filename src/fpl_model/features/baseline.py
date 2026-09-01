"""Causal construction of the fixed 46-feature Sprint 4 baseline frame."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from numbers import Integral, Real
from typing import Any

import pandas as pd

from fpl_model.features.contract import BASELINE_FEATURE_CONTRACT, BASELINE_FEATURE_NAMES

KEY_COLUMNS = ("season", "gameweek", "player_key")
PROVENANCE_COLUMNS = (
    "deadline_utc",
    "snapshot_captured_at_utc",
    "feature_cutoff_utc",
    "feature_contract_version",
    "feature_source_artifact_ids",
)
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

_STATUS_RISK = {"a": 0.0, "d": 1.0, "i": 2.0, "s": 2.0, "u": 2.0, "n": 2.0}


class FeatureBuildError(ValueError):
    """Raised when input data cannot produce an unambiguous causal feature frame."""


class BaselineFeatureBuilder:
    """Build one deadline-anchored row with exactly 46 named features per player-GW.

    Player windows operate on prior player-fixture observations. Zero-minute observations remain
    in those windows so participation rates retain their meaning; ``history_appearances_count`` and
    ``minutes_last_appearance`` explicitly use positive-minute appearances. Team windows operate on
    prior completed EPL fixtures. All histories reset at the season boundary.
    """

    def __init__(self) -> None:
        self.contract = BASELINE_FEATURE_CONTRACT

    def build(
        self,
        player_gameweeks: Iterable[Mapping[str, Any]],
        *,
        fixture_contexts: Iterable[Mapping[str, Any]],
        player_history: Iterable[Mapping[str, Any]],
        team_history: Iterable[Mapping[str, Any]],
    ) -> pd.DataFrame:
        """Return a deterministic DataFrame containing keys, provenance, and 46 features.

        Inputs may include future rows: they are filtered independently for every deadline using
        completion and availability timestamps. Rows at the deadline are never eligible.
        """
        targets = [dict(row) for row in player_gameweeks]
        contexts = [dict(row) for row in fixture_contexts]
        players = [
            _normalize_player_history(row, index) for index, row in enumerate(player_history)
        ]
        teams = _normalize_team_history(team_history)
        _unique(targets, KEY_COLUMNS, "player_gameweeks")
        _unique(contexts, (*KEY_COLUMNS, "fixture_id"), "fixture_contexts")
        _unique(players, ("season", "fixture_id", "player_key"), "player_history")
        _unique(teams, ("season", "fixture_id", "team_id"), "team_history")

        contexts_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for context in contexts:
            contexts_by_key[_key(context)].append(context)
        players_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for history in players:
            players_by_key[(history["season"], history["player_key"])].append(history)
        teams_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for history in teams:
            teams_by_key[(history["season"], history["team_id"])].append(history)

        records: list[dict[str, Any]] = []
        for target in sorted(targets, key=lambda row: (_key(row))):
            records.append(
                self._build_one(
                    target,
                    contexts_by_key.get(_key(target), []),
                    players_by_key.get(
                        (str(target.get("season")), str(target.get("player_key"))), []
                    ),
                    teams_by_key,
                )
            )
        columns = [*KEY_COLUMNS, *PROVENANCE_COLUMNS, *BASELINE_FEATURE_NAMES]
        frame = pd.DataFrame.from_records(records, columns=columns)
        for definition in self.contract.features:
            if definition.dtype == "category":
                frame[definition.name] = frame[definition.name].astype("category")
            else:
                frame[definition.name] = frame[definition.name].astype(definition.dtype)
        if tuple(name for name in frame.columns if name in BASELINE_FEATURE_NAMES) != (
            BASELINE_FEATURE_NAMES
        ):
            raise FeatureBuildError("feature frame does not match the fixed 46-feature contract")
        return frame

    def _build_one(
        self,
        target: dict[str, Any],
        contexts: list[dict[str, Any]],
        player_rows: list[dict[str, Any]],
        teams_by_key: Mapping[tuple[str, int], list[dict[str, Any]]],
    ) -> dict[str, Any]:
        _require_fields(
            target,
            (
                *KEY_COLUMNS,
                "deadline_utc",
                "snapshot_captured_at_utc",
                "feature_cutoff_utc",
                "team_id_at_deadline",
                "position_at_deadline",
                "fixture_count",
                "home_fixture_count",
                "now_cost",
            ),
            "player_gameweek",
        )
        deadline = _utc_datetime(target["deadline_utc"], "deadline_utc")
        snapshot_captured = _utc_datetime(
            target["snapshot_captured_at_utc"], "snapshot_captured_at_utc"
        )
        initial_cutoff = _utc_datetime(target["feature_cutoff_utc"], "feature_cutoff_utc")
        if snapshot_captured >= deadline:
            raise FeatureBuildError("snapshot_captured_at_utc must be strictly before deadline_utc")
        if initial_cutoff >= deadline:
            raise FeatureBuildError("feature_cutoff_utc must be strictly before deadline_utc")
        if initial_cutoff < snapshot_captured:
            raise FeatureBuildError("feature_cutoff_utc cannot precede snapshot_captured_at_utc")
        if TARGET_COLUMNS.intersection(BASELINE_FEATURE_NAMES):  # defensive invariant
            raise FeatureBuildError("target columns must not occur in the feature contract")

        ordered_contexts = sorted(contexts, key=_context_order)
        fixture_count = _integer(target["fixture_count"], "fixture_count", minimum=0)
        if len(ordered_contexts) != fixture_count:
            raise FeatureBuildError(
                f"{_key(target)!r} fixture_count={fixture_count} but has "
                f"{len(ordered_contexts)} fixture contexts"
            )
        for context in ordered_contexts:
            if _key(context) != _key(target):
                raise FeatureBuildError("fixture context key disagrees with player-gameweek key")
            context_deadline = _utc_datetime(context["deadline_utc"], "context deadline_utc")
            context_cutoff = _utc_datetime(
                context["feature_cutoff_utc"], "context feature_cutoff_utc"
            )
            if context_deadline != deadline or context_cutoff >= deadline:
                raise FeatureBuildError("fixture context is not anchored strictly pre-deadline")
            if context_cutoff != initial_cutoff:
                raise FeatureBuildError(
                    "all fixtures in a player-GW must share the player-gameweek deadline anchor"
                )
        anchor_ids = {
            context["context_anchor_id"]
            for context in ordered_contexts
            if context.get("context_anchor_id") is not None
        }
        if len(anchor_ids) > 1:
            raise FeatureBuildError("all fixtures in a player-DGW must share one context_anchor_id")
        expected_home_count = _integer(
            target["home_fixture_count"], "home_fixture_count", minimum=0
        )
        if sum(bool(context["was_home"]) for context in ordered_contexts) != expected_home_count:
            raise FeatureBuildError("home_fixture_count disagrees with fixture contexts")

        eligible_players = [
            row
            for row in player_rows
            if _history_is_available(row, deadline) and row["season"] == target["season"]
        ]
        eligible_players.sort(key=_history_order)
        team_id = _integer(target["team_id_at_deadline"], "team_id_at_deadline", minimum=1)
        eligible_team = _eligible_team_history(
            teams_by_key.get((str(target["season"]), team_id), []), deadline
        )
        opponent_windows: list[list[dict[str, Any]]] = []
        for context in ordered_contexts:
            opponent_id = _integer(context["opponent_team_id"], "opponent_team_id", minimum=1)
            opponent_windows.append(
                _eligible_team_history(
                    teams_by_key.get((str(target["season"]), opponent_id), []), deadline
                )[-5:]
            )

        feature_values: dict[str, Any] = {}
        feature_values.update(
            self._context_features(target, ordered_contexts, eligible_players, eligible_team)
        )
        feature_values.update(self._player_features(target, ordered_contexts, eligible_players))
        team_features = _team_features(eligible_team[-5:], prefix="team")
        opponent_features = _opponent_features(opponent_windows)
        feature_values.update(team_features)
        feature_values.update(opponent_features)
        feature_values["attack_matchup"] = _multiply_nullable(
            feature_values["team_goals_for_mean_last5"],
            feature_values["opp_goals_against_mean_last5"],
        )
        feature_values["clean_sheet_matchup"] = _clean_sheet_matchup(
            feature_values["team_clean_sheet_rate_last5"],
            feature_values["opp_goals_for_mean_last5"],
        )
        if tuple(feature_values) != BASELINE_FEATURE_NAMES:
            missing = set(BASELINE_FEATURE_NAMES).difference(feature_values)
            extras = set(feature_values).difference(BASELINE_FEATURE_NAMES)
            raise FeatureBuildError(
                f"feature implementation mismatch; missing={missing}, extras={extras}"
            )

        used_rows = [*ordered_contexts, *eligible_players, *eligible_team]
        for window in opponent_windows:
            used_rows.extend(window)
        source_ids = _source_ids(target, used_rows)
        availability_times = [initial_cutoff]
        for row in used_rows:
            value = row.get("available_at_utc")
            if value is not None:
                availability_times.append(_utc_datetime(value, "available_at_utc"))
        final_cutoff = max(availability_times)
        if final_cutoff >= deadline:
            raise FeatureBuildError(
                "a feature source is not available strictly before the deadline"
            )
        return {
            "season": str(target["season"]),
            "gameweek": _integer(target["gameweek"], "gameweek", minimum=1),
            "player_key": str(target["player_key"]),
            "deadline_utc": deadline,
            "snapshot_captured_at_utc": snapshot_captured,
            "feature_cutoff_utc": final_cutoff,
            "feature_contract_version": self.contract.contract_version,
            "feature_source_artifact_ids": source_ids,
            **feature_values,
        }

    def _context_features(
        self,
        target: Mapping[str, Any],
        contexts: Sequence[Mapping[str, Any]],
        player_rows: Sequence[Mapping[str, Any]],
        team_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        first_kickoff = (
            min(_utc_datetime(row["kickoff_utc"], "kickoff_utc") for row in contexts)
            if contexts
            else None
        )
        last_played = next(
            (row for row in reversed(player_rows) if _number(row["minutes"], "minutes") > 0),
            None,
        )
        last_team = team_rows[-1] if team_rows else None
        status_state = target.get("status_value_state")
        status = target.get("status")
        _validate_value_state(status, status_state, "status")
        status_risk = None
        if status_state not in {"source_unavailable", "acquisition_failure"} and status is not None:
            status_risk = _STATUS_RISK.get(str(status).casefold())
            if status_risk is None:
                raise FeatureBuildError(f"unknown FPL status code {status!r}")
        chance_state = target.get("chance_of_playing_value_state")
        chance = target.get("chance_of_playing_next_round")
        _validate_value_state(chance, chance_state, "chance_of_playing_next_round")
        chance_value = None
        if chance_state not in {"source_unavailable", "acquisition_failure"} and chance is not None:
            chance_value = float(_number(chance, "chance_of_playing_next_round"))
        gameweek = _integer(target["gameweek"], "gameweek", minimum=1)
        return {
            "position": str(target["position_at_deadline"]),
            "price": float(_number(target["now_cost"], "now_cost")) / 10.0,
            "fixture_count": _integer(target["fixture_count"], "fixture_count", minimum=0),
            "home_fixture_count": _integer(
                target["home_fixture_count"], "home_fixture_count", minimum=0
            ),
            "fixture_difficulty_mean": _nullable_number(target.get("fixture_difficulty_mean")),
            "fixture_difficulty_min": _nullable_number(target.get("fixture_difficulty_min")),
            "rest_days_before_first_fixture": _elapsed_days(last_team, first_kickoff),
            "days_since_last_epl_minutes": _elapsed_days(last_played, first_kickoff),
            "status_risk_ordinal": status_risk,
            "chance_of_playing": chance_value,
            "history_appearances_count": sum(
                _number(row["minutes"], "minutes") > 0 for row in player_rows
            ),
            "season_phase": min(max((gameweek - 1) / 37.0, 0.0), 1.0),
        }

    def _player_features(
        self,
        target: Mapping[str, Any],
        contexts: Sequence[Mapping[str, Any]],
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        last3 = list(history[-3:])
        last5 = list(history[-5:])
        last10 = list(history[-10:])
        appearances = [row for row in history if _number(row["minutes"], "minutes") > 0]
        understat_resolved = not (
            target.get("understat_resolved") is False
            or ("understat_id" in target and target.get("understat_id") is None)
        )
        relevant: list[float] = []
        for context in contexts:
            venue_rows = [
                row for row in history if bool(row["was_home"]) == bool(context["was_home"])
            ]
            value = _mean_field(venue_rows[-5:], "total_points")
            if value is not None:
                relevant.append(value)
        return {
            "minutes_last_appearance": (
                float(_number(appearances[-1]["minutes"], "minutes")) if appearances else None
            ),
            "minutes_mean_last3": _mean_field(last3, "minutes"),
            "minutes_mean_last5": _mean_field(last5, "minutes"),
            "minutes_std_last5": _population_std(last5, "minutes"),
            "play_any_rate_last5": _predicate_rate(last5, lambda value: value > 0),
            "minutes_60_rate_last5": _predicate_rate(last5, lambda value: value >= 60),
            "starts_rate_last5": _mean_field(last5, "starts"),
            "points_mean_last3": _mean_field(last3, "total_points"),
            "points_mean_last5": _mean_field(last5, "total_points"),
            "points_per90_last5": _per90(last5, "total_points"),
            "goals_per90_last5": _per90(last5, "goals_scored"),
            "assists_per90_last5": _per90(last5, "assists"),
            "xg_per90_last5": _per90(last5, "xg") if understat_resolved else None,
            "xa_per90_last5": _per90(last5, "xa") if understat_resolved else None,
            "shots_per90_last5": _per90(last5, "shots") if understat_resolved else None,
            "key_passes_per90_last5": (_per90(last5, "key_passes") if understat_resolved else None),
            "bps_per90_last5": _per90(last5, "bps"),
            "bonus_per90_last5": _per90(last5, "bonus"),
            "yellow_cards_per90_last10": _per90(last10, "yellow_cards"),
            "points_home_away_relevant_mean_last5": (
                sum(relevant) / len(relevant) if relevant else None
            ),
        }


def _normalize_player_history(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    value = dict(row)
    _require_fields(
        value,
        (
            "season",
            "fixture_id",
            "player_key",
            "kickoff_utc",
            "team_id",
            "was_home",
            "minutes",
            "total_points",
        ),
        f"player_history row {index + 1}",
    )
    value["kickoff_utc"] = _utc_datetime(value["kickoff_utc"], "kickoff_utc")
    value["season"] = str(value["season"])
    value["player_key"] = str(value["player_key"])
    value["fixture_id"] = _integer(value["fixture_id"], "fixture_id", minimum=1)
    value["team_id"] = _integer(value["team_id"], "team_id", minimum=1)
    value["was_home"] = _boolean(value["was_home"], "was_home")
    value["completed_at_utc"] = _completion_time(value)
    value["available_at_utc"] = _availability_time(value)
    aliases = {
        "goals_scored": ("goals_scored", "goals"),
        "assists": ("assists",),
        "starts": ("starts", "started"),
        "bps": ("bps",),
        "bonus": ("bonus",),
        "yellow_cards": ("yellow_cards",),
        "xg": ("xg", "xG"),
        "xa": ("xa", "xA"),
        "shots": ("shots",),
        "key_passes": ("key_passes", "keyPasses"),
    }
    for destination, names in aliases.items():
        value[destination] = _first_nullable_number(value, names)
    return value


def _normalize_team_history(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        row = dict(source)
        if "team_id" in row:
            _require_fields(
                row,
                ("season", "fixture_id", "kickoff_utc", "team_id", "goals_for", "goals_against"),
                f"team_history row {index}",
            )
            row["kickoff_utc"] = _utc_datetime(row["kickoff_utc"], "kickoff_utc")
            row["season"] = str(row["season"])
            row["fixture_id"] = _integer(row["fixture_id"], "fixture_id", minimum=1)
            row["team_id"] = _integer(row["team_id"], "team_id", minimum=1)
            row["completed_at_utc"] = _completion_time(row)
            row["available_at_utc"] = _availability_time(row)
            row["xg"] = _first_nullable_number(row, ("xg", "team_xg"))
            row["xga"] = _first_nullable_number(row, ("xga", "team_xga"))
            normalized.append(row)
            continue
        _require_fields(
            row,
            (
                "season",
                "fixture_id",
                "kickoff_utc",
                "home_team_id",
                "away_team_id",
                "home_goals",
                "away_goals",
            ),
            f"team_history row {index}",
        )
        kickoff = _utc_datetime(row["kickoff_utc"], "kickoff_utc")
        common = {
            "season": str(row["season"]),
            "fixture_id": _integer(row["fixture_id"], "fixture_id", minimum=1),
            "kickoff_utc": kickoff,
            "completed_at_utc": _completion_time({**row, "kickoff_utc": kickoff}),
            "available_at_utc": _availability_time({**row, "kickoff_utc": kickoff}),
            "source_artifact_id": row.get("source_artifact_id"),
        }
        home_goals = _number(row["home_goals"], "home_goals")
        away_goals = _number(row["away_goals"], "away_goals")
        home_xg = _first_nullable_number(row, ("home_xg", "h_xg"))
        away_xg = _first_nullable_number(row, ("away_xg", "a_xg"))
        normalized.extend(
            [
                {
                    **common,
                    "team_id": _integer(row["home_team_id"], "home_team_id", minimum=1),
                    "opponent_team_id": _integer(row["away_team_id"], "away_team_id", minimum=1),
                    "goals_for": home_goals,
                    "goals_against": away_goals,
                    "xg": home_xg,
                    "xga": away_xg,
                },
                {
                    **common,
                    "team_id": _integer(row["away_team_id"], "away_team_id", minimum=1),
                    "opponent_team_id": _integer(row["home_team_id"], "home_team_id", minimum=1),
                    "goals_for": away_goals,
                    "goals_against": home_goals,
                    "xg": away_xg,
                    "xga": home_xg,
                },
            ]
        )
    return normalized


def _team_features(rows: Sequence[Mapping[str, Any]], *, prefix: str) -> dict[str, Any]:
    if prefix != "team":
        raise ValueError("only team prefix is supported")
    return {
        "team_goals_for_mean_last5": _mean_field(rows, "goals_for"),
        "team_xg_mean_last5": _mean_field(rows, "xg"),
        "team_goals_against_mean_last5": _mean_field(rows, "goals_against"),
        "team_xga_mean_last5": _mean_field(rows, "xga"),
        "team_clean_sheet_rate_last5": _predicate_field_rate(
            rows, "goals_against", lambda value: value == 0
        ),
        "team_win_rate_last5": _paired_rate(
            rows, "goals_for", "goals_against", lambda left, right: left > right
        ),
    }


def _opponent_features(windows: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    mappings = (
        ("opp_goals_for_mean_last5", "goals_for", "mean"),
        ("opp_xg_mean_last5", "xg", "mean"),
        ("opp_goals_against_mean_last5", "goals_against", "mean"),
        ("opp_xga_mean_last5", "xga", "mean"),
        ("opp_clean_sheet_rate_last5", "goals_against", "zero_rate"),
        ("opp_loss_rate_last5", "result", "loss_rate"),
    )
    result: dict[str, Any] = {}
    for output, field, kind in mappings:
        fixture_values: list[float] = []
        complete = bool(windows)
        for rows in windows:
            if kind == "mean":
                value = _mean_field(rows, field)
            elif kind == "zero_rate":
                value = _predicate_field_rate(rows, field, lambda item: item == 0)
            else:
                value = _paired_rate(
                    rows, "goals_for", "goals_against", lambda left, right: left < right
                )
            if value is None:
                complete = False
                break
            fixture_values.append(value)
        result[output] = sum(fixture_values) / len(fixture_values) if complete else None
    return result


def _eligible_team_history(
    rows: Sequence[dict[str, Any]], deadline: datetime
) -> list[dict[str, Any]]:
    return sorted((row for row in rows if _history_is_available(row, deadline)), key=_history_order)


def _history_is_available(row: Mapping[str, Any], deadline: datetime) -> bool:
    completed = _utc_datetime(row["completed_at_utc"], "completed_at_utc")
    available = _utc_datetime(row["available_at_utc"], "available_at_utc")
    return completed < deadline and available < deadline


def _completion_time(row: Mapping[str, Any]) -> datetime:
    if row.get("completed_at_utc") is not None:
        return _utc_datetime(row["completed_at_utc"], "completed_at_utc")
    return _utc_datetime(row["kickoff_utc"], "kickoff_utc") + timedelta(hours=3)


def _availability_time(row: Mapping[str, Any]) -> datetime:
    if row.get("available_at_utc") is not None:
        return _utc_datetime(row["available_at_utc"], "available_at_utc")
    return _completion_time(row)


def _per90(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    values = [_nullable_number(row.get(field)) for row in rows]
    if any(value is None for value in values):
        return None
    minutes = sum(_number(row["minutes"], "minutes") for row in rows)
    if minutes < BASELINE_FEATURE_CONTRACT.per90_minimum_minutes:
        return None
    return (
        90.0
        * sum(value for value in values if value is not None)
        / (minutes + BASELINE_FEATURE_CONTRACT.per90_epsilon_minutes)
    )


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    values = [_nullable_number(row.get(field)) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None) / len(values)


def _population_std(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [_nullable_number(row.get(field)) for row in rows]
    if not values or any(value is None for value in values):
        return None
    numeric = [value for value in values if value is not None]
    mean = sum(numeric) / len(numeric)
    return math.sqrt(sum((value - mean) ** 2 for value in numeric) / len(numeric))


def _predicate_rate(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float | None:
    if not rows:
        return None
    return sum(predicate(_number(row["minutes"], "minutes")) for row in rows) / len(rows)


def _predicate_field_rate(
    rows: Sequence[Mapping[str, Any]], field: str, predicate: Any
) -> float | None:
    if not rows:
        return None
    values = [_nullable_number(row.get(field)) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(predicate(value) for value in values if value is not None) / len(values)


def _paired_rate(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, predicate: Any
) -> float | None:
    if not rows:
        return None
    pairs = [(_nullable_number(row.get(left)), _nullable_number(row.get(right))) for row in rows]
    if any(first is None or second is None for first, second in pairs):
        return None
    return sum(predicate(first, second) for first, second in pairs) / len(pairs)


def _elapsed_days(row: Mapping[str, Any] | None, target_kickoff: datetime | None) -> float | None:
    if row is None or target_kickoff is None:
        return None
    previous = _utc_datetime(row["kickoff_utc"], "kickoff_utc")
    return (target_kickoff - previous).total_seconds() / 86_400.0


def _source_ids(target: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values: set[str] = set()
    for item in target.get("source_artifact_ids", []):
        if isinstance(item, str) and item:
            values.add(item)
    for field in (
        "source_artifact_id",
        "snapshot_source_artifact_id",
        "fixture_source_artifact_id",
    ):
        item = target.get(field)
        if isinstance(item, str) and item:
            values.add(item)
    for row in rows:
        for field in (
            "source_artifact_id",
            "snapshot_source_artifact_id",
            "fixture_source_artifact_id",
        ):
            item = row.get(field)
            if isinstance(item, str) and item:
                values.add(item)
    return sorted(values)


def _multiply_nullable(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) * float(right)


def _clean_sheet_matchup(clean_sheet_rate: Any, opponent_goals: Any) -> float | None:
    if clean_sheet_rate is None or opponent_goals is None:
        return None
    return float(clean_sheet_rate) / (1.0 + float(opponent_goals))


def _first_nullable_number(row: Mapping[str, Any], fields: Sequence[str]) -> float | None:
    for field in fields:
        if field in row:
            return _nullable_number(row[field])
    return None


def _nullable_number(value: Any) -> float | None:
    if value is None or (isinstance(value, Real) and math.isnan(float(value))):
        return None
    return float(_number(value, "numeric value"))


def _validate_value_state(value: Any, state: Any, field: str) -> None:
    missing_states = {"source_unavailable", "acquisition_failure"}
    value_is_missing = value is None or (
        isinstance(value, Real) and not isinstance(value, bool) and math.isnan(float(value))
    )
    if state in missing_states:
        if not value_is_missing:
            raise FeatureBuildError(f"field {field!r} must be missing when state is {state!r}")
        return
    if state == "genuine_zero":
        if isinstance(value, bool) or not isinstance(value, Real) or float(value) != 0.0:
            raise FeatureBuildError(f"field {field!r} must be numeric zero for genuine_zero")
        return
    if state != "value":
        raise FeatureBuildError(f"field {field!r} has unknown value state {state!r}")
    if value_is_missing:
        raise FeatureBuildError(f"field {field!r} must not be missing when state is 'value'")
    if isinstance(value, Real) and not isinstance(value, bool) and float(value) == 0.0:
        raise FeatureBuildError(f"field {field!r} numeric zero must use genuine_zero")


def _number(value: Any, field: str) -> float:
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            pass
        else:
            if math.isfinite(parsed):
                return parsed
    raise FeatureBuildError(f"field {field!r} must be a finite number")


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum:
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            pass
        else:
            if str(parsed) == value.strip().lstrip("+") and parsed >= minimum:
                return parsed
    raise FeatureBuildError(f"field {field!r} must be an integer >= {minimum}")


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, "0", "false", "False"):
        return False
    if value in (1, "1", "true", "True"):
        return True
    raise FeatureBuildError(f"field {field!r} must be boolean")


def _utc_datetime(value: Any, field: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise FeatureBuildError(f"field {field!r} must be an ISO datetime") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise FeatureBuildError(f"field {field!r} must be timezone-aware")
    return value.astimezone(UTC)


def _require_fields(row: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise FeatureBuildError(f"{label} is missing fields {missing!r}")


def _unique(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], label: str) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        _require_fields(row, fields, label)
        key = tuple(row[field] for field in fields)
        if key in seen:
            raise FeatureBuildError(f"{label} has duplicate key {key!r}")
        seen.add(key)


def _key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["season"]),
        _integer(row["gameweek"], "gameweek", minimum=1),
        str(row["player_key"]),
    )


def _history_order(row: Mapping[str, Any]) -> tuple[datetime, int]:
    return (
        _utc_datetime(row["kickoff_utc"], "kickoff_utc"),
        _integer(row["fixture_id"], "fixture_id", minimum=1),
    )


def _context_order(row: Mapping[str, Any]) -> tuple[datetime, int]:
    return _history_order(row)
