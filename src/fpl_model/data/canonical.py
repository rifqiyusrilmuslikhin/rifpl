"""Causal construction of fixture facts and canonical player-gameweek rows.

The module deliberately freezes schedule/snapshot context before outcomes are attached.  A target
GW therefore has one deadline anchor even when it contains multiple fixtures.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from fpl_model.data.identity import IdentityLookupError, PlayerIdentityRegistry
from fpl_model.data.schemas import (
    DEADLINE_SNAPSHOT_SCHEMA,
    PLAYER_FIXTURE_FACT_SCHEMA,
    PLAYER_GAMEWEEK_MODEL_SCHEMA,
    SchemaValidationError,
)


class CanonicalDatasetError(ValueError):
    """Raised when canonical construction would be ambiguous or non-causal."""


class DGWAnchorError(CanonicalDatasetError):
    """Raised when fixtures in one player-DGW do not share one deadline anchor."""


class ReconciliationError(CanonicalDatasetError):
    """Raised when fixture facts and player-gameweek totals do not reconcile."""


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Machine-readable proof that fixture outcomes aggregate without loss or duplication."""

    fixture_fact_rows: int
    scheduled_player_fixture_rows: int
    fixture_fact_player_gameweeks: int
    player_gameweek_rows: int
    dgw_player_gameweek_rows: int
    blank_player_gameweek_rows: int
    total_fixture_points: int
    total_gameweek_points: int
    total_fixture_minutes: int
    total_gameweek_minutes: int
    duplicate_fixture_keys: int
    duplicate_scheduled_player_fixture_keys: int
    duplicate_player_gameweek_keys: int
    status: str = "passed"

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        """Write a deterministic reconciliation artifact and return its resolved path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination.resolve()


@dataclass(frozen=True, slots=True)
class CanonicalDataset:
    fixture_facts: tuple[dict[str, Any], ...]
    fixture_contexts: tuple[dict[str, Any], ...]
    player_gameweeks: tuple[dict[str, Any], ...]
    reconciliation: ReconciliationReport


class CanonicalDatasetBuilder:
    """Build canonical records using an interval-versioned player registry."""

    def __init__(self, registry: PlayerIdentityRegistry) -> None:
        self.registry = registry

    def build_player_fixture_facts(
        self,
        outcomes: Iterable[Mapping[str, Any]],
        fixtures: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize raw outcomes and join team/opponent at the fixture's gameweek."""
        fixture_index = _fixture_index(fixtures)
        facts: list[dict[str, Any]] = []
        for default_row_number, outcome in enumerate(outcomes, start=1):
            season = _string(outcome, "season")
            fixture_id = _integer(outcome, "fixture_id", "fixture")
            fixture = _lookup_fixture(fixture_index, season, fixture_id)
            gameweek = fixture["gameweek"]
            _agree_if_present(outcome, gameweek, "gameweek", "event", "round")

            player_key = self._resolve_player(outcome, season, gameweek)
            try:
                identity = self.registry.identity_at(
                    player_key=player_key,
                    season=season,
                    gameweek=gameweek,
                )
            except IdentityLookupError as error:
                raise CanonicalDatasetError(str(error)) from error

            _agree_if_present(outcome, identity.fpl_code, "fpl_code", "code")
            _agree_if_present(
                outcome,
                identity.fpl_element_id,
                "fpl_element_id",
                "element",
                "element_id",
            )
            team_id = identity.team_id
            if team_id == fixture["home_team_id"]:
                was_home = True
                opponent_team_id = fixture["away_team_id"]
            elif team_id == fixture["away_team_id"]:
                was_home = False
                opponent_team_id = fixture["home_team_id"]
            else:
                raise CanonicalDatasetError(
                    f"player {player_key} is assigned to team {team_id} at {season} GW "
                    f"{gameweek}, which is not in fixture {fixture_id}"
                )
            _agree_if_present(outcome, team_id, "team_id", "team")
            _agree_if_present(outcome, opponent_team_id, "opponent_team_id", "opponent_team")
            _agree_if_present(outcome, was_home, "was_home")
            _agree_datetime_if_present(
                outcome,
                fixture["kickoff_utc"],
                "kickoff_utc",
                "kickoff_time",
            )

            source_row_number = outcome.get("source_row_number", default_row_number)
            source_row_number = _coerce_integer(source_row_number, "source_row_number")
            if source_row_number <= 0:
                raise CanonicalDatasetError("source_row_number must be a positive integer")
            facts.append(
                {
                    "season": season,
                    "fixture_id": fixture_id,
                    "gameweek": gameweek,
                    "player_key": player_key,
                    "fpl_code": identity.fpl_code,
                    "fpl_element_id": identity.fpl_element_id,
                    "kickoff_utc": fixture["kickoff_utc"],
                    "team_id": team_id,
                    "opponent_team_id": opponent_team_id,
                    "was_home": was_home,
                    "total_points": _integer(outcome, "total_points", non_negative=False),
                    "minutes": _integer(outcome, "minutes"),
                    "source_artifact_id": _string(outcome, "source_artifact_id"),
                    "source_row_number": source_row_number,
                }
            )
        try:
            PLAYER_FIXTURE_FACT_SCHEMA.validate_records(facts)
        except SchemaValidationError as error:
            raise CanonicalDatasetError(str(error)) from error
        return sorted(facts, key=_fact_sort_key)

    def build_predeadline_fixture_context(
        self,
        snapshots: Iterable[Mapping[str, Any]],
        fixtures: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create player-fixture context using only information strictly before the deadline."""
        snapshot_rows = [dict(row) for row in snapshots]
        try:
            DEADLINE_SNAPSHOT_SCHEMA.validate_records(snapshot_rows)
        except SchemaValidationError as error:
            raise CanonicalDatasetError(str(error)) from error
        fixture_rows = list(_fixture_index(fixtures).values())
        fixtures_by_gameweek: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for fixture in fixture_rows:
            fixtures_by_gameweek[(fixture["season"], fixture["gameweek"])].append(fixture)

        contexts: list[dict[str, Any]] = []
        for snapshot in snapshot_rows:
            season = snapshot["season"]
            gameweek = snapshot["gameweek"]
            player_key = snapshot["player_key"]
            try:
                identity = self.registry.identity_at(
                    player_key=player_key,
                    season=season,
                    gameweek=gameweek,
                )
            except IdentityLookupError as error:
                raise CanonicalDatasetError(str(error)) from error
            identity_fields = {
                "team_id": identity.team_id,
                "fpl_code": identity.fpl_code,
                "fpl_element_id": identity.fpl_element_id,
            }
            for field, expected in identity_fields.items():
                if snapshot[field] != expected:
                    raise CanonicalDatasetError(
                        f"snapshot {field} {snapshot[field]!r} disagrees with time-valid "
                        f"identity value {expected!r} for {player_key}, {season} GW {gameweek}"
                    )
            team_id = identity.team_id

            player_fixtures = [
                row
                for row in fixtures_by_gameweek.get((season, gameweek), [])
                if team_id in {row["home_team_id"], row["away_team_id"]}
            ]
            for fixture in player_fixtures:
                if fixture["available_at_utc"] >= snapshot["deadline_utc"]:
                    raise CanonicalDatasetError(
                        f"fixture {fixture['fixture_id']} schedule was not available before "
                        f"{season} GW {gameweek} deadline"
                    )
                if fixture["kickoff_utc"] <= snapshot["deadline_utc"]:
                    raise CanonicalDatasetError(
                        f"fixture {fixture['fixture_id']} kickoff must be after its GW deadline"
                    )

            anchor = max(
                [snapshot["captured_at_utc"]]
                + [fixture["available_at_utc"] for fixture in player_fixtures]
            )
            anchor_id = f"{season}:gw{gameweek}:{player_key}"
            for fixture in sorted(player_fixtures, key=_fixture_sort_key):
                was_home = team_id == fixture["home_team_id"]
                opponent_team_id = fixture["away_team_id"] if was_home else fixture["home_team_id"]
                difficulty = fixture["home_difficulty"] if was_home else fixture["away_difficulty"]
                contexts.append(
                    {
                        "season": season,
                        "gameweek": gameweek,
                        "player_key": player_key,
                        "fixture_id": fixture["fixture_id"],
                        "kickoff_utc": fixture["kickoff_utc"],
                        "deadline_utc": snapshot["deadline_utc"],
                        "snapshot_captured_at_utc": snapshot["captured_at_utc"],
                        "feature_cutoff_utc": anchor,
                        "context_anchor_id": anchor_id,
                        "team_id_at_deadline": team_id,
                        "opponent_team_id": opponent_team_id,
                        "was_home": was_home,
                        "fixture_difficulty": difficulty,
                        "snapshot_source_artifact_id": snapshot["source_artifact_id"],
                        "fixture_source_artifact_id": fixture["source_artifact_id"],
                    }
                )
        validate_dgw_anchors(contexts)
        return sorted(contexts, key=_context_sort_key)

    def build_player_gameweek_context(
        self,
        snapshots: Iterable[Mapping[str, Any]],
        fixtures: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Aggregate frozen fixture context to exactly one row per snapshot player-GW.

        Snapshot players are the decision universe.  Consequently players with no scheduled
        fixture remain explicit blank rows rather than disappearing in an inner join.
        """
        snapshot_rows = [dict(row) for row in snapshots]
        contexts = self.build_predeadline_fixture_context(snapshot_rows, fixtures)
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for context in contexts:
            grouped[_player_gameweek_key(context)].append(context)

        model_rows: list[dict[str, Any]] = []
        for snapshot in snapshot_rows:
            key = _player_gameweek_key(snapshot)
            entries = sorted(grouped.get(key, []), key=_context_sort_key)
            fixture_ids = [str(row["fixture_id"]) for row in entries]
            difficulties = [row["fixture_difficulty"] for row in entries]
            complete_difficulty = bool(difficulties) and all(
                value is not None for value in difficulties
            )
            source_ids = {snapshot["source_artifact_id"]}
            source_ids.update(row["fixture_source_artifact_id"] for row in entries)
            cutoff = entries[0]["feature_cutoff_utc"] if entries else snapshot["captured_at_utc"]
            row = {
                "season": snapshot["season"],
                "gameweek": snapshot["gameweek"],
                "player_key": snapshot["player_key"],
                "deadline_utc": snapshot["deadline_utc"],
                "snapshot_captured_at_utc": snapshot["captured_at_utc"],
                "feature_cutoff_utc": cutoff,
                "fixture_ids": fixture_ids,
                "team_id_at_deadline": snapshot["team_id"],
                "position_at_deadline": snapshot["position"],
                "fixture_count": len(entries),
                "is_blank": not entries,
                "source_artifact_ids": sorted(source_ids),
                "fpl_code": snapshot["fpl_code"],
                "fpl_element_id": snapshot["fpl_element_id"],
                "home_fixture_count": sum(row["was_home"] for row in entries),
                "opponent_team_ids": [str(row["opponent_team_id"]) for row in entries],
                "fixture_difficulty_mean": (
                    sum(difficulties) / len(difficulties) if complete_difficulty else None
                ),
                "fixture_difficulty_min": min(difficulties) if complete_difficulty else None,
                "fixture_difficulty_available": complete_difficulty,
                "now_cost": snapshot["now_cost"],
                "status": snapshot["status"],
                "status_value_state": snapshot["status_value_state"],
                "chance_of_playing_next_round": snapshot["chance_of_playing_next_round"],
                "chance_of_playing_value_state": snapshot["chance_of_playing_value_state"],
                "ep_next": snapshot["ep_next"],
                "ep_next_value_state": snapshot["ep_next_value_state"],
            }
            model_rows.append(row)
        try:
            PLAYER_GAMEWEEK_MODEL_SCHEMA.validate_records(model_rows)
        except SchemaValidationError as error:
            raise CanonicalDatasetError(str(error)) from error
        return sorted(model_rows, key=_model_sort_key)

    def build_gameweek_targets(
        self, fixture_facts: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Sum post-GW fixture outcomes to one target record per player-gameweek."""
        facts = [dict(row) for row in fixture_facts]
        try:
            PLAYER_FIXTURE_FACT_SCHEMA.validate_records(facts)
        except SchemaValidationError as error:
            raise CanonicalDatasetError(str(error)) from error
        grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for fact in facts:
            grouped[_player_gameweek_key(fact)].append(fact)

        targets: list[dict[str, Any]] = []
        for (season, gameweek, player_key), entries in grouped.items():
            points = sum(row["total_points"] for row in entries)
            minutes = sum(row["minutes"] for row in entries)
            targets.append(
                {
                    "season": season,
                    "gameweek": gameweek,
                    "player_key": player_key,
                    "fixture_ids": [
                        str(row["fixture_id"]) for row in sorted(entries, key=_fact_sort_key)
                    ],
                    "target_source_artifact_ids": sorted(
                        {row["source_artifact_id"] for row in entries}
                    ),
                    **_target_values(points, minutes),
                }
            )
        return sorted(targets, key=_model_sort_key)

    def attach_gameweek_targets(
        self,
        contexts: Iterable[Mapping[str, Any]],
        fixture_facts: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Join post-GW labels only after the context frame has been frozen."""
        context_rows = [dict(row) for row in contexts]
        facts = [dict(row) for row in fixture_facts]
        try:
            PLAYER_GAMEWEEK_MODEL_SCHEMA.validate_records(context_rows)
            PLAYER_FIXTURE_FACT_SCHEMA.validate_records(facts)
        except SchemaValidationError as error:
            raise CanonicalDatasetError(str(error)) from error

        context_index = {_player_gameweek_key(row): row for row in context_rows}
        if len(context_index) != len(context_rows):
            raise CanonicalDatasetError("duplicate canonical player-gameweek key")
        for fact in facts:
            key = _player_gameweek_key(fact)
            context = context_index.get(key)
            if context is None:
                raise ReconciliationError(f"fixture fact {key!r} has no snapshot decision row")
            if str(fact["fixture_id"]) not in context["fixture_ids"]:
                raise ReconciliationError(
                    f"fixture fact {fact['fixture_id']} is not scheduled for {key!r}"
                )

        targets = {_player_gameweek_key(row): row for row in self.build_gameweek_targets(facts)}

        output: list[dict[str, Any]] = []
        for key, context in context_index.items():
            target = targets.get(key)
            row = dict(context)
            if target is None:
                row.update(_target_values(0, 0))
                row["target_source_artifact_ids"] = []
            else:
                row.update(
                    {
                        field: value
                        for field, value in target.items()
                        if field.startswith("actual_") or field.startswith("y_")
                    }
                )
                row["target_source_artifact_ids"] = target["target_source_artifact_ids"]
            output.append(row)
        try:
            PLAYER_GAMEWEEK_MODEL_SCHEMA.validate_records(output)
        except SchemaValidationError as error:
            raise CanonicalDatasetError(str(error)) from error
        return sorted(output, key=_model_sort_key)

    def reconcile(
        self,
        fixture_facts: Iterable[Mapping[str, Any]],
        fixture_contexts: Iterable[Mapping[str, Any]],
        player_gameweeks: Iterable[Mapping[str, Any]],
    ) -> ReconciliationReport:
        facts = [dict(row) for row in fixture_facts]
        contexts = [dict(row) for row in fixture_contexts]
        models = [dict(row) for row in player_gameweeks]
        fact_keys = [(row["season"], row["fixture_id"], row["player_key"]) for row in facts]
        model_keys = [_player_gameweek_key(row) for row in models]
        duplicate_fact_keys = len(fact_keys) - len(set(fact_keys))
        duplicate_model_keys = len(model_keys) - len(set(model_keys))
        context_keys = [
            (row["season"], row["gameweek"], row["player_key"], row["fixture_id"])
            for row in contexts
        ]
        duplicate_context_keys = len(context_keys) - len(set(context_keys))
        fixture_points = sum(row["total_points"] for row in facts)
        fixture_minutes = sum(row["minutes"] for row in facts)
        gameweek_points = sum(row["actual_points_gw"] for row in models)
        gameweek_minutes = sum(row["actual_minutes_gw"] for row in models)

        problems: list[str] = []
        if duplicate_fact_keys:
            problems.append(f"{duplicate_fact_keys} duplicate fixture keys")
        if duplicate_model_keys:
            problems.append(f"{duplicate_model_keys} duplicate player-gameweek keys")
        if duplicate_context_keys:
            problems.append(f"{duplicate_context_keys} duplicate scheduled player-fixture keys")
        if fixture_points != gameweek_points:
            problems.append(f"points differ ({fixture_points} fixture vs {gameweek_points} GW)")
        if fixture_minutes != gameweek_minutes:
            problems.append(f"minutes differ ({fixture_minutes} fixture vs {gameweek_minutes} GW)")
        if sum(row["fixture_count"] for row in models) != len(contexts):
            problems.append("scheduled player-fixture rows do not match GW fixture counts")

        contexts_by_key: dict[tuple[str, int, str], set[str]] = defaultdict(set)
        for context in contexts:
            contexts_by_key[_player_gameweek_key(context)].add(str(context["fixture_id"]))
        for model in models:
            key = _player_gameweek_key(model)
            if set(model["fixture_ids"]) != contexts_by_key.get(key, set()):
                problems.append(f"scheduled fixture IDs do not reconcile for {key!r}")

        facts_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
        for fact in facts:
            facts_by_key[_player_gameweek_key(fact)].append(fact)
        model_by_key = {_player_gameweek_key(row): row for row in models}
        for key, entries in facts_by_key.items():
            model = model_by_key.get(key)
            if model is None:
                problems.append(f"fixture facts for {key!r} have no player-gameweek row")
                continue
            expected_points = sum(row["total_points"] for row in entries)
            expected_minutes = sum(row["minutes"] for row in entries)
            if model["actual_points_gw"] != expected_points:
                problems.append(f"player-gameweek points do not reconcile for {key!r}")
            if model["actual_minutes_gw"] != expected_minutes:
                problems.append(f"player-gameweek minutes do not reconcile for {key!r}")
        if problems:
            raise ReconciliationError("; ".join(problems))

        return ReconciliationReport(
            fixture_fact_rows=len(facts),
            scheduled_player_fixture_rows=len(contexts),
            fixture_fact_player_gameweeks=len({_player_gameweek_key(row) for row in facts}),
            player_gameweek_rows=len(models),
            dgw_player_gameweek_rows=sum(row["fixture_count"] > 1 for row in models),
            blank_player_gameweek_rows=sum(row["is_blank"] for row in models),
            total_fixture_points=fixture_points,
            total_gameweek_points=gameweek_points,
            total_fixture_minutes=fixture_minutes,
            total_gameweek_minutes=gameweek_minutes,
            duplicate_fixture_keys=duplicate_fact_keys,
            duplicate_scheduled_player_fixture_keys=duplicate_context_keys,
            duplicate_player_gameweek_keys=duplicate_model_keys,
        )

    def build(
        self,
        *,
        outcomes: Iterable[Mapping[str, Any]],
        fixtures: Iterable[Mapping[str, Any]],
        snapshots: Iterable[Mapping[str, Any]],
    ) -> CanonicalDataset:
        fixture_rows = list(fixtures)
        snapshot_rows = list(snapshots)
        facts = self.build_player_fixture_facts(outcomes, fixture_rows)
        fixture_contexts = self.build_predeadline_fixture_context(snapshot_rows, fixture_rows)
        contexts = self.build_player_gameweek_context(snapshot_rows, fixture_rows)
        player_gameweeks = self.attach_gameweek_targets(contexts, facts)
        reconciliation = self.reconcile(facts, fixture_contexts, player_gameweeks)
        return CanonicalDataset(
            fixture_facts=tuple(facts),
            fixture_contexts=tuple(fixture_contexts),
            player_gameweeks=tuple(player_gameweeks),
            reconciliation=reconciliation,
        )

    def _resolve_player(self, outcome: Mapping[str, Any], season: str, gameweek: int) -> str:
        supplied_key = outcome.get("player_key")
        element_id = _optional_integer(outcome, "fpl_element_id", "element", "element_id")
        if supplied_key is None and element_id is None:
            raise CanonicalDatasetError(
                "outcome must contain player_key or a season-specific FPL element ID"
            )
        alias_key: str | None = None
        if element_id is not None:
            try:
                alias_key = self.registry.resolve_fpl_alias(
                    season=season,
                    fpl_element_id=element_id,
                    gameweek=gameweek,
                )
            except IdentityLookupError as error:
                raise CanonicalDatasetError(str(error)) from error
        if supplied_key is not None and not isinstance(supplied_key, str):
            raise CanonicalDatasetError("player_key must be a string")
        if supplied_key is not None and alias_key is not None and supplied_key != alias_key:
            raise CanonicalDatasetError("player_key conflicts with the time-valid element alias")
        return supplied_key or alias_key  # type: ignore[return-value]


def validate_dgw_anchors(
    fixture_contexts: Iterable[Mapping[str, Any]],
    *,
    history_fields: Sequence[str] = (),
) -> None:
    """Fail if a player-GW's fixtures have different deadline/history anchors.

    ``history_fields`` supports later feature builders and is also a direct guard against the
    common error of sorting rolling history only by kickoff within a DGW.
    """
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    forbidden = {
        "total_points",
        "minutes",
        "actual_points_gw",
        "actual_minutes_gw",
        "y_points",
        "y_minutes",
    }
    for row in fixture_contexts:
        leaked = forbidden.intersection(row)
        if leaked:
            raise DGWAnchorError(f"fixture context contains outcome fields {sorted(leaked)!r}")
        grouped[_player_gameweek_key(row)].append(row)

    anchor_fields = (
        "deadline_utc",
        "snapshot_captured_at_utc",
        "feature_cutoff_utc",
        "context_anchor_id",
        "team_id_at_deadline",
        *history_fields,
    )
    for key, rows in grouped.items():
        for row in rows:
            if row["snapshot_captured_at_utc"] >= row["deadline_utc"]:
                raise DGWAnchorError(f"snapshot is not pre-deadline for {key!r}")
            if row["feature_cutoff_utc"] >= row["deadline_utc"]:
                raise DGWAnchorError(f"feature cutoff is not pre-deadline for {key!r}")
        if len(rows) < 2:
            continue
        for field in anchor_fields:
            values = {_hashable(row.get(field)) for row in rows}
            if len(values) != 1:
                raise DGWAnchorError(
                    f"DGW fixture context varies in {field!r} for {key!r}; "
                    "all fixtures must use one deadline anchor"
                )


def _fixture_index(fixtures: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for row in fixtures:
        season = _string(row, "season")
        fixture_id = _integer(row, "fixture_id", "id")
        gameweek = _integer(row, "gameweek", "event", "round")
        home_team_id = _integer(row, "home_team_id", "team_h")
        away_team_id = _integer(row, "away_team_id", "team_a")
        if home_team_id == away_team_id:
            raise CanonicalDatasetError(f"fixture {fixture_id} has the same home and away team")
        normalized = {
            "season": season,
            "fixture_id": fixture_id,
            "gameweek": gameweek,
            "kickoff_utc": _datetime(row, "kickoff_utc", "kickoff_time"),
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_difficulty": _optional_number(row, "home_difficulty", "team_h_difficulty"),
            "away_difficulty": _optional_number(row, "away_difficulty", "team_a_difficulty"),
            "available_at_utc": _datetime(
                row,
                "available_at_utc",
                "schedule_available_at_utc",
            ),
            "source_artifact_id": _string(row, "source_artifact_id"),
        }
        key = (season, fixture_id)
        if key in output:
            raise CanonicalDatasetError(f"duplicate fixture schedule key {key!r}")
        output[key] = normalized
    return output


def _lookup_fixture(
    fixture_index: Mapping[tuple[str, int], dict[str, Any]], season: str, fixture_id: int
) -> dict[str, Any]:
    try:
        return fixture_index[(season, fixture_id)]
    except KeyError as error:
        raise CanonicalDatasetError(
            f"no fixture schedule for season {season}, fixture {fixture_id}"
        ) from error


def _target_values(points: int, minutes: int) -> dict[str, int | None]:
    return {
        "actual_points_gw": points,
        "actual_minutes_gw": minutes,
        "y_play_any": int(minutes > 0),
        "y_minutes_60": int(minutes >= 60),
        "y_minutes": minutes,
        "y_points": points,
        "y_points_if_play": points if minutes > 0 else None,
        "y_haul_5": int(points >= 5),
        "y_haul_10": int(points >= 10),
    }


def _player_gameweek_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (row["season"], row["gameweek"], row["player_key"])


def _fact_sort_key(row: Mapping[str, Any]) -> tuple[str, int, datetime, int, str]:
    return (
        row["season"],
        row["gameweek"],
        row["kickoff_utc"],
        row["fixture_id"],
        row["player_key"],
    )


def _fixture_sort_key(row: Mapping[str, Any]) -> tuple[datetime, int]:
    return (row["kickoff_utc"], row["fixture_id"])


def _context_sort_key(row: Mapping[str, Any]) -> tuple[str, int, str, datetime, int]:
    return (*_player_gameweek_key(row), row["kickoff_utc"], row["fixture_id"])


def _model_sort_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return _player_gameweek_key(row)


def _string(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise CanonicalDatasetError(f"missing non-empty string field from {names!r}")


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _integer(row: Mapping[str, Any], *names: str, non_negative: bool = True) -> int:
    present = [(name, row[name]) for name in names if name in row]
    if not present:
        raise CanonicalDatasetError(f"missing field from {names!r}")
    values = {_coerce_integer(value, name) for name, value in present}
    if len(values) != 1:
        raise CanonicalDatasetError(f"fields {names!r} disagree")
    value = next(iter(values))
    if non_negative and value < 0:
        qualifier = "non-negative " if non_negative else ""
        raise CanonicalDatasetError(f"field {names[0]!r} must be a {qualifier}integer")
    return value


def _optional_integer(row: Mapping[str, Any], *names: str) -> int | None:
    present = [(name, row[name]) for name in names if name in row and row[name] is not None]
    if not present:
        return None
    values = {_coerce_integer(value, name) for name, value in present}
    if len(values) != 1:
        raise CanonicalDatasetError(f"fields {names!r} disagree")
    value = next(iter(values))
    if value <= 0:
        raise CanonicalDatasetError(f"field {present[0][0]!r} must be a positive integer")
    return value


def _optional_number(row: Mapping[str, Any], *names: str) -> float | int | None:
    present = [(name, row[name]) for name in names if name in row and row[name] is not None]
    if not present:
        return None
    values = {_coerce_number(value, name) for name, value in present}
    if len(values) != 1:
        raise CanonicalDatasetError(f"fields {names!r} disagree")
    return next(iter(values))


def _datetime(row: Mapping[str, Any], *names: str) -> datetime:
    present = [(name, row[name]) for name in names if name in row]
    if not present:
        raise CanonicalDatasetError(f"missing field from {names!r}")
    parsed = [_as_utc_datetime(value, name) for name, value in present]
    if len(set(parsed)) != 1:
        raise CanonicalDatasetError(f"fields {names!r} disagree")
    return parsed[0]


def _as_utc_datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise CanonicalDatasetError(
                f"field {field_name!r} must be a valid ISO-8601 timestamp"
            ) from error
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset().total_seconds() != 0
    ):
        raise CanonicalDatasetError(f"field {field_name!r} must be a timezone-aware UTC datetime")
    return value


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    present = [(name, row[name]) for name in names if name in row]
    if not present:
        raise CanonicalDatasetError(f"missing field from {tuple(names)!r}")
    values = {_hashable(value) for _, value in present}
    if len(values) != 1:
        raise CanonicalDatasetError(f"fields {tuple(names)!r} disagree")
    return present[0][1]


def _agree_if_present(row: Mapping[str, Any], expected: Any, *names: str) -> None:
    for name in names:
        if name not in row:
            continue
        supplied = row[name]
        if isinstance(expected, bool):
            supplied = _coerce_boolean(supplied, name)
        elif _is_integer(expected):
            supplied = _coerce_integer(supplied, name)
        if supplied != expected:
            raise CanonicalDatasetError(
                f"field {name!r} value {row[name]!r} disagrees with canonical {expected!r}"
            )


def _agree_datetime_if_present(row: Mapping[str, Any], expected: datetime, *names: str) -> None:
    for name in names:
        if name in row and _as_utc_datetime(row[name], name) != expected:
            raise CanonicalDatasetError(
                f"field {name!r} value {row[name]!r} disagrees with canonical {expected!r}"
            )


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


def _coerce_integer(value: Any, field_name: str) -> int:
    if _is_integer(value):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = int(stripped)
        except ValueError:
            pass
        else:
            if stripped and str(parsed) == stripped.lstrip("+"):
                return parsed
    raise CanonicalDatasetError(f"field {field_name!r} must be an integer")


def _coerce_number(value: Any, field_name: str) -> float | int:
    if isinstance(value, Real) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    raise CanonicalDatasetError(f"field {field_name!r} must be numeric")


def _coerce_boolean(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, "0", "False", "false"):
        return False
    if value in (1, "1", "True", "true"):
        return True
    raise CanonicalDatasetError(f"field {field_name!r} must be boolean")
