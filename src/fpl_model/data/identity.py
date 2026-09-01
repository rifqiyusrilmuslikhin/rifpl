"""Cross-season FPL identity and audited external identity mappings."""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from numbers import Integral
from typing import Any

from fpl_model.data.schemas import PLAYER_IDENTITY_REGISTRY_SCHEMA

_PLAYER_KEY_NAMESPACE = uuid.UUID("48a85cc5-c436-4d92-b73d-ee37a8b95dac")


class IdentityConflictError(ValueError):
    """Identity evidence conflicts with an existing audited mapping."""


class IdentityLookupError(LookupError):
    """No identity version covers the requested season and gameweek."""


class MatchMethod(StrEnum):
    EXACT_ID = "exact_id"
    AUDITED_NAME_DOB = "audited_name_dob"
    MANUAL = "manual"
    UNRESOLVED = "unresolved"


class MatchConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class IdentityVersion:
    player_key: str
    fpl_code: int
    season: str
    fpl_element_id: int
    understat_id: str | None
    valid_from_gw: int
    valid_to_gw: int | None
    team_id: int
    match_method: str
    confidence: str
    audit_note: str | None
    source_artifact_id: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> IdentityVersion:
        return cls(**{field: record[field] for field in cls.__dataclass_fields__})

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UnderstatMatchProposal:
    """Name-based suggestions are evidence only and never mutate the registry."""

    player_key: str
    normalized_name: str
    candidate_ids: tuple[str, ...]
    status: str


class PlayerIdentityRegistry:
    """Versioned registry keyed globally by stable FPL ``code`` values."""

    def __init__(self, records: Iterable[IdentityVersion | Mapping[str, Any]] = ()) -> None:
        self._versions = [
            record if isinstance(record, IdentityVersion) else IdentityVersion.from_record(record)
            for record in records
        ]
        self._validate()

    @property
    def records(self) -> tuple[IdentityVersion, ...]:
        return tuple(
            sorted(
                self._versions,
                key=lambda row: (row.season, row.player_key, row.valid_from_gw),
            )
        )

    def to_records(self) -> list[dict[str, Any]]:
        records = [version.to_record() for version in self.records]
        PLAYER_IDENTITY_REGISTRY_SCHEMA.validate_records(records)
        return records

    def register_fpl_rows(
        self,
        *,
        season: str,
        gameweek: int,
        rows: Iterable[Mapping[str, Any]],
        source_artifact_id: str,
    ) -> list[dict[str, Any]]:
        """Attach exactly one player_key to each FPL row and version team changes atomically."""
        if not season or not source_artifact_id:
            raise ValueError("season and source_artifact_id must not be empty")
        _positive_int(gameweek, "gameweek")
        materialized = list(rows)
        staged = PlayerIdentityRegistry(self._versions)
        output: list[dict[str, Any]] = []
        seen_codes: set[int] = set()
        seen_elements: set[int] = set()

        for row_number, row in enumerate(materialized, start=1):
            if not isinstance(row, Mapping):
                raise IdentityConflictError(f"FPL row {row_number} must be an object")
            fpl_code = _source_int(row, "code", "fpl_code")
            element_id = _source_int(row, "id", "element", "fpl_element_id")
            team_id = _source_int(row, "team", "team_id")
            if fpl_code in seen_codes:
                raise IdentityConflictError(f"duplicate fpl_code {fpl_code} in one FPL payload")
            if element_id in seen_elements:
                raise IdentityConflictError(
                    f"duplicate element_id {element_id} in season {season} payload"
                )
            seen_codes.add(fpl_code)
            seen_elements.add(element_id)

            player_key = staged._register_one(
                season=season,
                gameweek=gameweek,
                fpl_code=fpl_code,
                element_id=element_id,
                team_id=team_id,
                source_artifact_id=source_artifact_id,
            )
            supplied_key = row.get("player_key")
            if supplied_key is not None and supplied_key != player_key:
                raise IdentityConflictError(
                    f"FPL row {row_number} carries conflicting player_key {supplied_key!r}"
                )
            resolved = dict(row)
            resolved["player_key"] = player_key
            output.append(resolved)

        staged._validate()
        self._versions = staged._versions
        return output

    def player_key_for_code(self, fpl_code: int) -> str:
        _positive_int(fpl_code, "fpl_code")
        keys = {row.player_key for row in self._versions if row.fpl_code == fpl_code}
        if not keys:
            raise IdentityLookupError(f"unknown fpl_code {fpl_code}")
        if len(keys) != 1:  # Defensive: schema validation normally makes this unreachable.
            raise IdentityConflictError(f"fpl_code {fpl_code} maps to multiple player keys")
        return next(iter(keys))

    def resolve_fpl_alias(self, *, season: str, fpl_element_id: int, gameweek: int) -> str:
        _positive_int(fpl_element_id, "fpl_element_id")
        _positive_int(gameweek, "gameweek")
        matches = [
            row
            for row in self._versions
            if row.season == season
            and row.fpl_element_id == fpl_element_id
            and _covers(row, gameweek)
        ]
        if len(matches) != 1:
            raise IdentityLookupError(
                f"expected one identity for season {season}, element {fpl_element_id}, "
                f"gameweek {gameweek}; found {len(matches)}"
            )
        return matches[0].player_key

    def team_at(self, *, player_key: str, season: str, gameweek: int) -> int:
        return self._version_at(player_key, season, gameweek).team_id

    def understat_id_at(self, *, player_key: str, season: str, gameweek: int) -> str | None:
        return self._version_at(player_key, season, gameweek).understat_id

    def assign_understat(
        self,
        *,
        player_key: str,
        season: str,
        valid_from_gw: int,
        understat_id: str,
        match_method: MatchMethod | str,
        confidence: MatchConfidence | str,
        audit_note: str | None,
        source_artifact_id: str,
    ) -> None:
        """Add an explicit, audited mapping; candidate matching never calls this method."""
        if not isinstance(understat_id, str) or not understat_id.strip():
            raise ValueError("understat_id must be a non-empty string")
        method = MatchMethod(match_method)
        match_confidence = MatchConfidence(confidence)
        if method is MatchMethod.UNRESOLVED or match_confidence is MatchConfidence.UNRESOLVED:
            raise IdentityConflictError("resolved Understat mappings cannot be unresolved")
        if method in {MatchMethod.MANUAL, MatchMethod.AUDITED_NAME_DOB} and not _has_note(
            audit_note
        ):
            raise IdentityConflictError(
                f"{method.value} Understat mappings require evidence in audit_note"
            )
        conflict_keys = {
            row.player_key
            for row in self._versions
            if row.understat_id == understat_id and row.player_key != player_key
        }
        if conflict_keys:
            raise IdentityConflictError(
                f"Understat ID {understat_id!r} is already assigned to another player_key"
            )
        self._change_mapping(
            player_key=player_key,
            season=season,
            gameweek=valid_from_gw,
            understat_id=understat_id,
            match_method=method.value,
            confidence=match_confidence.value,
            audit_note=audit_note,
            source_artifact_id=source_artifact_id,
        )

    def mark_understat_unresolved(
        self,
        *,
        player_key: str,
        season: str,
        valid_from_gw: int,
        audit_note: str,
        source_artifact_id: str,
    ) -> None:
        """Version an explicit unresolved decision without inventing an external identity."""
        if not _has_note(audit_note):
            raise ValueError("unresolved decisions require an audit_note")
        self._change_mapping(
            player_key=player_key,
            season=season,
            gameweek=valid_from_gw,
            understat_id=None,
            match_method=MatchMethod.UNRESOLVED.value,
            confidence=MatchConfidence.UNRESOLVED.value,
            audit_note=audit_note,
            source_artifact_id=source_artifact_id,
        )

    def _register_one(
        self,
        *,
        season: str,
        gameweek: int,
        fpl_code: int,
        element_id: int,
        team_id: int,
        source_artifact_id: str,
    ) -> str:
        key_rows = [row for row in self._versions if row.fpl_code == fpl_code]
        player_key = key_rows[0].player_key if key_rows else _player_key(fpl_code)

        alias_keys = {
            row.player_key
            for row in self._versions
            if row.season == season and row.fpl_element_id == element_id
        }
        if alias_keys and alias_keys != {player_key}:
            raise IdentityConflictError(
                f"season alias ({season}, {element_id}) is already assigned to another player"
            )
        season_aliases = {
            row.fpl_element_id
            for row in self._versions
            if row.season == season and row.player_key == player_key
        }
        if season_aliases and season_aliases != {element_id}:
            raise IdentityConflictError(
                f"fpl_code {fpl_code} already has another element alias in season {season}"
            )

        active = self._matching_versions(player_key, season, gameweek)
        if not active:
            existing_season = [
                row
                for row in self._versions
                if row.player_key == player_key and row.season == season
            ]
            if existing_season:
                raise IdentityConflictError(
                    f"cannot backfill a gap in identity history for {player_key} at GW {gameweek}"
                )
            self._versions.append(
                IdentityVersion(
                    player_key=player_key,
                    fpl_code=fpl_code,
                    season=season,
                    fpl_element_id=element_id,
                    understat_id=None,
                    valid_from_gw=gameweek,
                    valid_to_gw=None,
                    team_id=team_id,
                    match_method=MatchMethod.UNRESOLVED.value,
                    confidence=MatchConfidence.UNRESOLVED.value,
                    audit_note="Understat mapping not yet audited.",
                    source_artifact_id=source_artifact_id,
                )
            )
            return player_key
        if len(active) != 1:
            raise IdentityConflictError(f"overlapping identity versions for {player_key}")
        current = active[0]
        if current.fpl_code != fpl_code or current.fpl_element_id != element_id:
            raise IdentityConflictError("stable FPL identity conflicts with the active version")
        if current.team_id != team_id:
            self._replace_from_gameweek(
                current,
                gameweek,
                replace(
                    current,
                    valid_from_gw=gameweek,
                    team_id=team_id,
                    source_artifact_id=source_artifact_id,
                ),
            )
        return player_key

    def _change_mapping(
        self,
        *,
        player_key: str,
        season: str,
        gameweek: int,
        understat_id: str | None,
        match_method: str,
        confidence: str,
        audit_note: str | None,
        source_artifact_id: str,
    ) -> None:
        _positive_int(gameweek, "valid_from_gw")
        if not source_artifact_id:
            raise ValueError("source_artifact_id must not be empty")
        current = self._version_at(player_key, season, gameweek)
        updated = replace(
            current,
            understat_id=understat_id,
            valid_from_gw=gameweek,
            match_method=match_method,
            confidence=confidence,
            audit_note=audit_note,
            source_artifact_id=source_artifact_id,
        )
        if all(
            getattr(current, field) == getattr(updated, field)
            for field in ("understat_id", "match_method", "confidence", "audit_note")
        ):
            return
        self._replace_from_gameweek(current, gameweek, updated)
        self._validate()

    def _replace_from_gameweek(
        self, current: IdentityVersion, gameweek: int, updated: IdentityVersion
    ) -> None:
        index = self._versions.index(current)
        if gameweek == current.valid_from_gw:
            self._versions[index] = replace(updated, valid_to_gw=current.valid_to_gw)
            return
        if gameweek < current.valid_from_gw or (
            current.valid_to_gw is not None and gameweek > current.valid_to_gw
        ):
            raise IdentityConflictError("version update falls outside the active interval")
        self._versions[index] = replace(current, valid_to_gw=gameweek - 1)
        self._versions.append(replace(updated, valid_to_gw=current.valid_to_gw))

    def _version_at(self, player_key: str, season: str, gameweek: int) -> IdentityVersion:
        _positive_int(gameweek, "gameweek")
        matches = self._matching_versions(player_key, season, gameweek)
        if len(matches) != 1:
            raise IdentityLookupError(
                f"expected one version for {player_key}, {season}, GW {gameweek}; "
                f"found {len(matches)}"
            )
        return matches[0]

    def _matching_versions(
        self, player_key: str, season: str, gameweek: int
    ) -> list[IdentityVersion]:
        return [
            row
            for row in self._versions
            if row.player_key == player_key and row.season == season and _covers(row, gameweek)
        ]

    def _validate(self) -> None:
        PLAYER_IDENTITY_REGISTRY_SCHEMA.validate_records(
            [version.to_record() for version in self._versions]
        )


def propose_understat_matches(
    fpl_players: Iterable[Mapping[str, Any]],
    understat_players: Iterable[Mapping[str, Any]],
) -> tuple[UnderstatMatchProposal, ...]:
    """Propose normalized-name candidates without automatically accepting any mapping."""
    external_by_name: dict[str, list[str]] = {}
    for row_number, player in enumerate(understat_players, start=1):
        name = _source_string(player, row_number, "name", "player_name")
        external_id = _source_string(player, row_number, "understat_id", "id")
        external_by_name.setdefault(normalize_player_name(name), []).append(external_id)

    proposals: list[UnderstatMatchProposal] = []
    for row_number, player in enumerate(fpl_players, start=1):
        player_key = _source_string(player, row_number, "player_key")
        name = _source_string(player, row_number, "name", "web_name", "player_name")
        normalized = normalize_player_name(name)
        candidates = tuple(sorted(set(external_by_name.get(normalized, []))))
        status = (
            "unmatched" if not candidates else "candidate" if len(candidates) == 1 else "ambiguous"
        )
        proposals.append(
            UnderstatMatchProposal(
                player_key=player_key,
                normalized_name=normalized,
                candidate_ids=candidates,
                status=status,
            )
        )
    return tuple(proposals)


def normalize_player_name(value: str) -> str:
    """Normalize a name for candidate generation, never for identity acceptance."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("player name must be a non-empty string")
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_like = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.sub(r"[^\w]+", " ", ascii_like.casefold()).split())


def _player_key(fpl_code: int) -> str:
    return f"player-{uuid.uuid5(_PLAYER_KEY_NAMESPACE, f'fpl-code:{fpl_code}')}"


def _covers(row: IdentityVersion, gameweek: int) -> bool:
    return row.valid_from_gw <= gameweek and (
        row.valid_to_gw is None or gameweek <= row.valid_to_gw
    )


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _source_int(row: Mapping[str, Any], *field_names: str) -> int:
    for field_name in field_names:
        if field_name in row:
            return _positive_int(row[field_name], field_name)
    raise IdentityConflictError(f"FPL row is missing one of fields {field_names!r}")


def _source_string(row: Mapping[str, Any], row_number: int, *field_names: str) -> str:
    for field_name in field_names:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise IdentityConflictError(
        f"identity candidate row {row_number} is missing a non-empty field from {field_names!r}"
    )


def _has_note(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())
