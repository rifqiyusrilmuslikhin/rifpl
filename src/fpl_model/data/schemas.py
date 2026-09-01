"""Versioned schema contracts for the canonical FPL data tables."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from numbers import Integral, Real
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when records do not satisfy a versioned table contract."""


class FieldKind(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    UTC_DATETIME = "utc_datetime"
    STRING_LIST = "string_list"


class ValueState(StrEnum):
    """Provenance-aware state for a value supplied by an external source."""

    VALUE = "value"
    GENUINE_ZERO = "genuine_zero"
    SOURCE_UNAVAILABLE = "source_unavailable"
    ACQUISITION_FAILURE = "acquisition_failure"


@dataclass(frozen=True, slots=True)
class SourcedValue:
    """A value paired with an explicit availability state.

    Missing source fields and acquisition failures deliberately carry ``None``;
    neither is silently converted to numeric zero.
    """

    value: Any
    state: ValueState

    def __post_init__(self) -> None:
        if self.state in {ValueState.SOURCE_UNAVAILABLE, ValueState.ACQUISITION_FAILURE}:
            if self.value is not None:
                raise ValueError(f"{self.state} must not carry a value")
        elif self.value is None:
            raise ValueError(f"{self.state} must carry a value")
        elif self.state is ValueState.GENUINE_ZERO and (
            isinstance(self.value, bool) or not isinstance(self.value, Real) or self.value != 0
        ):
            raise ValueError("genuine_zero must carry numeric zero")

    @classmethod
    def from_source(cls, value: Any, *, available: bool = True) -> SourcedValue:
        if not available or _is_null(value):
            return cls(None, ValueState.SOURCE_UNAVAILABLE)
        if isinstance(value, Real) and not isinstance(value, bool) and value == 0:
            return cls(value, ValueState.GENUINE_ZERO)
        return cls(value, ValueState.VALUE)

    @classmethod
    def acquisition_failure(cls) -> SourcedValue:
        return cls(None, ValueState.ACQUISITION_FAILURE)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    kind: FieldKind
    nullable: bool = False
    choices: frozenset[Any] | None = None


RecordValidator = Callable[[Mapping[str, Any], int], None]
DatasetValidator = Callable[[Sequence[Mapping[str, Any]]], None]


@dataclass(frozen=True, slots=True)
class TableSchema:
    """Small, dependency-light schema contract with key and invariant checks."""

    name: str
    version: str
    fields: tuple[FieldSpec, ...]
    primary_key: tuple[str, ...]
    allow_extra_fields: bool = False
    record_validators: tuple[RecordValidator, ...] = ()
    dataset_validators: tuple[DatasetValidator, ...] = ()

    def validate_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        rows = list(records)
        expected = {field.name for field in self.fields}
        seen_keys: set[tuple[Any, ...]] = set()

        for row_number, row in enumerate(rows, start=1):
            missing = expected.difference(row)
            if missing:
                raise SchemaValidationError(
                    f"{self.name} schema v{self.version}: row {row_number} is missing fields "
                    f"{sorted(missing)}"
                )
            extras = set(row).difference(expected)
            if extras and not self.allow_extra_fields:
                raise SchemaValidationError(
                    f"{self.name} schema v{self.version}: row {row_number} has unexpected fields "
                    f"{sorted(extras)}"
                )

            for field in self.fields:
                _validate_field(self.name, self.version, field, row[field.name], row_number)

            key = tuple(row[field] for field in self.primary_key)
            if key in seen_keys:
                raise SchemaValidationError(
                    f"{self.name} schema v{self.version}: duplicate primary key {key!r}"
                )
            seen_keys.add(key)

            for validator in self.record_validators:
                validator(row, row_number)

        for validator in self.dataset_validators:
            validator(rows)

    def validate_frame(self, frame: Any) -> None:
        """Validate a pandas-like frame without importing pandas in this module."""
        if not hasattr(frame, "to_dict"):
            raise TypeError("frame must provide to_dict(orient='records')")
        self.validate_records(frame.to_dict(orient="records"))


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = value != value
    except (TypeError, ValueError):
        return False
    return isinstance(result, bool) and result


def _validate_field(
    table_name: str,
    version: str,
    field: FieldSpec,
    value: Any,
    row_number: int,
) -> None:
    prefix = f"{table_name} schema v{version}: row {row_number} field {field.name!r}"
    if _is_null(value):
        if field.nullable:
            return
        raise SchemaValidationError(f"{prefix} must not be null")

    valid = {
        FieldKind.STRING: lambda item: isinstance(item, str),
        FieldKind.INTEGER: lambda item: isinstance(item, Integral) and not isinstance(item, bool),
        FieldKind.NUMBER: lambda item: isinstance(item, Real)
        and not isinstance(item, bool)
        and math.isfinite(float(item)),
        FieldKind.BOOLEAN: lambda item: isinstance(item, bool),
        FieldKind.UTC_DATETIME: _is_utc_datetime,
        FieldKind.STRING_LIST: _is_string_list,
    }[field.kind](value)
    if not valid:
        raise SchemaValidationError(f"{prefix} must be {field.kind}")
    if field.choices is not None and value not in field.choices:
        raise SchemaValidationError(f"{prefix} must be one of {sorted(field.choices)!r}")


def _is_utc_datetime(value: Any) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() is not None
        and value.utcoffset().total_seconds() == 0
    )


def _is_string_list(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | bytearray)
        and all(isinstance(item, str) for item in value)
    )


def _positive_fields(*names: str) -> RecordValidator:
    def validate(row: Mapping[str, Any], row_number: int) -> None:
        for name in names:
            value = row[name]
            if value is not None and value < 0:
                raise SchemaValidationError(f"row {row_number} field {name!r} must be non-negative")

    return validate


def _snapshot_times(row: Mapping[str, Any], row_number: int) -> None:
    if row["captured_at_utc"] >= row["deadline_utc"]:
        raise SchemaValidationError(
            f"row {row_number} captured_at_utc must be strictly before deadline_utc"
        )


def _snapshot_value_states(row: Mapping[str, Any], row_number: int) -> None:
    pairs = (
        ("status", "status_value_state"),
        ("chance_of_playing_next_round", "chance_of_playing_value_state"),
        ("ep_next", "ep_next_value_state"),
    )
    for value_field, state_field in pairs:
        value = row[value_field]
        state = ValueState(row[state_field])
        if state in {ValueState.SOURCE_UNAVAILABLE, ValueState.ACQUISITION_FAILURE}:
            if not _is_null(value):
                raise SchemaValidationError(
                    f"row {row_number} field {value_field!r} must be null when "
                    f"{state_field!r} is {state.value!r}"
                )
        elif _is_null(value):
            raise SchemaValidationError(
                f"row {row_number} field {value_field!r} must not be null when "
                f"{state_field!r} is {state.value!r}"
            )
        elif state is ValueState.GENUINE_ZERO and (
            isinstance(value, bool) or not isinstance(value, Real) or value != 0
        ):
            raise SchemaValidationError(
                f"row {row_number} field {value_field!r} must be numeric zero for genuine_zero"
            )
        elif (
            state is ValueState.VALUE
            and isinstance(value, Real)
            and not isinstance(value, bool)
            and value == 0
        ):
            raise SchemaValidationError(
                f"row {row_number} field {value_field!r} numeric zero must use genuine_zero"
            )


def _model_times(row: Mapping[str, Any], row_number: int) -> None:
    deadline = row["deadline_utc"]
    if row["snapshot_captured_at_utc"] >= deadline:
        raise SchemaValidationError(
            f"row {row_number} snapshot_captured_at_utc must be strictly before deadline_utc"
        )
    if row["feature_cutoff_utc"] >= deadline:
        raise SchemaValidationError(
            f"row {row_number} feature_cutoff_utc must be strictly before deadline_utc"
        )
    if row["is_blank"] != (row["fixture_count"] == 0):
        raise SchemaValidationError(f"row {row_number} is_blank must agree with fixture_count")


def _registry_record(row: Mapping[str, Any], row_number: int) -> None:
    if row["valid_to_gw"] is not None and row["valid_to_gw"] < row["valid_from_gw"]:
        raise SchemaValidationError(f"row {row_number} has an inverted validity interval")
    if row["match_method"] == "manual" and not row["audit_note"]:
        raise SchemaValidationError(f"row {row_number} manual matches require an audit_note")
    if row["match_method"] == "unresolved" and row["confidence"] != "unresolved":
        raise SchemaValidationError(
            f"row {row_number} unresolved matches require unresolved confidence"
        )


def _registry_intervals(rows: Sequence[Mapping[str, Any]]) -> None:
    for source_field in ("fpl_element_id", "understat_id"):
        groups: dict[tuple[str, Any], list[Mapping[str, Any]]] = {}
        for row in rows:
            source_id = row[source_field]
            if source_id is not None:
                groups.setdefault((row["season"], source_id), []).append(row)
        for source_key, entries in groups.items():
            ordered = sorted(entries, key=lambda item: item["valid_from_gw"])
            for previous, current in zip(ordered, ordered[1:], strict=False):
                previous_end = previous["valid_to_gw"]
                if previous_end is None or current["valid_from_gw"] <= previous_end:
                    raise SchemaValidationError(
                        f"player_identity_registry has overlapping {source_field} intervals "
                        f"for {source_key!r}"
                    )


_VALUE_STATES = frozenset(state.value for state in ValueState)

PLAYER_FIXTURE_FACT_SCHEMA = TableSchema(
    name="player_fixture_fact",
    version="1.0.0",
    fields=(
        FieldSpec("season", FieldKind.STRING),
        FieldSpec("fixture_id", FieldKind.INTEGER),
        FieldSpec("gameweek", FieldKind.INTEGER, nullable=True),
        FieldSpec("player_key", FieldKind.STRING),
        FieldSpec("fpl_code", FieldKind.INTEGER),
        FieldSpec("fpl_element_id", FieldKind.INTEGER),
        FieldSpec("kickoff_utc", FieldKind.UTC_DATETIME),
        FieldSpec("team_id", FieldKind.INTEGER),
        FieldSpec("opponent_team_id", FieldKind.INTEGER),
        FieldSpec("was_home", FieldKind.BOOLEAN),
        FieldSpec("total_points", FieldKind.INTEGER),
        FieldSpec("minutes", FieldKind.INTEGER),
        FieldSpec("source_artifact_id", FieldKind.STRING),
        FieldSpec("source_row_number", FieldKind.INTEGER),
    ),
    primary_key=("season", "fixture_id", "player_key"),
    record_validators=(
        _positive_fields(
            "fixture_id",
            "gameweek",
            "fpl_code",
            "fpl_element_id",
            "team_id",
            "opponent_team_id",
            "minutes",
            "source_row_number",
        ),
    ),
)

DEADLINE_SNAPSHOT_SCHEMA = TableSchema(
    name="deadline_snapshot",
    version="1.0.0",
    fields=(
        FieldSpec("season", FieldKind.STRING),
        FieldSpec("gameweek", FieldKind.INTEGER),
        FieldSpec("player_key", FieldKind.STRING),
        FieldSpec("fpl_code", FieldKind.INTEGER),
        FieldSpec("fpl_element_id", FieldKind.INTEGER),
        FieldSpec("deadline_utc", FieldKind.UTC_DATETIME),
        FieldSpec("captured_at_utc", FieldKind.UTC_DATETIME),
        FieldSpec("team_id", FieldKind.INTEGER),
        FieldSpec("position", FieldKind.STRING, choices=frozenset({"GKP", "DEF", "MID", "FWD"})),
        FieldSpec("now_cost", FieldKind.INTEGER),
        FieldSpec("status", FieldKind.STRING, nullable=True),
        FieldSpec("status_value_state", FieldKind.STRING, choices=_VALUE_STATES),
        FieldSpec("chance_of_playing_next_round", FieldKind.INTEGER, nullable=True),
        FieldSpec("chance_of_playing_value_state", FieldKind.STRING, choices=_VALUE_STATES),
        FieldSpec("ep_next", FieldKind.NUMBER, nullable=True),
        FieldSpec("ep_next_value_state", FieldKind.STRING, choices=_VALUE_STATES),
        FieldSpec("source_artifact_id", FieldKind.STRING),
    ),
    primary_key=("season", "gameweek", "player_key"),
    record_validators=(
        _positive_fields(
            "gameweek",
            "fpl_code",
            "fpl_element_id",
            "team_id",
            "now_cost",
            "chance_of_playing_next_round",
        ),
        _snapshot_times,
        _snapshot_value_states,
    ),
)

PLAYER_GAMEWEEK_MODEL_SCHEMA = TableSchema(
    name="player_gameweek_model",
    version="1.0.0",
    fields=(
        FieldSpec("season", FieldKind.STRING),
        FieldSpec("gameweek", FieldKind.INTEGER),
        FieldSpec("player_key", FieldKind.STRING),
        FieldSpec("deadline_utc", FieldKind.UTC_DATETIME),
        FieldSpec("snapshot_captured_at_utc", FieldKind.UTC_DATETIME),
        FieldSpec("feature_cutoff_utc", FieldKind.UTC_DATETIME),
        FieldSpec("fixture_ids", FieldKind.STRING_LIST),
        FieldSpec("team_id_at_deadline", FieldKind.INTEGER),
        FieldSpec(
            "position_at_deadline",
            FieldKind.STRING,
            choices=frozenset({"GKP", "DEF", "MID", "FWD"}),
        ),
        FieldSpec("fixture_count", FieldKind.INTEGER),
        FieldSpec("is_blank", FieldKind.BOOLEAN),
        FieldSpec("source_artifact_ids", FieldKind.STRING_LIST),
    ),
    primary_key=("season", "gameweek", "player_key"),
    allow_extra_fields=True,
    record_validators=(
        _positive_fields("gameweek", "team_id_at_deadline", "fixture_count"),
        _model_times,
    ),
)

PLAYER_IDENTITY_REGISTRY_SCHEMA = TableSchema(
    name="player_identity_registry",
    version="1.0.0",
    fields=(
        FieldSpec("player_key", FieldKind.STRING),
        FieldSpec("fpl_code", FieldKind.INTEGER),
        FieldSpec("season", FieldKind.STRING),
        FieldSpec("fpl_element_id", FieldKind.INTEGER),
        FieldSpec("understat_id", FieldKind.STRING, nullable=True),
        FieldSpec("valid_from_gw", FieldKind.INTEGER),
        FieldSpec("valid_to_gw", FieldKind.INTEGER, nullable=True),
        FieldSpec("team_id", FieldKind.INTEGER),
        FieldSpec(
            "match_method",
            FieldKind.STRING,
            choices=frozenset({"exact_id", "audited_name_dob", "manual", "unresolved"}),
        ),
        FieldSpec(
            "confidence",
            FieldKind.STRING,
            choices=frozenset({"high", "medium", "unresolved"}),
        ),
        FieldSpec("audit_note", FieldKind.STRING, nullable=True),
        FieldSpec("source_artifact_id", FieldKind.STRING),
    ),
    primary_key=("player_key", "season", "valid_from_gw"),
    record_validators=(
        _positive_fields("fpl_code", "fpl_element_id", "valid_from_gw", "valid_to_gw", "team_id"),
        _registry_record,
    ),
    dataset_validators=(_registry_intervals,),
)

CANONICAL_SCHEMAS = {
    schema.name: schema
    for schema in (
        PLAYER_FIXTURE_FACT_SCHEMA,
        DEADLINE_SNAPSHOT_SCHEMA,
        PLAYER_GAMEWEEK_MODEL_SCHEMA,
        PLAYER_IDENTITY_REGISTRY_SCHEMA,
    )
}
