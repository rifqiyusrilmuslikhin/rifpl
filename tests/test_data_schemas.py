"""Tests for canonical data and identity contracts."""

from datetime import UTC, datetime, timedelta

import pytest

from fpl_model.data.schemas import (
    DEADLINE_SNAPSHOT_SCHEMA,
    PLAYER_FIXTURE_FACT_SCHEMA,
    PLAYER_GAMEWEEK_MODEL_SCHEMA,
    PLAYER_IDENTITY_REGISTRY_SCHEMA,
    SchemaValidationError,
    SourcedValue,
    ValueState,
)

DEADLINE = datetime(2026, 8, 15, 10, 0, tzinfo=UTC)


def _fixture_fact(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": "2026-27",
        "fixture_id": 9001,
        "gameweek": 1,
        "player_key": "player-5001",
        "fpl_code": 5001,
        "fpl_element_id": 101,
        "kickoff_utc": DEADLINE + timedelta(hours=3),
        "team_id": 1,
        "opponent_team_id": 2,
        "was_home": True,
        "total_points": 0,
        "minutes": 90,
        "source_artifact_id": "artifact-a",
        "source_row_number": 2,
    }
    row.update(updates)
    return row


def _deadline_snapshot(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": "2026-27",
        "gameweek": 1,
        "player_key": "player-5001",
        "fpl_code": 5001,
        "fpl_element_id": 101,
        "deadline_utc": DEADLINE,
        "captured_at_utc": DEADLINE - timedelta(hours=1),
        "team_id": 1,
        "position": "MID",
        "now_cost": 75,
        "status": "a",
        "status_value_state": "value",
        "chance_of_playing_next_round": None,
        "chance_of_playing_value_state": "source_unavailable",
        "ep_next": 0.0,
        "ep_next_value_state": "genuine_zero",
        "source_artifact_id": "artifact-a",
    }
    row.update(updates)
    return row


def _model_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "season": "2026-27",
        "gameweek": 1,
        "player_key": "player-5001",
        "deadline_utc": DEADLINE,
        "snapshot_captured_at_utc": DEADLINE - timedelta(hours=1),
        "feature_cutoff_utc": DEADLINE - timedelta(microseconds=1),
        "fixture_ids": ["9001"],
        "team_id_at_deadline": 1,
        "position_at_deadline": "MID",
        "fixture_count": 1,
        "is_blank": False,
        "source_artifact_ids": ["artifact-a"],
    }
    row.update(updates)
    return row


def _identity_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "player_key": "player-5001",
        "fpl_code": 5001,
        "season": "2026-27",
        "fpl_element_id": 101,
        "understat_id": "u-100",
        "valid_from_gw": 1,
        "valid_to_gw": 20,
        "team_id": 1,
        "match_method": "exact_id",
        "confidence": "high",
        "audit_note": None,
        "source_artifact_id": "artifact-a",
    }
    row.update(updates)
    return row


def test_all_canonical_contracts_accept_valid_records() -> None:
    PLAYER_FIXTURE_FACT_SCHEMA.validate_records([_fixture_fact()])
    DEADLINE_SNAPSHOT_SCHEMA.validate_records([_deadline_snapshot()])
    PLAYER_GAMEWEEK_MODEL_SCHEMA.validate_records([_model_row()])
    PLAYER_IDENTITY_REGISTRY_SCHEMA.validate_records([_identity_row()])


def test_schema_drift_and_duplicate_keys_fail_clearly() -> None:
    drifted = _fixture_fact()
    drifted.pop("minutes")
    with pytest.raises(SchemaValidationError, match="missing fields.*minutes"):
        PLAYER_FIXTURE_FACT_SCHEMA.validate_records([drifted])

    with pytest.raises(SchemaValidationError, match="duplicate primary key"):
        PLAYER_FIXTURE_FACT_SCHEMA.validate_records([_fixture_fact(), _fixture_fact()])


def test_point_in_time_and_blank_invariants_are_enforced() -> None:
    with pytest.raises(SchemaValidationError, match="strictly before"):
        DEADLINE_SNAPSHOT_SCHEMA.validate_records([_deadline_snapshot(captured_at_utc=DEADLINE)])
    with pytest.raises(SchemaValidationError, match="is_blank"):
        PLAYER_GAMEWEEK_MODEL_SCHEMA.validate_records(
            [_model_row(fixture_count=0, is_blank=False, fixture_ids=[])]
        )


def test_identity_intervals_cannot_overlap_for_a_source_id() -> None:
    second = _identity_row(
        player_key="player-other",
        fpl_element_id=102,
        valid_from_gw=20,
        valid_to_gw=None,
    )
    with pytest.raises(SchemaValidationError, match="overlapping understat_id intervals"):
        PLAYER_IDENTITY_REGISTRY_SCHEMA.validate_records([_identity_row(), second])


def test_manual_identity_requires_audit_note() -> None:
    with pytest.raises(SchemaValidationError, match="audit_note"):
        PLAYER_IDENTITY_REGISTRY_SCHEMA.validate_records(
            [_identity_row(match_method="manual", audit_note=None)]
        )


def test_zero_missing_and_acquisition_failure_remain_distinct() -> None:
    zero = SourcedValue.from_source(0)
    missing = SourcedValue.from_source(None)
    failed = SourcedValue.acquisition_failure()

    assert zero == SourcedValue(0, ValueState.GENUINE_ZERO)
    assert missing == SourcedValue(None, ValueState.SOURCE_UNAVAILABLE)
    assert failed == SourcedValue(None, ValueState.ACQUISITION_FAILURE)
    assert len({zero.state, missing.state, failed.state}) == 3

    with pytest.raises(SchemaValidationError, match="must be null"):
        DEADLINE_SNAPSHOT_SCHEMA.validate_records(
            [
                _deadline_snapshot(
                    ep_next=4.2,
                    ep_next_value_state="source_unavailable",
                )
            ]
        )
