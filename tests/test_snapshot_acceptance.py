"""Acceptance tests for canonical deadlines and pre-deadline snapshots."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from fpl_model.data.snapshots import (
    DeadlineCalendar,
    SnapshotGate,
    SnapshotRejectedError,
)

DEADLINE = datetime(2026, 8, 15, 10, 0, tzinfo=UTC)


def _events() -> list[dict[str, object]]:
    return [
        {"id": 1, "deadline_time": "2026-08-15T10:00:00Z"},
        {"id": 2, "deadline_time": "2026-08-22T10:00:00Z"},
    ]


def _snapshot() -> dict[str, object]:
    events = deepcopy(_events())
    events[0]["is_next"] = True
    events[1]["is_next"] = False
    return {
        "events": events,
        "elements": [{"id": 101, "code": 5001, "team": 1}],
    }


def _gate() -> SnapshotGate:
    return SnapshotGate(DeadlineCalendar.from_events("2026-27", _events()))


def test_pre_deadline_snapshot_with_canonical_next_event_is_accepted() -> None:
    raw = _snapshot()
    accepted = _gate().accept(
        raw,
        target_gameweek=1,
        captured_at_utc=DEADLINE - timedelta(microseconds=1),
        source_artifact_id="artifact-before-deadline",
    )

    raw["elements"][0]["team"] = 99  # type: ignore[index]
    assert accepted.gameweek == 1
    assert accepted.deadline_utc == DEADLINE
    assert accepted.payload["elements"][0]["team"] == 1  # type: ignore[index]


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(seconds=1)])
def test_post_deadline_snapshot_is_deliberately_rejected(offset: timedelta) -> None:
    """Negative leakage test: at/after-deadline data must fail, never merely warn."""
    with pytest.raises(SnapshotRejectedError, match="strictly before"):
        _gate().accept(
            _snapshot(),
            target_gameweek=1,
            captured_at_utc=DEADLINE + offset,
            source_artifact_id="artifact-post-deadline",
        )


def test_embedded_deadline_mismatch_is_rejected() -> None:
    payload = _snapshot()
    payload["events"][0]["deadline_time"] = "2026-08-15T10:01:00Z"  # type: ignore[index]

    with pytest.raises(SnapshotRejectedError, match="does not match"):
        _gate().accept(
            payload,
            target_gameweek=1,
            captured_at_utc=DEADLINE - timedelta(hours=1),
            source_artifact_id="artifact-wrong-deadline",
        )


def test_payload_must_point_to_target_as_its_next_event() -> None:
    payload = _snapshot()
    payload["events"][0]["is_next"] = False  # type: ignore[index]
    payload["events"][1]["is_next"] = True  # type: ignore[index]

    with pytest.raises(SnapshotRejectedError, match="expected 1"):
        _gate().accept(
            payload,
            target_gameweek=1,
            captured_at_utc=DEADLINE - timedelta(hours=1),
            source_artifact_id="artifact-wrong-next-event",
        )


def test_missing_pre_deadline_snapshot_is_not_future_backfilled() -> None:
    with pytest.raises(SnapshotRejectedError, match="leave it missing"):
        _gate().select_latest([], gameweek=1)
