"""Deadline calendars and strict point-in-time snapshot acceptance."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral
from types import MappingProxyType
from typing import Any


class DeadlineCalendarError(ValueError):
    """The canonical deadline calendar is incomplete or internally inconsistent."""


class SnapshotRejectedError(ValueError):
    """A raw snapshot does not satisfy the pre-deadline acceptance contract."""


@dataclass(frozen=True, slots=True)
class GameweekDeadline:
    season: str
    gameweek: int
    deadline_utc: datetime


@dataclass(frozen=True, slots=True)
class AcceptedSnapshot:
    """A payload that has passed every point-in-time acceptance check."""

    season: str
    gameweek: int
    deadline_utc: datetime
    captured_at_utc: datetime
    source_artifact_id: str
    payload: Mapping[str, Any]


class DeadlineCalendar:
    """Canonical FPL gameweek deadlines for one season."""

    def __init__(self, season: str, deadlines: Iterable[GameweekDeadline]) -> None:
        if not season:
            raise DeadlineCalendarError("season must not be empty")
        by_gameweek: dict[int, datetime] = {}
        for entry in deadlines:
            if entry.season != season:
                raise DeadlineCalendarError(
                    f"deadline season {entry.season!r} does not match calendar season {season!r}"
                )
            _require_positive_integer(entry.gameweek, "gameweek", DeadlineCalendarError)
            _require_utc(entry.deadline_utc, "deadline_utc", DeadlineCalendarError)
            if entry.gameweek in by_gameweek:
                raise DeadlineCalendarError(f"duplicate deadline for gameweek {entry.gameweek}")
            by_gameweek[entry.gameweek] = entry.deadline_utc
        if not by_gameweek:
            raise DeadlineCalendarError("deadline calendar must contain at least one gameweek")

        ordered = sorted(by_gameweek.items())
        for (previous_gw, previous), (current_gw, current) in zip(
            ordered, ordered[1:], strict=False
        ):
            if current <= previous:
                raise DeadlineCalendarError(
                    f"deadline for gameweek {current_gw} must be after gameweek {previous_gw}"
                )
        self.season = season
        self._by_gameweek = MappingProxyType(by_gameweek)

    @classmethod
    def from_events(cls, season: str, events: Iterable[Mapping[str, Any]]) -> DeadlineCalendar:
        """Build the canonical calendar from FPL event records."""
        entries: list[GameweekDeadline] = []
        for row_number, event in enumerate(events, start=1):
            if not isinstance(event, Mapping):
                raise DeadlineCalendarError(f"event {row_number} must be an object")
            try:
                gameweek = event["id"]
                raw_deadline = event["deadline_time"]
            except KeyError as error:
                raise DeadlineCalendarError(
                    f"event {row_number} is missing required field {error.args[0]!r}"
                ) from error
            _require_positive_integer(gameweek, "event id", DeadlineCalendarError)
            entries.append(
                GameweekDeadline(
                    season=season,
                    gameweek=gameweek,
                    deadline_utc=_parse_utc_timestamp(
                        raw_deadline, f"event {gameweek} deadline_time", DeadlineCalendarError
                    ),
                )
            )
        return cls(season, entries)

    @classmethod
    def from_bootstrap(cls, season: str, payload: Mapping[str, Any]) -> DeadlineCalendar:
        events = payload.get("events")
        if not isinstance(events, list):
            raise DeadlineCalendarError("bootstrap payload field 'events' must be a list")
        return cls.from_events(season, events)

    def deadline_for(self, gameweek: int) -> datetime:
        _require_positive_integer(gameweek, "gameweek", DeadlineCalendarError)
        try:
            return self._by_gameweek[gameweek]
        except KeyError as error:
            raise DeadlineCalendarError(f"no canonical deadline for gameweek {gameweek}") from error

    @property
    def gameweeks(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_gameweek))


class SnapshotGate:
    """Accept only snapshots known to have existed before the target deadline."""

    def __init__(self, calendar: DeadlineCalendar) -> None:
        self.calendar = calendar

    def accept(
        self,
        payload: Mapping[str, Any],
        *,
        target_gameweek: int,
        captured_at_utc: datetime,
        source_artifact_id: str,
    ) -> AcceptedSnapshot:
        if not isinstance(payload, Mapping):
            raise SnapshotRejectedError("snapshot payload must be an object")
        if not source_artifact_id:
            raise SnapshotRejectedError("source_artifact_id must not be empty")
        _require_positive_integer(target_gameweek, "target_gameweek", SnapshotRejectedError)
        _require_utc(captured_at_utc, "captured_at_utc", SnapshotRejectedError)

        events = payload.get("events")
        if not isinstance(events, list):
            raise SnapshotRejectedError("snapshot payload field 'events' must be a list")
        if not isinstance(payload.get("elements"), list):
            raise SnapshotRejectedError("snapshot payload field 'elements' must be a list")

        next_events = [event for event in events if _is_next_event(event)]
        if len(next_events) != 1:
            raise SnapshotRejectedError(
                "snapshot must identify exactly one event as the next event"
            )
        next_event = next_events[0]
        next_gameweek = next_event.get("id")
        if next_gameweek != target_gameweek:
            raise SnapshotRejectedError(
                f"snapshot next event is gameweek {next_gameweek!r}, expected {target_gameweek}"
            )

        canonical_deadline = self.calendar.deadline_for(target_gameweek)
        embedded_deadline = _parse_utc_timestamp(
            next_event.get("deadline_time"),
            f"event {target_gameweek} deadline_time",
            SnapshotRejectedError,
        )
        if embedded_deadline != canonical_deadline:
            raise SnapshotRejectedError(
                "snapshot embedded deadline does not match the canonical deadline"
            )
        if captured_at_utc >= canonical_deadline:
            raise SnapshotRejectedError(
                "snapshot captured_at_utc must be strictly before the canonical deadline"
            )

        # Copy the JSON-like payload so later mutation of a raw candidate cannot alter
        # the already accepted point-in-time object.
        accepted_payload = MappingProxyType(deepcopy(dict(payload)))
        return AcceptedSnapshot(
            season=self.calendar.season,
            gameweek=target_gameweek,
            deadline_utc=canonical_deadline,
            captured_at_utc=captured_at_utc,
            source_artifact_id=source_artifact_id,
            payload=accepted_payload,
        )

    def select_latest(
        self, snapshots: Sequence[AcceptedSnapshot], gameweek: int
    ) -> AcceptedSnapshot:
        """Choose only among accepted snapshots; raw future candidates cannot be backfilled."""
        candidates = [
            snapshot
            for snapshot in snapshots
            if snapshot.season == self.calendar.season and snapshot.gameweek == gameweek
        ]
        if not candidates:
            raise SnapshotRejectedError(
                f"no accepted pre-deadline snapshot for gameweek {gameweek}; leave it missing"
            )
        for snapshot in candidates:
            if snapshot.deadline_utc != self.calendar.deadline_for(gameweek):
                raise SnapshotRejectedError("accepted snapshot has non-canonical provenance")
            if snapshot.captured_at_utc >= snapshot.deadline_utc:
                raise SnapshotRejectedError("accepted snapshot contains post-deadline provenance")
        return max(candidates, key=lambda snapshot: snapshot.captured_at_utc)


def _is_next_event(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("is_next") is True


def _parse_utc_timestamp(value: Any, field_name: str, error_type: type[ValueError]) -> datetime:
    if not isinstance(value, str) or not value:
        raise error_type(f"{field_name} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise error_type(f"{field_name} is not a valid ISO-8601 timestamp") from error
    _require_utc(parsed, field_name, error_type)
    return parsed.astimezone(UTC)


def _require_utc(value: Any, field_name: str, error_type: type[ValueError]) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != UTC.utcoffset(value)
    ):
        raise error_type(f"{field_name} must be timezone-aware UTC")


def _require_positive_integer(value: Any, field_name: str, error_type: type[ValueError]) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise error_type(f"{field_name} must be a positive integer")
