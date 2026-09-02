"""Public Team ID ingestion with a deliberately small, offline-testable boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from urllib.parse import urljoin

from fpl_model.data.acquisition import AcquisitionError, Fetcher, fetch_url


class TeamPayloadError(ValueError):
    """Raised when a public Team ID response cannot represent an auditable squad state."""


@dataclass(frozen=True, slots=True)
class TeamPick:
    element_id: int
    squad_position: int
    multiplier: int = 0
    is_captain: bool = False
    is_vice_captain: bool = False
    purchase_price: int | None = None
    selling_price: int | None = None


@dataclass(frozen=True, slots=True)
class TeamState:
    """The public portion of a manager's squad for one FPL event."""

    team_id: int | str
    gameweek: int
    picks: tuple[TeamPick, ...]
    bank: int | None = None
    event_transfers: int | None = None
    free_transfers: int | None = None
    source_url: str | None = None
    source_payload_sha256: str | None = None

    @property
    def element_ids(self) -> tuple[int, ...]:
        return tuple(
            pick.element_id for pick in sorted(self.picks, key=lambda item: item.squad_position)
        )

    @classmethod
    def from_element_ids(
        cls,
        element_ids: Sequence[int],
        *,
        team_id: int | str,
        gameweek: int,
    ) -> TeamState:
        """Build an in-memory state for historical backtests or offline use."""
        picks = tuple(
            TeamPick(
                element_id=int(element_id),
                squad_position=position,
                is_captain=position == 1,
                is_vice_captain=position == 2,
            )
            for position, element_id in enumerate(element_ids, start=1)
        )
        return cls(team_id=team_id, gameweek=gameweek, picks=picks)


class PublicTeamLoader:
    """Load a manager's event picks from the unauthenticated official FPL endpoint.

    The public API exposes an event squad after that event's deadline. Callers remain
    responsible for ensuring the selected event is a valid point-in-time proxy for the target
    decision; this loader never guesses an unpublished next-event squad.
    """

    DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api/"

    def __init__(
        self,
        fetcher: Fetcher = fetch_url,
        *,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.fetcher = fetcher
        self.base_url = base_url.rstrip("/") + "/"

    def load(self, team_id: int, gameweek: int) -> TeamState:
        if isinstance(team_id, bool) or not isinstance(team_id, Integral) or team_id <= 0:
            raise ValueError("team_id must be a positive integer")
        if isinstance(gameweek, bool) or not isinstance(gameweek, Integral) or gameweek <= 0:
            raise ValueError("gameweek must be a positive integer")
        url = urljoin(self.base_url, f"entry/{int(team_id)}/event/{int(gameweek)}/picks/")
        try:
            response = self.fetcher(url)
        except AcquisitionError:
            raise
        except Exception as error:
            raise AcquisitionError(url, str(error)) from error
        try:
            payload = json.loads(response.content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TeamPayloadError(f"{url}: response is not valid JSON: {error}") from error
        picks, history = validate_team_payload(payload, source=url)
        return TeamState(
            team_id=int(team_id),
            gameweek=int(gameweek),
            picks=tuple(_parse_pick(pick) for pick in picks),
            bank=_optional_integer(history, "bank", minimum=0),
            event_transfers=_optional_integer(history, "event_transfers", minimum=0),
            # This field is not reliably present in the public event endpoint.
            free_transfers=_optional_integer(history, "free_transfers", minimum=0),
            source_url=url,
            source_payload_sha256=hashlib.sha256(response.content).hexdigest(),
        )


def validate_team_payload(
    payload: object,
    *,
    source: str = "public Team ID",
) -> tuple[list[Mapping[str, object]], Mapping[str, object]]:
    """Validate the minimum public event-picks contract before constructing a state."""
    if not isinstance(payload, Mapping):
        raise TeamPayloadError(f"{source}: expected a JSON object")
    if "detail" in payload and "picks" not in payload:
        raise TeamPayloadError(f"{source}: Team ID is unavailable: {payload['detail']!s}")
    picks = payload.get("picks")
    history = payload.get("entry_history")
    if not isinstance(picks, list):
        raise TeamPayloadError(f"{source}: field 'picks' must be a list")
    if not isinstance(history, Mapping):
        raise TeamPayloadError(f"{source}: field 'entry_history' must be an object")
    if len(picks) != 15:
        raise TeamPayloadError(f"{source}: expected exactly 15 picks, received {len(picks)}")

    required = {"element", "position", "multiplier", "is_captain", "is_vice_captain"}
    elements: list[int] = []
    positions: list[int] = []
    captain_count = 0
    vice_count = 0
    typed_picks: list[Mapping[str, object]] = []
    for index, pick in enumerate(picks):
        if not isinstance(pick, Mapping):
            raise TeamPayloadError(f"{source}: pick {index} must be an object")
        missing = sorted(required.difference(pick))
        if missing:
            raise TeamPayloadError(f"{source}: pick {index} missing fields {missing!r}")
        element = _required_integer(pick, "element", minimum=1, context=f"pick {index}")
        position = _required_integer(pick, "position", minimum=1, context=f"pick {index}")
        _required_integer(pick, "multiplier", minimum=0, context=f"pick {index}")
        for flag in ("is_captain", "is_vice_captain"):
            if not isinstance(pick[flag], bool):
                raise TeamPayloadError(f"{source}: pick {index} field {flag!r} must be boolean")
        elements.append(element)
        positions.append(position)
        captain_count += int(pick["is_captain"])
        vice_count += int(pick["is_vice_captain"])
        typed_picks.append(pick)
    if len(set(elements)) != 15:
        raise TeamPayloadError(f"{source}: pick element IDs must be unique")
    if sorted(positions) != list(range(1, 16)):
        raise TeamPayloadError(f"{source}: pick positions must be exactly 1 through 15")
    if captain_count != 1 or vice_count != 1:
        raise TeamPayloadError(
            f"{source}: payload must identify exactly one captain and vice-captain"
        )
    if any(pick["is_captain"] and pick["is_vice_captain"] for pick in typed_picks):
        raise TeamPayloadError(f"{source}: captain and vice-captain must be different players")
    return typed_picks, history


def _parse_pick(pick: Mapping[str, object]) -> TeamPick:
    return TeamPick(
        element_id=int(pick["element"]),
        squad_position=int(pick["position"]),
        multiplier=int(pick["multiplier"]),
        is_captain=bool(pick["is_captain"]),
        is_vice_captain=bool(pick["is_vice_captain"]),
        purchase_price=_optional_integer(pick, "purchase_price", minimum=0),
        selling_price=_optional_integer(pick, "selling_price", minimum=0),
    )


def _required_integer(
    payload: Mapping[str, object],
    field: str,
    *,
    minimum: int,
    context: str,
) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise TeamPayloadError(f"{context} field {field!r} must be an integer >= {minimum}")
    return int(value)


def _optional_integer(payload: Mapping[str, object], field: str, *, minimum: int) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    return _required_integer(payload, field, minimum=minimum, context="entry_history")
