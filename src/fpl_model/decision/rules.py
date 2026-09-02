"""Versioned FPL squad and formation rules used by the decision layer."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

POSITIONS = ("GKP", "DEF", "MID", "FWD")
DEFAULT_RULES_PATH = Path("config/fpl_rules.toml")


class RulesConfigError(ValueError):
    """Raised when the captured FPL rules are incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class FPLRules:
    """The small current-rules surface needed for a one-GW decision."""

    version: str = "2026-27-v1"
    verified_on: str = "2026-09-02"
    source_url: str = "https://www.premierleague.com/en/news/2174899/fpl-basics-managing-your-team"
    squad_size: int = 15
    starting_size: int = 11
    max_players_per_club: int = 3
    squad_counts: tuple[tuple[str, int], ...] = (
        ("GKP", 2),
        ("DEF", 5),
        ("MID", 5),
        ("FWD", 3),
    )
    starting_min: tuple[tuple[str, int], ...] = (
        ("GKP", 1),
        ("DEF", 3),
        ("MID", 2),
        ("FWD", 1),
    )
    starting_max: tuple[tuple[str, int], ...] = (
        ("GKP", 1),
        ("DEF", 5),
        ("MID", 5),
        ("FWD", 3),
    )
    low_play_any: float = 0.5
    low_minutes_60: float = 0.5
    low_expected_minutes: float = 30.0

    def __post_init__(self) -> None:
        if not self.version or not self.verified_on or not self.source_url.startswith("https://"):
            raise RulesConfigError(
                "rules version, verification date, and HTTPS source are required"
            )
        if self.squad_size <= 0 or self.starting_size <= 0:
            raise RulesConfigError("squad and starting sizes must be positive")
        if self.starting_size >= self.squad_size:
            raise RulesConfigError("starting size must be smaller than squad size")
        if self.max_players_per_club <= 0:
            raise RulesConfigError("max_players_per_club must be positive")

        squad = self.squad_count_map
        minimum = self.starting_min_map
        maximum = self.starting_max_map
        for name, values in (
            ("squad", squad),
            ("starting_min", minimum),
            ("starting_max", maximum),
        ):
            if tuple(values) != POSITIONS or any(value < 0 for value in values.values()):
                raise RulesConfigError(f"{name} must define non-negative counts for {POSITIONS!r}")
        if sum(squad.values()) != self.squad_size:
            raise RulesConfigError("squad position counts must sum to squad_size")
        if sum(minimum.values()) > self.starting_size or sum(maximum.values()) < self.starting_size:
            raise RulesConfigError("starting position bounds cannot produce starting_size players")
        if any(minimum[position] > maximum[position] for position in POSITIONS):
            raise RulesConfigError("each starting minimum must be at most its maximum")
        if any(maximum[position] > squad[position] for position in POSITIONS):
            raise RulesConfigError("starting maxima cannot exceed squad counts")
        for threshold in (self.low_play_any, self.low_minutes_60):
            if not 0.0 <= threshold <= 1.0:
                raise RulesConfigError("probability risk thresholds must be in [0, 1]")
        if self.low_expected_minutes < 0:
            raise RulesConfigError("low_expected_minutes must be non-negative")

    @property
    def squad_count_map(self) -> dict[str, int]:
        return dict(self.squad_counts)

    @property
    def starting_min_map(self) -> dict[str, int]:
        return dict(self.starting_min)

    @property
    def starting_max_map(self) -> dict[str, int]:
        return dict(self.starting_max)

    def legal_formation(self, counts: dict[str, int]) -> bool:
        """Return whether exact position counts form a legal Starting XI."""
        minimum = self.starting_min_map
        maximum = self.starting_max_map
        return (
            sum(counts.get(position, 0) for position in POSITIONS) == self.starting_size
            and all(
                minimum[position] <= counts.get(position, 0) <= maximum[position]
                for position in POSITIONS
            )
            and not set(counts).difference(POSITIONS)
        )


def load_fpl_rules(path: str | Path = DEFAULT_RULES_PATH) -> FPLRules:
    """Load and validate the repository's captured FPL rules."""
    source = Path(path)
    with source.open("rb") as rules_file:
        payload = tomllib.load(rules_file)
    try:
        rules = payload["rules"]
        risk = payload["risk"]
        return FPLRules(
            version=str(rules["version"]),
            verified_on=str(rules["verified_on"]),
            source_url=str(rules["source_url"]),
            squad_size=int(rules["squad_size"]),
            starting_size=int(rules["starting_size"]),
            max_players_per_club=int(rules["max_players_per_club"]),
            squad_counts=_position_items(rules["squad"], "rules.squad"),
            starting_min=_position_items(rules["starting_min"], "rules.starting_min"),
            starting_max=_position_items(rules["starting_max"], "rules.starting_max"),
            low_play_any=float(risk["low_play_any"]),
            low_minutes_60=float(risk["low_minutes_60"]),
            low_expected_minutes=float(risk["low_expected_minutes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RulesConfigError(f"invalid FPL rules configuration at {source}: {error}") from error


def _position_items(payload: object, field: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(payload, dict):
        raise RulesConfigError(f"{field} must be a table")
    missing = sorted(set(POSITIONS).difference(payload))
    extra = sorted(set(payload).difference(POSITIONS))
    if missing or extra:
        raise RulesConfigError(f"{field} positions differ; missing={missing!r}, extra={extra!r}")
    return tuple((position, int(payload[position])) for position in POSITIONS)
