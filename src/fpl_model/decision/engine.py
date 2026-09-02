"""Deterministic one-gameweek lineup, bench, and captain decisions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.decision.rules import POSITIONS, FPLRules, load_fpl_rules
from fpl_model.decision.team import TeamState


class DecisionInputError(ValueError):
    """Raised when a legal, unambiguous decision cannot be produced."""


@dataclass(frozen=True, slots=True)
class PredictionColumns:
    """Map a promoted model artifact's columns to the stable decision contract."""

    element_id: str = "fpl_element_id"
    player_key: str = "player_key"
    player_name: str | None = "player_name"
    position: str = "position"
    club_id: str = "team_id"
    xpts: str = "xpts"
    p_play_any: str = "p_play_any"
    p_minutes_60: str = "p_minutes_60"
    expected_minutes: str = "expected_minutes"
    data_quality: str | None = "data_quality_flag"


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    rules_version: str
    rules_verified_on: str
    rules_source_url: str
    xpts_column: str
    team_payload_sha256: str | None
    snapshot_artifact_id: str | None
    model_artifact_id: str | None


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """A legal recommendation plus its complete player-level evidence."""

    team_id: int | str
    gameweek: int
    formation: str
    captain_player_key: str
    vice_captain_player_key: str
    player_table: pd.DataFrame
    trace: DecisionTrace
    bank: int | None = None
    free_transfers: int | None = None

    @property
    def starting_xi(self) -> pd.DataFrame:
        return self.player_table.loc[self.player_table["squad_role"] == "STARTER"].copy()

    @property
    def bench(self) -> pd.DataFrame:
        return self.player_table.loc[self.player_table["squad_role"] == "BENCH"].copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "team_id": self.team_id,
            "gameweek": self.gameweek,
            "formation": self.formation,
            "captain_player_key": self.captain_player_key,
            "vice_captain_player_key": self.vice_captain_player_key,
            "bank": self.bank,
            "free_transfers": self.free_transfers,
            "trace": asdict(self.trace),
            "players": self.player_table.to_dict(orient="records"),
        }


class DecisionEngine:
    """Apply captured FPL constraints to an already-selected champion prediction."""

    def __init__(self, rules: FPLRules | None = None) -> None:
        self.rules = rules or load_fpl_rules()

    def decide(
        self,
        team: TeamState,
        predictions: pd.DataFrame,
        *,
        columns: PredictionColumns | None = None,
        snapshot_artifact_id: str | None = None,
        model_artifact_id: str | None = None,
    ) -> DecisionResult:
        mapping = columns or PredictionColumns()
        squad = validate_squad(team, predictions, columns=mapping, rules=self.rules)
        starter_indices = select_legal_starting_xi(squad, rules=self.rules)
        starters = squad.loc[list(starter_indices)].copy()
        bench = squad.drop(index=list(starter_indices)).copy()
        ordered_bench = order_bench(bench)

        captain_index = _captain_index(starters)
        vice_index = _vice_captain_index(
            starters.drop(index=captain_index), starters.loc[captain_index]
        )
        starter_order = _rank_rows(starters, ("xpts", "p_minutes_60", "p_play_any"))

        table = squad.copy()
        table["squad_role"] = "BENCH"
        table["lineup_rank"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
        table["bench_order"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
        table["is_captain"] = False
        table["is_vice_captain"] = False
        for rank, index in enumerate(starter_order, start=1):
            table.at[index, "squad_role"] = "STARTER"
            table.at[index, "lineup_rank"] = rank
        for bench_position, index in enumerate(ordered_bench, start=1):
            table.at[index, "bench_order"] = bench_position
        table.at[captain_index, "is_captain"] = True
        table.at[vice_index, "is_vice_captain"] = True
        table["decision_reason"] = _decision_reasons(
            table,
            captain_index=captain_index,
            vice_index=vice_index,
        )
        table = table.sort_values(
            ["squad_role", "lineup_rank", "bench_order", "player_key"],
            ascending=[False, True, True, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)

        formation_counts = Counter(starters["position"])
        formation = "-".join(str(formation_counts[position]) for position in ("DEF", "MID", "FWD"))
        trace = DecisionTrace(
            rules_version=self.rules.version,
            rules_verified_on=self.rules.verified_on,
            rules_source_url=self.rules.source_url,
            xpts_column=mapping.xpts,
            team_payload_sha256=team.source_payload_sha256,
            snapshot_artifact_id=snapshot_artifact_id,
            model_artifact_id=model_artifact_id,
        )
        return DecisionResult(
            team_id=team.team_id,
            gameweek=team.gameweek,
            formation=formation,
            captain_player_key=str(squad.at[captain_index, "player_key"]),
            vice_captain_player_key=str(squad.at[vice_index, "player_key"]),
            player_table=table,
            trace=trace,
            bank=team.bank,
            free_transfers=team.free_transfers,
        )


def make_decision(
    team: TeamState,
    predictions: pd.DataFrame,
    *,
    rules: FPLRules | None = None,
    columns: PredictionColumns | None = None,
    snapshot_artifact_id: str | None = None,
    model_artifact_id: str | None = None,
) -> DecisionResult:
    """Functional entry point for notebooks and small integrations."""
    return DecisionEngine(rules).decide(
        team,
        predictions,
        columns=columns,
        snapshot_artifact_id=snapshot_artifact_id,
        model_artifact_id=model_artifact_id,
    )


def validate_squad(
    team: TeamState,
    predictions: pd.DataFrame,
    *,
    columns: PredictionColumns | None = None,
    rules: FPLRules | None = None,
) -> pd.DataFrame:
    """Join a Team ID state to predictions and enforce full squad constraints."""
    mapping = columns or PredictionColumns()
    active_rules = rules or load_fpl_rules()
    if team.gameweek <= 0:
        raise DecisionInputError("team state gameweek must be positive")
    if len(team.picks) != active_rules.squad_size:
        raise DecisionInputError(
            "squad must contain exactly "
            f"{active_rules.squad_size} picks; received {len(team.picks)}"
        )
    elements = [pick.element_id for pick in team.picks]
    if len(set(elements)) != len(elements):
        raise DecisionInputError("squad element IDs must be unique")

    required = {
        mapping.element_id,
        mapping.position,
        mapping.club_id,
        mapping.xpts,
        mapping.p_play_any,
        mapping.p_minutes_60,
        mapping.expected_minutes,
    }
    missing_columns = sorted(required.difference(predictions.columns))
    if missing_columns:
        raise DecisionInputError(f"prediction frame is missing columns {missing_columns!r}")
    if predictions[mapping.element_id].duplicated().any():
        duplicates = sorted(
            predictions.loc[predictions[mapping.element_id].duplicated(False), mapping.element_id]
            .astype(str)
            .unique()
        )
        raise DecisionInputError(f"prediction frame has duplicate element IDs {duplicates!r}")

    selected = predictions.loc[predictions[mapping.element_id].isin(elements)].copy()
    found = set(selected[mapping.element_id].tolist())
    missing_predictions = sorted(set(elements).difference(found))
    if missing_predictions:
        raise DecisionInputError(
            f"missing prediction for squad element IDs {missing_predictions!r}"
        )
    if len(selected) != active_rules.squad_size:
        raise DecisionInputError(
            "prediction element IDs must use the same scalar type as Team ID picks"
        )

    rename = {
        mapping.element_id: "element_id",
        mapping.position: "position",
        mapping.club_id: "club_id",
        mapping.xpts: "xpts",
        mapping.p_play_any: "p_play_any",
        mapping.p_minutes_60: "p_minutes_60",
        mapping.expected_minutes: "expected_minutes",
    }
    if mapping.player_key and mapping.player_key in selected:
        rename[mapping.player_key] = "player_key"
    if mapping.player_name and mapping.player_name in selected:
        rename[mapping.player_name] = "player_name"
    if mapping.data_quality and mapping.data_quality in selected:
        rename[mapping.data_quality] = "source_data_quality"
    # Retained artifacts can contain both the promoted model and a canonical-looking column
    # from another arm. The explicit mapping wins without creating duplicate column labels.
    for source, destination in rename.items():
        if source != destination and destination in selected.columns:
            selected = selected.drop(columns=destination)
    selected = selected.rename(columns=rename)
    if "player_key" not in selected:
        selected["player_key"] = selected["element_id"].map(lambda value: f"fpl:{value}")
    if "player_name" not in selected:
        selected["player_name"] = selected["player_key"]
    selected["position"] = selected["position"].map(_normalise_position)
    if selected["position"].isna().any():
        bad = predictions.loc[
            selected.index[selected["position"].isna()], mapping.position
        ].tolist()
        raise DecisionInputError(f"unknown player positions {bad!r}")
    if selected["player_key"].isna().any() or selected["player_key"].astype(str).duplicated().any():
        raise DecisionInputError("player keys must be present and unique within a squad")
    selected["player_key"] = selected["player_key"].astype(str)
    if selected["club_id"].isna().any():
        raise DecisionInputError("club IDs must be present to validate the per-club limit")

    numeric = ("xpts", "p_play_any", "p_minutes_60", "expected_minutes")
    for column in numeric:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        values = selected[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DecisionInputError(f"missing or non-finite {column} prediction")
    if (selected["expected_minutes"] < 0).any():
        raise DecisionInputError("expected minutes must be non-negative")
    probabilities = selected[["p_play_any", "p_minutes_60"]]
    if ((probabilities < 0) | (probabilities > 1)).any(axis=None):
        raise DecisionInputError("participation probabilities must be in [0, 1]")
    if (selected["p_minutes_60"] > selected["p_play_any"] + 1e-12).any():
        raise DecisionInputError("P(60+) cannot exceed P(play_any)")

    position_counts = Counter(selected["position"])
    expected_counts = active_rules.squad_count_map
    if dict(position_counts) != expected_counts:
        observed = {position: position_counts.get(position, 0) for position in POSITIONS}
        raise DecisionInputError(
            f"illegal squad position counts; expected {expected_counts!r}, observed {observed!r}"
        )
    club_counts = selected.groupby("club_id", dropna=False).size()
    if (club_counts > active_rules.max_players_per_club).any():
        offenders = club_counts.loc[club_counts > active_rules.max_players_per_club].to_dict()
        raise DecisionInputError(
            f"squad exceeds {active_rules.max_players_per_club} players per club: {offenders!r}"
        )

    selected["expected_substitute_value"] = selected["xpts"] * selected["p_play_any"]
    if "source_data_quality" not in selected:
        selected["data_quality_flag"] = selected.apply(_inferred_data_quality, axis=1)
    else:
        selected["data_quality_flag"] = (
            selected["source_data_quality"].fillna("UNSPECIFIED").astype(str)
        )
    selected["risk_flags"] = selected.apply(lambda row: _risk_flags(row, active_rules), axis=1)
    return selected.sort_values("player_key", kind="stable").reset_index(drop=True)


def select_legal_starting_xi(
    squad: pd.DataFrame,
    *,
    rules: FPLRules | None = None,
    score_column: str = "xpts",
) -> tuple[int, ...]:
    """Return indices of the maximum-score legal XI with canonical tie-breaking."""
    active_rules = rules or load_fpl_rules()
    required = {"position", "player_key", score_column}
    missing = sorted(required.difference(squad.columns))
    if missing:
        raise DecisionInputError(f"lineup frame is missing columns {missing!r}")
    best_indices: tuple[int, ...] | None = None
    best_score = float("-inf")
    best_key: tuple[str, ...] | None = None
    for candidate in combinations(squad.index.tolist(), active_rules.starting_size):
        selected = squad.loc[list(candidate)]
        if not active_rules.legal_formation(dict(Counter(selected["position"]))):
            continue
        score = float(selected[score_column].sum())
        key = tuple(sorted(selected["player_key"].astype(str)))
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and (best_key is None or key < best_key)
        ):
            best_indices = tuple(candidate)
            best_score = score
            best_key = key
    if best_indices is None:
        raise DecisionInputError("no legal Starting XI can be formed from the squad")
    return best_indices


def order_bench(bench: pd.DataFrame) -> tuple[int, ...]:
    """Order outfield substitutes by availability-weighted value, then place reserve GKP."""
    required = {"position", "player_key", "xpts", "p_play_any", "expected_substitute_value"}
    missing = sorted(required.difference(bench.columns))
    if missing:
        raise DecisionInputError(f"bench frame is missing columns {missing!r}")
    goalkeepers = bench.loc[bench["position"] == "GKP"]
    outfield = bench.loc[bench["position"] != "GKP"]
    if len(goalkeepers) != 1 or len(outfield) != 3:
        raise DecisionInputError(
            "a legal FPL bench must contain one GKP and three outfield players"
        )
    ordered = outfield.sort_values(
        ["expected_substitute_value", "xpts", "p_play_any", "player_key"],
        ascending=[False, False, False, True],
        kind="stable",
    ).index.tolist()
    ordered.extend(goalkeepers.sort_values("player_key", kind="stable").index.tolist())
    return tuple(ordered)


def _captain_index(starters: pd.DataFrame) -> int:
    return int(
        starters.sort_values(
            ["xpts", "p_minutes_60", "p_play_any", "player_key"],
            ascending=[False, False, False, True],
            kind="stable",
        ).index[0]
    )


def _vice_captain_index(candidates: pd.DataFrame, captain: pd.Series) -> int:
    ranked = candidates.copy()
    ranked["vice_expected_coverage"] = (
        (1.0 - float(captain["p_play_any"])) * ranked["p_play_any"] * ranked["xpts"]
    )
    return int(
        ranked.sort_values(
            ["vice_expected_coverage", "xpts", "p_play_any", "p_minutes_60", "player_key"],
            ascending=[False, False, False, False, True],
            kind="stable",
        ).index[0]
    )


def _rank_rows(frame: pd.DataFrame, score_columns: Sequence[str]) -> list[int]:
    columns = [*score_columns, "player_key"]
    ascending = [False] * len(score_columns) + [True]
    return frame.sort_values(columns, ascending=ascending, kind="stable").index.tolist()


def _normalise_position(value: object) -> str | None:
    aliases: dict[object, str] = {
        1: "GKP",
        2: "DEF",
        3: "MID",
        4: "FWD",
        "1": "GKP",
        "2": "DEF",
        "3": "MID",
        "4": "FWD",
        "GK": "GKP",
        "GKP": "GKP",
        "DEF": "DEF",
        "MID": "MID",
        "FWD": "FWD",
    }
    return aliases.get(value if not isinstance(value, str) else value.upper())


def _risk_flags(row: pd.Series, rules: FPLRules) -> str:
    flags = []
    if "fixture_count" in row and pd.notna(row["fixture_count"]) and row["fixture_count"] == 0:
        flags.append("BLANK_GAMEWEEK")
    if "status" in row and pd.notna(row["status"]) and str(row["status"]).casefold() != "a":
        flags.append(f"STATUS_{str(row['status']).upper()}")
    chance_field = "chance_of_playing_next_round"
    if chance_field in row and pd.notna(row[chance_field]) and float(row[chance_field]) < 100.0:
        flags.append("OFFICIAL_CHANCE_BELOW_100")
    if float(row["p_play_any"]) < rules.low_play_any:
        flags.append("LOW_PLAY_ANY")
    if float(row["p_minutes_60"]) < rules.low_minutes_60:
        flags.append("LOW_60_PLUS")
    if float(row["expected_minutes"]) < rules.low_expected_minutes:
        flags.append("LOW_EXPECTED_MINUTES")
    if str(row["data_quality_flag"]).upper() not in {"OK", "COMPLETE"}:
        flags.append("DATA_QUALITY_WARNING")
    return "|".join(flags) if flags else "NONE"


def _inferred_data_quality(row: pd.Series) -> str:
    state_columns = [column for column in row.index if str(column).endswith("_value_state")]
    states = {str(row[column]).casefold() for column in state_columns if pd.notna(row[column])}
    if "acquisition_failure" in states:
        return "ACQUISITION_FAILURE"
    if "source_unavailable" in states:
        return "SOURCE_UNAVAILABLE"
    return "OK"


def _decision_reasons(
    table: pd.DataFrame,
    *,
    captain_index: int,
    vice_index: int,
) -> pd.Series:
    reasons: dict[int, str] = {}
    captain_play = float(table.at[captain_index, "p_play_any"])
    for index, row in table.iterrows():
        if index == captain_index:
            reason = f"captain: highest starter xPts ({row['xpts']:.3f})"
        elif index == vice_index:
            coverage = (1.0 - captain_play) * float(row["p_play_any"]) * float(row["xpts"])
            reason = (
                "vice-captain: highest captain-nonplay coverage value "
                f"({coverage:.3f}; P(play_any)={row['p_play_any']:.3f})"
            )
        elif row["squad_role"] == "STARTER":
            reason = f"starter: maximum-xPts legal XI (xPts={row['xpts']:.3f})"
        else:
            reason = (
                f"bench {int(row['bench_order'])}: expected substitute value "
                f"{row['expected_substitute_value']:.3f}"
            )
        reasons[index] = reason
    return pd.Series(reasons)
