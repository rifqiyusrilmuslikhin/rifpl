"""Historical, GW-paired decision-regret evaluation without model refitting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.decision.engine import (
    DecisionEngine,
    DecisionInputError,
    PredictionColumns,
    select_legal_starting_xi,
    validate_squad,
)
from fpl_model.decision.rules import FPLRules, load_fpl_rules
from fpl_model.decision.team import TeamState

REGRET_COLUMNS = ("xi_regret", "captain_regret", "bench_order_regret")
POINT_COLUMNS = (
    "starting_xi_points",
    "autosub_points",
    "captain_bonus_points",
    "decision_points",
)


@dataclass(frozen=True, slots=True)
class DecisionRegretReport:
    """Per-squad evidence, policy summaries, and exact paired deltas."""

    by_gameweek: pd.DataFrame
    summary: pd.DataFrame
    paired_deltas: pd.DataFrame
    champion_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "champion_policy": self.champion_policy,
            "by_gameweek": self.by_gameweek.to_dict(orient="records"),
            "summary": self.summary.to_dict(orient="records"),
            "paired_deltas": self.paired_deltas.to_dict(orient="records"),
        }


def decision_regret_backtest(
    frame: pd.DataFrame,
    *,
    champion_prediction: str,
    baselines: Mapping[str, str] | None = None,
    champion_policy: str = "champion",
    columns: PredictionColumns | None = None,
    season_column: str = "season",
    gameweek_column: str = "gameweek",
    squad_id_column: str = "squad_id",
    actual_points_column: str = "actual_points_gw",
    actual_minutes_column: str = "actual_minutes_gw",
    rules: FPLRules | None = None,
) -> DecisionRegretReport:
    """Evaluate legal decisions against hindsight and paired ranking baselines.

    Each input group must be one complete historical 15-player squad. Models are only read as
    retained prediction columns; this function never trains, tunes, or filters on outcomes.
    """
    mapping = columns or PredictionColumns()
    active_rules = rules or load_fpl_rules()
    required = {
        season_column,
        gameweek_column,
        squad_id_column,
        actual_points_column,
        actual_minutes_column,
        champion_prediction,
        mapping.element_id,
        mapping.position,
        mapping.club_id,
        mapping.p_play_any,
        mapping.p_minutes_60,
        mapping.expected_minutes,
    }
    policies = {champion_policy: champion_prediction, **dict(baselines or {})}
    if len(policies) != 1 + len(baselines or {}):
        raise DecisionInputError("champion and baseline policy names must be unique")
    required.update(policies.values())
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DecisionInputError(f"decision backtest frame is missing columns {missing!r}")
    if frame.empty:
        raise DecisionInputError("decision backtest frame must not be empty")
    keys = [season_column, gameweek_column, squad_id_column, mapping.element_id]
    if frame[keys].isna().any(axis=None) or frame.duplicated(keys).any():
        raise DecisionInputError("decision backtest requires unique, non-missing squad-player keys")
    actual = frame[[actual_points_column, actual_minutes_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(actual.to_numpy(dtype=float)).all():
        raise DecisionInputError("actual points and minutes must be finite")
    if (actual[actual_minutes_column] < 0).any():
        raise DecisionInputError("actual minutes must be non-negative")

    rows: list[dict[str, Any]] = []
    group_columns = [season_column, gameweek_column, squad_id_column]
    for group_key, group in frame.groupby(group_columns, sort=True, observed=True, dropna=False):
        season, gameweek, squad_id = group_key
        team = TeamState.from_element_ids(
            group[mapping.element_id].tolist(), team_id=str(squad_id), gameweek=int(gameweek)
        )
        for policy, prediction in policies.items():
            policy_columns = PredictionColumns(
                element_id=mapping.element_id,
                player_key=mapping.player_key,
                player_name=mapping.player_name,
                position=mapping.position,
                club_id=mapping.club_id,
                xpts=prediction,
                p_play_any=mapping.p_play_any,
                p_minutes_60=mapping.p_minutes_60,
                expected_minutes=mapping.expected_minutes,
                data_quality=mapping.data_quality,
            )
            decision = DecisionEngine(active_rules).decide(team, group, columns=policy_columns)
            rows.append(
                _score_decision(
                    decision.player_table,
                    group,
                    team=team,
                    policy=policy,
                    season=season,
                    prediction_column=prediction,
                    mapping=policy_columns,
                    actual_points_column=actual_points_column,
                    actual_minutes_column=actual_minutes_column,
                    rules=active_rules,
                )
            )

    by_gameweek = (
        pd.DataFrame(rows)
        .sort_values(["season", "gameweek", "squad_id", "policy"], kind="stable")
        .reset_index(drop=True)
    )
    summary = _summarise(by_gameweek)
    paired = _paired_deltas(by_gameweek, champion_policy)
    return DecisionRegretReport(by_gameweek, summary, paired, champion_policy)


def _score_decision(
    decision_table: pd.DataFrame,
    source_group: pd.DataFrame,
    *,
    team: TeamState,
    policy: str,
    season: object,
    prediction_column: str,
    mapping: PredictionColumns,
    actual_points_column: str,
    actual_minutes_column: str,
    rules: FPLRules,
) -> dict[str, Any]:
    evidence_columns = [mapping.element_id, actual_points_column, actual_minutes_column]
    evidence = source_group[evidence_columns].rename(columns={mapping.element_id: "element_id"})
    scored = decision_table.copy()
    indexed_evidence = evidence.set_index("element_id")
    for actual_column in (actual_points_column, actual_minutes_column):
        scored[actual_column] = scored["element_id"].map(indexed_evidence[actual_column])
    starters = scored.loc[scored["squad_role"] == "STARTER"]
    starting_points = float(starters[actual_points_column].sum())
    autosub_points, substitutes = _auto_sub_points(
        scored,
        actual_points_column=actual_points_column,
        actual_minutes_column=actual_minutes_column,
        rules=rules,
    )
    optimal_autosub = max(
        _auto_sub_points(
            _with_bench_permutation(scored, order),
            actual_points_column=actual_points_column,
            actual_minutes_column=actual_minutes_column,
            rules=rules,
        )[0]
        for order in permutations(
            scored.loc[
                (scored["squad_role"] == "BENCH") & (scored["position"] != "GKP"),
                "player_key",
            ].tolist()
        )
    )

    captain = starters.loc[starters["is_captain"]].iloc[0]
    vice = starters.loc[starters["is_vice_captain"]].iloc[0]
    captain_played = float(captain[actual_minutes_column]) > 0
    vice_played = float(vice[actual_minutes_column]) > 0
    if captain_played:
        captain_bonus = float(captain[actual_points_column])
    elif vice_played:
        captain_bonus = float(vice[actual_points_column])
    else:
        captain_bonus = 0.0
    eligible_captains = starters.loc[starters[actual_minutes_column] > 0, actual_points_column]
    hindsight_captain = float(eligible_captains.max()) if not eligible_captains.empty else 0.0

    normalised = validate_squad(team, source_group, columns=mapping, rules=rules)
    normalised["hindsight_points"] = normalised["element_id"].map(
        indexed_evidence[actual_points_column]
    )
    optimal_indices = select_legal_starting_xi(
        normalised, rules=rules, score_column="hindsight_points"
    )
    hindsight_xi = float(normalised.loc[list(optimal_indices), "hindsight_points"].sum())
    decision_points = starting_points + autosub_points + captain_bonus
    return {
        "season": season,
        "gameweek": int(team.gameweek),
        "squad_id": team.team_id,
        "policy": policy,
        "prediction_column": prediction_column,
        "formation": _formation(starters),
        "starting_xi_points": starting_points,
        "hindsight_xi_points": hindsight_xi,
        "xi_regret": hindsight_xi - starting_points,
        "autosub_points": autosub_points,
        "optimal_bench_autosub_points": optimal_autosub,
        "bench_order_regret": optimal_autosub - autosub_points,
        "captain_player_key": str(captain["player_key"]),
        "vice_captain_player_key": str(vice["player_key"]),
        "captain_bonus_points": captain_bonus,
        "hindsight_captain_points": hindsight_captain,
        "captain_regret": hindsight_captain - captain_bonus,
        "vice_activated": bool(not captain_played),
        "vice_covered": bool(not captain_played and vice_played),
        "vice_points_when_needed": float(vice[actual_points_column])
        if not captain_played and vice_played
        else 0.0,
        "autosub_player_keys": "|".join(substitutes),
        "decision_points": decision_points,
    }


def _auto_sub_points(
    table: pd.DataFrame,
    *,
    actual_points_column: str,
    actual_minutes_column: str,
    rules: FPLRules,
) -> tuple[float, list[str]]:
    starters = table.loc[table["squad_role"] == "STARTER"].copy()
    bench = table.loc[table["squad_role"] == "BENCH"].sort_values("bench_order")
    unavailable = starters.loc[starters[actual_minutes_column] <= 0].copy()
    substitutions: list[str] = []
    points = 0.0

    missing_goalkeepers = unavailable.loc[unavailable["position"] == "GKP"]
    reserve_goalkeepers = bench.loc[
        (bench["position"] == "GKP") & (bench[actual_minutes_column] > 0)
    ]
    if not missing_goalkeepers.empty and not reserve_goalkeepers.empty:
        reserve = reserve_goalkeepers.iloc[0]
        points += float(reserve[actual_points_column])
        substitutions.append(str(reserve["player_key"]))

    missing = unavailable.loc[unavailable["position"] != "GKP"].copy()
    formation = Counter(starters["position"])
    for _, candidate in bench.loc[
        (bench["position"] != "GKP") & (bench[actual_minutes_column] > 0)
    ].iterrows():
        for missing_index, absent in missing.sort_values("player_key", kind="stable").iterrows():
            candidate_counts = dict(formation)
            candidate_counts[str(absent["position"])] -= 1
            candidate_counts[str(candidate["position"])] = (
                candidate_counts.get(str(candidate["position"]), 0) + 1
            )
            if rules.legal_formation(candidate_counts):
                formation = Counter(candidate_counts)
                missing = missing.drop(index=missing_index)
                points += float(candidate[actual_points_column])
                substitutions.append(str(candidate["player_key"]))
                break
        if missing.empty:
            break
    return points, substitutions


def _with_bench_permutation(table: pd.DataFrame, player_keys: tuple[str, ...]) -> pd.DataFrame:
    result = table.copy()
    for order, key in enumerate(player_keys, start=1):
        result.loc[result["player_key"] == key, "bench_order"] = order
    return result


def _formation(starters: pd.DataFrame) -> str:
    counts = Counter(starters["position"])
    return "-".join(str(counts[position]) for position in ("DEF", "MID", "FWD"))


def _summarise(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [*REGRET_COLUMNS, *POINT_COLUMNS, "vice_activated", "vice_covered"]
    summary = frame.groupby("policy", sort=True)[metrics].mean().reset_index()
    counts = frame.groupby("policy", sort=True).size().rename("decision_count").reset_index()
    summary = summary.merge(counts, on="policy", validate="one_to_one")
    summary["vice_coverage_rate_when_needed"] = (
        frame.groupby("policy", sort=True)
        .apply(
            lambda group: (
                float(group.loc[group["vice_activated"], "vice_covered"].mean())
                if group["vice_activated"].any()
                else float("nan")
            ),
            include_groups=False,
        )
        .to_numpy()
    )
    return summary


def _paired_deltas(frame: pd.DataFrame, champion_policy: str) -> pd.DataFrame:
    identity = ["season", "gameweek", "squad_id"]
    champion = frame.loc[frame["policy"] == champion_policy]
    if champion.empty:
        raise DecisionInputError(f"champion policy {champion_policy!r} has no decisions")
    metrics = [*REGRET_COLUMNS, *POINT_COLUMNS]
    rows = []
    for baseline_name, baseline in frame.loc[frame["policy"] != champion_policy].groupby(
        "policy", sort=True
    ):
        paired = champion[identity + metrics].merge(
            baseline[identity + metrics],
            on=identity,
            suffixes=("_champion", "_baseline"),
            validate="one_to_one",
        )
        if len(paired) != len(champion) or len(paired) != len(baseline):
            raise DecisionInputError(f"policy {baseline_name!r} is not exactly paired")
        for metric in metrics:
            deltas = paired[f"{metric}_champion"] - paired[f"{metric}_baseline"]
            rows.append(
                {
                    "baseline_policy": str(baseline_name),
                    "metric": metric,
                    "champion_minus_baseline": float(deltas.mean()),
                    "gameweek_count": len(deltas),
                    "better_direction": "lower" if metric in REGRET_COLUMNS else "higher",
                }
            )
    return pd.DataFrame(rows)
