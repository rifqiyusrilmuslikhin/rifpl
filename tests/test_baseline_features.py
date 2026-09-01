"""Acceptance tests for the fixed causal Sprint 4 feature contract."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fpl_model.features import (
    BASELINE_FEATURE_CONTRACT,
    BASELINE_FEATURE_NAMES,
    BaselineFeatureBuilder,
    FeatureBuildError,
    build_coverage_report,
    write_coverage_report,
    write_spot_check_report,
)

SEASON = "2026-27"
DEADLINE = datetime(2026, 8, 30, 10, tzinfo=UTC)


def _target(
    player_key: str = "player-a",
    *,
    team: int = 1,
    fixtures: int = 2,
    status: str = "a",
    understat_id: str | None = "understat-a",
) -> dict[str, object]:
    return {
        "season": SEASON,
        "gameweek": 4,
        "player_key": player_key,
        "deadline_utc": DEADLINE,
        "snapshot_captured_at_utc": DEADLINE - timedelta(hours=2),
        "feature_cutoff_utc": DEADLINE - timedelta(hours=1),
        "team_id_at_deadline": team,
        "position_at_deadline": "MID",
        "fixture_count": fixtures,
        "home_fixture_count": 1 if fixtures else 0,
        "fixture_difficulty_mean": 2.5 if fixtures == 2 else 3.0 if fixtures else None,
        "fixture_difficulty_min": 2.0 if fixtures else None,
        "now_cost": 75,
        "status": status,
        "status_value_state": "value",
        "chance_of_playing_next_round": 50 if status != "a" else None,
        "chance_of_playing_value_state": "value" if status != "a" else "source_unavailable",
        "understat_id": understat_id,
        "source_artifact_ids": ["snapshot-gw4", "fixtures-gw4"],
    }


def _contexts(
    player_key: str = "player-a", *, team: int = 1, opponents: tuple[int, ...] = (2, 3)
) -> list[dict[str, object]]:
    rows = []
    for index, opponent in enumerate(opponents):
        rows.append(
            {
                "season": SEASON,
                "gameweek": 4,
                "player_key": player_key,
                "fixture_id": 9000 + index,
                "kickoff_utc": DEADLINE + timedelta(days=1 + index * 3),
                "deadline_utc": DEADLINE,
                "feature_cutoff_utc": DEADLINE - timedelta(hours=1),
                "team_id_at_deadline": team,
                "opponent_team_id": opponent,
                "was_home": index == 0,
                "available_at_utc": DEADLINE - timedelta(days=20),
                "snapshot_source_artifact_id": "snapshot-gw4",
                "fixture_source_artifact_id": "fixtures-gw4",
            }
        )
    return rows


def _player_history(player_key: str = "player-a", *, team: int = 1) -> list[dict[str, object]]:
    minutes = [30, 60, 90, 0, 90, 45]
    rows: list[dict[str, object]] = []
    for index, played in enumerate(minutes, start=1):
        rows.append(
            {
                "season": SEASON,
                "fixture_id": 1000 + index,
                "player_key": player_key,
                "kickoff_utc": DEADLINE - timedelta(days=28 - index * 4),
                "team_id": team if index > 2 else 8,
                "was_home": index % 2 == 0,
                "minutes": played,
                "total_points": index,
                "goals_scored": int(index == 5),
                "assists": int(index == 6),
                "starts": int(played >= 60),
                "bps": index * 2,
                "bonus": int(index >= 5),
                "yellow_cards": int(index == 4),
                "xg": index / 10,
                "xa": index / 20,
                "shots": index,
                "key_passes": index - 1,
                "source_artifact_id": f"player-history-{index}",
            }
        )
    return rows


def _team_history() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for team in (1, 2, 3, 4):
        for index in range(1, 7):
            rows.append(
                {
                    "season": SEASON,
                    "fixture_id": team * 100 + index,
                    "kickoff_utc": DEADLINE - timedelta(days=42 - index * 5),
                    "team_id": team,
                    "opponent_team_id": 20 + index,
                    "goals_for": team + index % 2,
                    "goals_against": index % 3,
                    "xg": team + index / 10,
                    "xga": index / 5,
                    "source_artifact_id": f"team-{team}-history",
                }
            )
    return rows


def _build(
    targets: list[dict[str, object]] | None = None,
    contexts: list[dict[str, object]] | None = None,
    player_history: list[dict[str, object]] | None = None,
    team_history: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    return BaselineFeatureBuilder().build(
        targets or [_target()],
        fixture_contexts=contexts if contexts is not None else _contexts(),
        player_history=player_history if player_history is not None else _player_history(),
        team_history=team_history if team_history is not None else _team_history(),
    )


def test_machine_readable_contract_has_exactly_the_declared_46_features() -> None:
    assert len(BASELINE_FEATURE_NAMES) == len(set(BASELINE_FEATURE_NAMES)) == 46
    assert BASELINE_FEATURE_CONTRACT.feature_names == BASELINE_FEATURE_NAMES
    assert all(
        all(value for value in definition.to_dict().values())
        for definition in BASELINE_FEATURE_CONTRACT.features
    )
    assert "snapshot_captured_at_utc" in _build().columns


def test_player_rolls_are_shifted_and_dgw_opponents_are_aggregated_once() -> None:
    frame = _build()
    row = frame.iloc[0]

    assert list(frame.columns[-46:]) == list(BASELINE_FEATURE_NAMES)
    assert not set(frame.columns).intersection({"y_points", "actual_points_gw"})
    assert str(frame["position"].dtype) == "category"
    assert str(frame["price"].dtype) == "float64"
    assert str(frame["fixture_count"].dtype) == "int64"
    assert row["minutes_mean_last3"] == pytest.approx((0 + 90 + 45) / 3)
    assert row["minutes_mean_last5"] == pytest.approx((60 + 90 + 0 + 90 + 45) / 5)
    assert row["play_any_rate_last5"] == pytest.approx(4 / 5)
    assert row["history_appearances_count"] == 5
    assert row["points_home_away_relevant_mean_last5"] == pytest.approx(3.5)
    # Team 2 and team 3 each contribute one deadline-locked last-5 rate to the DGW mean.
    assert row["opp_goals_for_mean_last5"] == pytest.approx(2.9)
    assert row["attack_matchup"] == pytest.approx(
        row["team_goals_for_mean_last5"] * row["opp_goals_against_mean_last5"]
    )


def test_understat_unresolved_is_missing_not_zero() -> None:
    target = _target(understat_id=None)
    frame = _build(targets=[target])
    row = frame.iloc[0]

    assert pd.isna(row["xg_per90_last5"])
    assert pd.isna(row["xa_per90_last5"])
    assert pd.isna(row["shots_per90_last5"])
    assert pd.isna(row["key_passes_per90_last5"])
    assert row["goals_per90_last5"] > 0


def test_genuine_zero_and_source_unavailable_remain_distinct() -> None:
    missing = _target()
    missing_frame = _build(targets=[missing])

    genuine_zero = _target(status="d")
    genuine_zero["chance_of_playing_next_round"] = 0
    genuine_zero["chance_of_playing_value_state"] = "genuine_zero"
    zero_history = _player_history()
    for row in zero_history:
        row.update(xg=0, xa=0, shots=0, key_passes=0)
    zero_frame = _build(targets=[genuine_zero], player_history=zero_history)

    assert pd.isna(missing_frame.iloc[0]["chance_of_playing"])
    assert zero_frame.iloc[0]["chance_of_playing"] == 0.0
    assert zero_frame.iloc[0]["xg_per90_last5"] == 0.0
    assert zero_frame.iloc[0]["xa_per90_last5"] == 0.0


def test_inconsistent_missing_value_state_fails_fast() -> None:
    target = _target()
    target["chance_of_playing_next_round"] = 0

    with pytest.raises(FeatureBuildError, match="must be missing"):
        _build(targets=[target])


def test_target_and_post_deadline_rows_never_enter_a_player_window() -> None:
    history = _player_history()
    future = deepcopy(history[-1])
    future.update(
        fixture_id=9999,
        kickoff_utc=DEADLINE + timedelta(hours=2),
        completed_at_utc=DEADLINE + timedelta(hours=5),
        total_points=999,
        minutes=90,
        xg=999,
    )
    before = _build(player_history=history)
    after = _build(player_history=[future, *reversed(history)])

    assert_frame_equal(before, after, check_like=False)


def test_mutating_all_post_deadline_information_cannot_change_feature_frame() -> None:
    future_players: list[dict[str, object]] = []
    for index in range(2):
        row = deepcopy(_player_history()[-1])
        row.update(
            fixture_id=9900 + index,
            kickoff_utc=DEADLINE + timedelta(days=index + 1),
            completed_at_utc=DEADLINE + timedelta(days=index + 1, hours=3),
            available_at_utc=DEADLINE + timedelta(days=index + 1, hours=4),
            source_artifact_id=f"post-deadline-player-{index}",
        )
        future_players.append(row)
    future_teams: list[dict[str, object]] = []
    for team in (1, 2, 3):
        future_teams.append(
            {
                "season": SEASON,
                "fixture_id": 9800 + team,
                "kickoff_utc": DEADLINE + timedelta(days=team),
                "completed_at_utc": DEADLINE + timedelta(days=team, hours=3),
                "available_at_utc": DEADLINE + timedelta(days=team, hours=4),
                "team_id": team,
                "opponent_team_id": 20,
                "goals_for": 1,
                "goals_against": 1,
                "xg": 1.0,
                "xga": 1.0,
                "source_artifact_id": f"post-deadline-team-{team}",
            }
        )
    before_target = _target()
    before_target.update(actual_minutes_gw=30, actual_points_gw=2, y_minutes=30, y_points=2)
    before = _build(
        targets=[before_target],
        player_history=[*_player_history(), *future_players],
        team_history=[*_team_history(), *future_teams],
    )

    mutated_players = deepcopy(future_players)
    for row in mutated_players:
        row.update(
            minutes=180,
            total_points=999,
            goals_scored=99,
            assists=99,
            starts=99,
            bps=999,
            bonus=99,
            yellow_cards=99,
            xg=999,
            xa=999,
            shots=999,
            key_passes=999,
            source_artifact_id="mutated-post-deadline-player",
        )
    mutated_teams = deepcopy(future_teams)
    for row in mutated_teams:
        row.update(
            goals_for=99,
            goals_against=88,
            xg=77,
            xga=66,
            source_artifact_id="mutated-post-deadline-team",
        )
    after_target = deepcopy(before_target)
    after_target.update(actual_minutes_gw=180, actual_points_gw=200, y_minutes=180, y_points=200)
    after = _build(
        targets=[after_target],
        player_history=[*_player_history(), *mutated_players],
        team_history=[*_team_history(), *mutated_teams],
    )

    assert_frame_equal(before, after, check_exact=True, check_like=False)


def test_history_available_at_deadline_is_excluded_and_bad_context_fails() -> None:
    history = _player_history()
    history[-1]["available_at_utc"] = DEADLINE
    frame = _build(player_history=history)
    assert frame.iloc[0]["minutes_mean_last3"] == pytest.approx((90 + 0 + 90) / 3)

    contexts = _contexts()
    contexts[1]["feature_cutoff_utc"] = DEADLINE
    with pytest.raises(FeatureBuildError, match="strictly pre-deadline"):
        _build(contexts=contexts)


def test_post_deadline_snapshot_provenance_is_rejected_directly() -> None:
    target = _target()
    target["snapshot_captured_at_utc"] = DEADLINE

    with pytest.raises(FeatureBuildError, match="snapshot_captured_at_utc"):
        _build(targets=[target])


def test_later_season_history_cannot_change_earlier_feature_frame() -> None:
    later = deepcopy(_player_history()[-1])
    later.update(
        season="2027-28",
        fixture_id=7777,
        kickoff_utc=DEADLINE - timedelta(days=1),
        completed_at_utc=DEADLINE - timedelta(hours=20),
        available_at_utc=DEADLINE - timedelta(hours=19),
        minutes=90,
        total_points=999,
    )

    before = _build()
    after = _build(player_history=[later, *_player_history()])

    assert_frame_equal(before, after, check_exact=True, check_like=False)


def test_representative_player_cases_and_reports_are_retained(tmp_path: object) -> None:
    output_path = tmp_path  # Keep the annotation independent of pathlib fixture internals.
    cases = {
        "single_gameweek": _target("sgw", fixtures=1),
        "double_gameweek": _target("dgw", fixtures=2),
        "debutant": _target("debut", fixtures=1),
        "flagged_injury": _target("injury", fixtures=1, status="i"),
        "club_transfer": _target("transfer", team=4, fixtures=1),
    }
    contexts = [
        *_contexts("sgw", opponents=(2,)),
        *_contexts("dgw", opponents=(2, 3)),
        *_contexts("debut", opponents=(2,)),
        *_contexts("injury", opponents=(2,)),
        *_contexts("transfer", team=4, opponents=(2,)),
    ]
    histories: list[dict[str, object]] = []
    for player in ("sgw", "dgw", "injury"):
        histories.extend(_player_history(player))
    histories.extend(_player_history("transfer", team=8))
    frame = _build(
        targets=list(cases.values()),
        contexts=contexts,
        player_history=histories,
    )
    keyed = frame.set_index("player_key")

    assert keyed.loc["sgw", "fixture_count"] == 1
    assert keyed.loc["dgw", "fixture_count"] == 2
    assert keyed.loc["debut", "history_appearances_count"] == 0
    assert pd.isna(keyed.loc["debut", "minutes_mean_last5"])
    assert keyed.loc["injury", "status_risk_ordinal"] == 2
    assert keyed.loc["transfer", "team_goals_for_mean_last5"] == pytest.approx(4.4)

    coverage = build_coverage_report(frame)
    assert len(coverage) == 46
    assert set(coverage["season"]) == {SEASON}
    coverage_path = write_coverage_report(frame, output_path / "coverage.json")  # type: ignore[operator]
    selections = {name: (SEASON, 4, target["player_key"]) for name, target in cases.items()}
    spot_path = write_spot_check_report(
        frame,
        selections,
        output_path / "spot-checks.json",  # type: ignore[operator]
    )
    assert json.loads(coverage_path.read_text(encoding="utf-8"))[0]["season"] == SEASON
    assert len(json.loads(spot_path.read_text(encoding="utf-8"))["spot_checks"]) == 5
