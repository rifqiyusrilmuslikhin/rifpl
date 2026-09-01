"""Acceptance tests for fixture facts, DGW anchoring, blanks, and reconciliation."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fpl_model.data.canonical import (
    CanonicalDatasetBuilder,
    CanonicalDatasetError,
    DGWAnchorError,
    validate_dgw_anchors,
)
from fpl_model.data.identity import PlayerIdentityRegistry

SEASON = "2026-27"
DEADLINE = datetime(2026, 8, 15, 10, 0, tzinfo=UTC)


def _registry() -> tuple[PlayerIdentityRegistry, dict[int, str]]:
    registry = PlayerIdentityRegistry()
    rows = registry.register_fpl_rows(
        season=SEASON,
        gameweek=1,
        rows=[
            {"code": 5001, "id": 101, "team": 1},
            {"code": 5002, "id": 102, "team": 4},
        ],
        source_artifact_id="bootstrap-gw1",
    )
    return registry, {row["code"]: row["player_key"] for row in rows}


def _snapshot(player_key: str, *, code: int, element: int, team: int) -> dict[str, object]:
    return {
        "season": SEASON,
        "gameweek": 1,
        "player_key": player_key,
        "fpl_code": code,
        "fpl_element_id": element,
        "deadline_utc": DEADLINE,
        "captured_at_utc": DEADLINE - timedelta(hours=2),
        "team_id": team,
        "position": "MID",
        "now_cost": 75,
        "status": "a",
        "status_value_state": "value",
        "chance_of_playing_next_round": None,
        "chance_of_playing_value_state": "source_unavailable",
        "ep_next": 4.5,
        "ep_next_value_state": "value",
        "source_artifact_id": "bootstrap-gw1",
    }


def _fixtures() -> list[dict[str, object]]:
    return [
        {
            "season": SEASON,
            "fixture_id": 9001,
            "gameweek": 1,
            "kickoff_utc": DEADLINE + timedelta(hours=3),
            "home_team_id": 1,
            "away_team_id": 2,
            "home_difficulty": 2,
            "away_difficulty": 4,
            "available_at_utc": DEADLINE - timedelta(days=30),
            "source_artifact_id": "fixtures-pre-deadline",
        },
        {
            "season": SEASON,
            "fixture_id": 9002,
            "gameweek": 1,
            "kickoff_utc": DEADLINE + timedelta(days=4),
            "home_team_id": 3,
            "away_team_id": 1,
            "home_difficulty": 4,
            "away_difficulty": 3,
            "available_at_utc": DEADLINE - timedelta(days=7),
            "source_artifact_id": "fixtures-pre-deadline",
        },
    ]


def _outcomes(points: tuple[int, int] = (5, 7)) -> list[dict[str, object]]:
    return [
        {
            "season": SEASON,
            "fixture_id": 9001,
            "element": 101,
            "total_points": points[0],
            "minutes": 90,
            "source_artifact_id": "gw-live-final",
            "source_row_number": 1,
        },
        {
            "season": SEASON,
            "fixture_id": 9002,
            "element": 101,
            "total_points": points[1],
            "minutes": 60,
            "source_artifact_id": "gw-live-final",
            "source_row_number": 2,
        },
    ]


def _snapshots(keys: dict[int, str]) -> list[dict[str, object]]:
    return [
        _snapshot(keys[5001], code=5001, element=101, team=1),
        _snapshot(keys[5002], code=5002, element=102, team=4),
    ]


def test_dgw_aggregates_once_and_blank_is_explicit() -> None:
    registry, keys = _registry()
    dataset = CanonicalDatasetBuilder(registry).build(
        outcomes=_outcomes(),
        fixtures=_fixtures(),
        snapshots=_snapshots(keys),
    )

    assert len(dataset.fixture_facts) == 2
    assert len(dataset.player_gameweeks) == 2
    assert (
        len(
            {
                (row["season"], row["gameweek"], row["player_key"])
                for row in dataset.player_gameweeks
            }
        )
        == 2
    )

    dgw = next(row for row in dataset.player_gameweeks if row["player_key"] == keys[5001])
    assert dgw["fixture_count"] == 2
    assert dgw["fixture_ids"] == ["9001", "9002"]
    assert dgw["actual_points_gw"] == 12
    assert dgw["actual_minutes_gw"] == 150
    assert dgw["y_play_any"] == 1
    assert dgw["y_minutes_60"] == 1
    assert dgw["fixture_difficulty_mean"] == 2.5

    blank = next(row for row in dataset.player_gameweeks if row["player_key"] == keys[5002])
    assert blank["is_blank"] is True
    assert blank["fixture_count"] == 0
    assert blank["fixture_ids"] == []
    assert blank["actual_points_gw"] == 0
    assert blank["actual_minutes_gw"] == 0
    assert blank["y_points_if_play"] is None

    report = dataset.reconciliation
    assert report.status == "passed"
    assert report.total_fixture_points == report.total_gameweek_points == 12
    assert report.total_fixture_minutes == report.total_gameweek_minutes == 150
    assert report.dgw_player_gameweek_rows == 1
    assert report.blank_player_gameweek_rows == 1


def test_dgw_outcome_mutation_cannot_change_frozen_context() -> None:
    registry, keys = _registry()
    builder = CanonicalDatasetBuilder(registry)
    before = builder.build(
        outcomes=_outcomes((5, 7)),
        fixtures=_fixtures(),
        snapshots=_snapshots(keys),
    )
    after = builder.build(
        outcomes=_outcomes((99, 7)),
        fixtures=_fixtures(),
        snapshots=_snapshots(keys),
    )

    assert before.fixture_contexts == after.fixture_contexts
    context_fields = set(before.player_gameweeks[0]).difference(
        {
            "actual_points_gw",
            "actual_minutes_gw",
            "y_play_any",
            "y_minutes_60",
            "y_minutes",
            "y_points",
            "y_points_if_play",
            "y_haul_5",
            "y_haul_10",
        }
    )
    before_context = [
        {field: row[field] for field in context_fields} for row in before.player_gameweeks
    ]
    after_context = [
        {field: row[field] for field in context_fields} for row in after.player_gameweeks
    ]
    assert before_context == after_context
    assert before.player_gameweeks[0]["actual_points_gw"] == 12
    assert after.player_gameweeks[0]["actual_points_gw"] == 106


def test_all_dgw_fixtures_share_deadline_anchor_not_kickoff_cutoffs() -> None:
    registry, keys = _registry()
    contexts = CanonicalDatasetBuilder(registry).build_predeadline_fixture_context(
        _snapshots(keys), _fixtures()
    )

    assert len(contexts) == 2
    assert contexts[0]["kickoff_utc"] != contexts[1]["kickoff_utc"]
    assert contexts[0]["feature_cutoff_utc"] == contexts[1]["feature_cutoff_utc"]
    assert contexts[0]["context_anchor_id"] == contexts[1]["context_anchor_id"]
    assert contexts[0]["feature_cutoff_utc"] < DEADLINE


def test_negative_kickoff_sorted_dgw_history_is_rejected() -> None:
    """A naive kickoff roll sees fixture one before fixture two and must fail the anchor gate."""
    registry, keys = _registry()
    contexts = CanonicalDatasetBuilder(registry).build_predeadline_fixture_context(
        _snapshots(keys), _fixtures()
    )
    leaked = deepcopy(contexts)
    leaked[0]["history_points_before_gw"] = 10
    leaked[1]["history_points_before_gw"] = 15
    leaked[1]["feature_cutoff_utc"] = leaked[0]["kickoff_utc"]

    with pytest.raises(DGWAnchorError, match="feature cutoff is not pre-deadline"):
        validate_dgw_anchors(leaked, history_fields=("history_points_before_gw",))


def test_history_variation_within_dgw_is_rejected_even_with_same_cutoff() -> None:
    registry, keys = _registry()
    contexts = CanonicalDatasetBuilder(registry).build_predeadline_fixture_context(
        _snapshots(keys), _fixtures()
    )
    contexts[0]["history_minutes"] = 450
    contexts[1]["history_minutes"] = 540

    with pytest.raises(DGWAnchorError, match="history_minutes"):
        validate_dgw_anchors(contexts, history_fields=("history_minutes",))


def test_transfer_and_opponent_join_use_gameweek_valid_interval() -> None:
    registry = PlayerIdentityRegistry()
    player = registry.register_fpl_rows(
        season=SEASON,
        gameweek=1,
        rows=[{"code": 5001, "id": 101, "team": 1}],
        source_artifact_id="bootstrap-gw1",
    )[0]
    registry.register_fpl_rows(
        season=SEASON,
        gameweek=10,
        rows=[{"code": 5001, "id": 101, "team": 8}],
        source_artifact_id="bootstrap-gw10",
    )
    fixtures = [
        {
            "season": SEASON,
            "fixture_id": 9010,
            "gameweek": 9,
            "kickoff_utc": DEADLINE + timedelta(days=60),
            "home_team_id": 1,
            "away_team_id": 2,
            "available_at_utc": DEADLINE - timedelta(days=1),
            "source_artifact_id": "fixtures",
        },
        {
            "season": SEASON,
            "fixture_id": 9011,
            "gameweek": 10,
            "kickoff_utc": DEADLINE + timedelta(days=67),
            "home_team_id": 3,
            "away_team_id": 8,
            "available_at_utc": DEADLINE - timedelta(days=1),
            "source_artifact_id": "fixtures",
        },
    ]
    outcomes = [
        {
            "season": SEASON,
            "fixture_id": fixture_id,
            "player_key": player["player_key"],
            "element": 101,
            "total_points": 2,
            "minutes": 45,
            "source_artifact_id": "outcomes",
        }
        for fixture_id in (9010, 9011)
    ]

    facts = CanonicalDatasetBuilder(registry).build_player_fixture_facts(outcomes, fixtures)

    assert [(row["gameweek"], row["team_id"], row["opponent_team_id"]) for row in facts] == [
        (9, 1, 2),
        (10, 8, 3),
    ]


def test_post_deadline_schedule_context_is_rejected() -> None:
    registry, keys = _registry()
    fixtures = _fixtures()
    fixtures[1]["available_at_utc"] = DEADLINE

    with pytest.raises(CanonicalDatasetError, match="not available before"):
        CanonicalDatasetBuilder(registry).build_player_gameweek_context(_snapshots(keys), fixtures)


def test_duplicate_fixture_facts_fail_instead_of_double_counting() -> None:
    registry, _ = _registry()
    duplicated = [_outcomes()[0], deepcopy(_outcomes()[0])]

    with pytest.raises(CanonicalDatasetError, match="duplicate primary key"):
        CanonicalDatasetBuilder(registry).build_player_fixture_facts(duplicated, _fixtures())


def test_fpl_json_and_historical_csv_scalar_aliases_are_normalized() -> None:
    registry, keys = _registry()
    fixture = {
        "season": SEASON,
        "id": "9001",
        "event": "1",
        "kickoff_time": "2026-08-15T13:00:00Z",
        "team_h": "1",
        "team_a": "2",
        "team_h_difficulty": "2",
        "team_a_difficulty": "4",
        "available_at_utc": DEADLINE - timedelta(days=1),
        "source_artifact_id": "fixtures-json",
    }
    outcome = {
        "season": SEASON,
        "fixture": "9001",
        "element": "101",
        "total_points": "5",
        "minutes": "90",
        "was_home": "True",
        "source_artifact_id": "history-csv",
        "source_row_number": "2",
    }

    builder = CanonicalDatasetBuilder(registry)
    facts = builder.build_player_fixture_facts([outcome], [fixture])
    contexts = builder.build_player_gameweek_context(_snapshots(keys), [fixture])

    assert facts[0]["fixture_id"] == 9001
    assert facts[0]["total_points"] == 5
    assert facts[0]["minutes"] == 90
    assert facts[0]["was_home"] is True
    assert contexts[0]["fixture_difficulty_mean"] == 2.0


def test_reconciliation_report_can_be_retained_as_json(tmp_path: Path) -> None:
    registry, keys = _registry()
    dataset = CanonicalDatasetBuilder(registry).build(
        outcomes=_outcomes(), fixtures=_fixtures(), snapshots=_snapshots(keys)
    )
    path = dataset.reconciliation.write_json(tmp_path / "reconciliation.json")
    payload = path.read_text(encoding="utf-8")

    assert '"status": "passed"' in payload
    assert '"total_gameweek_points": 12' in payload
