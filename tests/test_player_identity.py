"""Tests for cross-season FPL identity and audited Understat mappings."""

import pytest

from fpl_model.data.identity import (
    IdentityConflictError,
    MatchConfidence,
    MatchMethod,
    PlayerIdentityRegistry,
    propose_understat_matches,
)


def _fpl_player(*, code: int, element: int, team: int, name: str = "Example") -> dict[str, object]:
    return {"code": code, "id": element, "team": team, "web_name": name}


def test_same_element_id_in_different_seasons_does_not_join_different_players() -> None:
    registry = PlayerIdentityRegistry()
    old = registry.register_fpl_rows(
        season="2025-26",
        gameweek=1,
        rows=[_fpl_player(code=5001, element=101, team=1)],
        source_artifact_id="old-bootstrap",
    )[0]
    new = registry.register_fpl_rows(
        season="2026-27",
        gameweek=1,
        rows=[_fpl_player(code=9009, element=101, team=2)],
        source_artifact_id="new-bootstrap",
    )[0]

    assert old["player_key"] != new["player_key"]
    assert (
        registry.resolve_fpl_alias(season="2025-26", fpl_element_id=101, gameweek=1)
        == old["player_key"]
    )
    assert (
        registry.resolve_fpl_alias(season="2026-27", fpl_element_id=101, gameweek=1)
        == new["player_key"]
    )


def test_fpl_code_reuses_identity_across_season_specific_element_aliases() -> None:
    registry = PlayerIdentityRegistry()
    old = registry.register_fpl_rows(
        season="2025-26",
        gameweek=1,
        rows=[_fpl_player(code=5001, element=101, team=1)],
        source_artifact_id="old-bootstrap",
    )[0]
    new = registry.register_fpl_rows(
        season="2026-27",
        gameweek=1,
        rows=[_fpl_player(code=5001, element=333, team=4)],
        source_artifact_id="new-bootstrap",
    )[0]

    assert old["player_key"] == new["player_key"]
    assert registry.player_key_for_code(5001) == old["player_key"]


def test_club_transfer_versions_team_without_changing_identity() -> None:
    registry = PlayerIdentityRegistry()
    first = registry.register_fpl_rows(
        season="2026-27",
        gameweek=1,
        rows=[_fpl_player(code=5001, element=101, team=1)],
        source_artifact_id="gw1-bootstrap",
    )[0]
    player_key = first["player_key"]
    registry.assign_understat(
        player_key=player_key,
        season="2026-27",
        valid_from_gw=1,
        understat_id="u-5001",
        match_method=MatchMethod.EXACT_ID,
        confidence=MatchConfidence.HIGH,
        audit_note="Stable source ID checked against source profile.",
        source_artifact_id="understat-audit",
    )
    transferred = registry.register_fpl_rows(
        season="2026-27",
        gameweek=10,
        rows=[_fpl_player(code=5001, element=101, team=8)],
        source_artifact_id="gw10-bootstrap",
    )[0]

    assert transferred["player_key"] == player_key
    assert registry.team_at(player_key=player_key, season="2026-27", gameweek=9) == 1
    assert registry.team_at(player_key=player_key, season="2026-27", gameweek=10) == 8
    assert (
        registry.understat_id_at(player_key=player_key, season="2026-27", gameweek=10) == "u-5001"
    )
    records = registry.to_records()
    assert [(row["valid_from_gw"], row["valid_to_gw"]) for row in records] == [
        (1, 9),
        (10, None),
    ]


def test_ambiguous_understat_name_stays_unresolved() -> None:
    registry = PlayerIdentityRegistry()
    resolved = registry.register_fpl_rows(
        season="2026-27",
        gameweek=1,
        rows=[_fpl_player(code=5001, element=101, team=1, name="Joao Pedro")],
        source_artifact_id="bootstrap",
    )[0]
    proposals = propose_understat_matches(
        [{"player_key": resolved["player_key"], "web_name": "João Pedro"}],
        [
            {"understat_id": "u-1", "name": "Joao Pedro"},
            {"understat_id": "u-2", "name": "João Pedro"},
        ],
    )

    assert proposals[0].status == "ambiguous"
    assert proposals[0].candidate_ids == ("u-1", "u-2")
    assert (
        registry.understat_id_at(player_key=resolved["player_key"], season="2026-27", gameweek=1)
        is None
    )
    unresolved = registry.to_records()[0]
    assert unresolved["match_method"] == "unresolved"
    assert unresolved["confidence"] == "unresolved"


def test_external_id_cannot_silently_attach_to_two_players() -> None:
    registry = PlayerIdentityRegistry()
    players = registry.register_fpl_rows(
        season="2026-27",
        gameweek=1,
        rows=[
            _fpl_player(code=5001, element=101, team=1),
            _fpl_player(code=5002, element=102, team=2),
        ],
        source_artifact_id="bootstrap",
    )
    arguments = {
        "season": "2026-27",
        "valid_from_gw": 1,
        "understat_id": "shared-external-id",
        "match_method": MatchMethod.MANUAL,
        "confidence": MatchConfidence.HIGH,
        "audit_note": "Manually checked profile and date of birth.",
        "source_artifact_id": "identity-audit",
    }
    registry.assign_understat(player_key=players[0]["player_key"], **arguments)

    with pytest.raises(IdentityConflictError, match="another player_key"):
        registry.assign_understat(player_key=players[1]["player_key"], **arguments)


def test_manual_or_name_dob_mapping_requires_audit_evidence() -> None:
    registry = PlayerIdentityRegistry()
    player = registry.register_fpl_rows(
        season="2026-27",
        gameweek=1,
        rows=[_fpl_player(code=5001, element=101, team=1)],
        source_artifact_id="bootstrap",
    )[0]

    with pytest.raises(IdentityConflictError, match="evidence"):
        registry.assign_understat(
            player_key=player["player_key"],
            season="2026-27",
            valid_from_gw=1,
            understat_id="u-5001",
            match_method=MatchMethod.AUDITED_NAME_DOB,
            confidence=MatchConfidence.MEDIUM,
            audit_note=None,
            source_artifact_id="identity-audit",
        )
