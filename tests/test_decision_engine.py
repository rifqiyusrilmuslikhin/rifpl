"""Offline acceptance tests for the deterministic Sprint 10 decision layer."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fpl_model.data.acquisition import HttpResponse
from fpl_model.decision import (
    DecisionInputError,
    PredictionColumns,
    PublicTeamLoader,
    TeamPayloadError,
    TeamState,
    decision_regret_backtest,
    make_decision,
    write_decision_report,
)


def _prediction_frame() -> pd.DataFrame:
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    return pd.DataFrame(
        {
            "fpl_element_id": range(1, 16),
            "player_key": [f"p{element:02d}" for element in range(1, 16)],
            "player_name": [f"Player {element}" for element in range(1, 16)],
            "position": positions,
            "team_id": [(element - 1) % 5 + 1 for element in range(1, 16)],
            "xpts": [float(element) for element in range(1, 16)],
            "p_play_any": [0.9] * 15,
            "p_minutes_60": [0.8] * 15,
            "expected_minutes": [72.0] * 15,
        }
    )


def _team() -> TeamState:
    return TeamState.from_element_ids(range(1, 16), team_id=123, gameweek=4)


def _public_payload() -> dict[str, object]:
    picks = []
    for position in range(1, 16):
        picks.append(
            {
                "element": position,
                "position": position,
                "multiplier": 2 if position == 1 else int(position <= 11),
                "is_captain": position == 1,
                "is_vice_captain": position == 2,
                "purchase_price": 50,
                "selling_price": 50,
            }
        )
    return {"picks": picks, "entry_history": {"bank": 12, "event_transfers": 1}}


def test_public_team_id_loader_validates_and_hashes_payload() -> None:
    requested: list[str] = []
    content = json.dumps(_public_payload()).encode()

    def fetcher(url: str) -> HttpResponse:
        requested.append(url)
        return HttpResponse(content, "application/json")

    state = PublicTeamLoader(fetcher, base_url="https://example.test/api/").load(123, 4)

    assert requested == ["https://example.test/api/entry/123/event/4/picks/"]
    assert state.element_ids == tuple(range(1, 16))
    assert state.bank == 12
    assert len(state.source_payload_sha256 or "") == 64


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"detail": "Not found."}, "Team ID is unavailable"),
        ({"picks": [], "entry_history": {}}, "exactly 15 picks"),
        ({"picks": "bad", "entry_history": {}}, "must be a list"),
    ],
)
def test_public_team_id_loader_rejects_invalid_payload(
    payload: dict[str, object], message: str
) -> None:
    loader = PublicTeamLoader(lambda _url: HttpResponse(json.dumps(payload).encode()))
    with pytest.raises(TeamPayloadError, match=message):
        loader.load(123, 4)


def test_starting_xi_is_legal_and_maximises_xpts() -> None:
    result = make_decision(_team(), _prediction_frame())
    counts = result.starting_xi["position"].value_counts().to_dict()

    assert len(result.starting_xi) == 11
    assert counts["GKP"] == 1
    assert 3 <= counts["DEF"] <= 5
    assert 2 <= counts["MID"] <= 5
    assert 1 <= counts["FWD"] <= 3
    assert result.formation == "3-4-3"
    assert result.starting_xi["xpts"].sum() == 104.0
    assert result.bench["bench_order"].astype(int).tolist()[-1] == 4


def test_captain_uses_xpts_and_vice_uses_nonplay_coverage() -> None:
    frame = _prediction_frame()
    frame.loc[frame["fpl_element_id"] == 15, ["xpts", "p_play_any", "p_minutes_60"]] = [
        20.0,
        0.4,
        0.3,
    ]
    frame.loc[frame["fpl_element_id"] == 14, ["xpts", "p_play_any", "p_minutes_60"]] = [
        19.0,
        0.1,
        0.1,
    ]
    frame.loc[frame["fpl_element_id"] == 13, ["xpts", "p_play_any", "p_minutes_60"]] = [
        18.0,
        1.0,
        0.9,
    ]

    result = make_decision(_team(), frame)

    assert result.captain_player_key == "p15"
    assert result.vice_captain_player_key == "p13"
    assert "LOW_PLAY_ANY" in result.player_table.set_index("player_key").at["p15", "risk_flags"]


def test_bench_order_uses_availability_weighted_value() -> None:
    frame = _prediction_frame()
    # In the maximum-xPts XI the outfield bench is p03, p04, and p08. Although p08 has the
    # highest raw xPts, low availability pushes it behind both available defenders.
    frame.loc[frame["fpl_element_id"] == 8, "p_play_any"] = 0.1
    frame.loc[frame["fpl_element_id"] == 8, "p_minutes_60"] = 0.1

    bench = make_decision(_team(), frame).bench.sort_values("bench_order")

    assert bench["player_key"].tolist() == ["p04", "p03", "p08", "p01"]


def test_frozen_decision_report_retains_artifact_trace(tmp_path) -> None:
    result = make_decision(
        _team(),
        _prediction_frame(),
        snapshot_artifact_id="snapshot-sha",
        model_artifact_id="model-sha",
    )
    destination = tmp_path / "decision.json"

    write_decision_report(result, destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["trace"]["snapshot_artifact_id"] == "snapshot-sha"
    assert payload["trace"]["model_artifact_id"] == "model-sha"
    assert payload["trace"]["rules_version"] == "2026-27-v1"
    with pytest.raises(FileExistsError):
        write_decision_report(result, destination)


def test_missing_prediction_and_illegal_squad_fail_closed() -> None:
    with pytest.raises(DecisionInputError, match="missing prediction.*15"):
        make_decision(_team(), _prediction_frame().iloc[:-1])

    illegal = _prediction_frame()
    illegal.loc[illegal["fpl_element_id"].isin([1, 6, 11, 15]), "team_id"] = 99
    with pytest.raises(DecisionInputError, match="players per club"):
        make_decision(_team(), illegal)


def test_tie_breaking_is_deterministic_under_input_reordering() -> None:
    frame = _prediction_frame()
    frame["xpts"] = 5.0
    first = make_decision(_team(), frame)
    second = make_decision(_team(), frame.sample(frac=1.0, random_state=42))

    first_starters = sorted(first.starting_xi["player_key"])
    second_starters = sorted(second.starting_xi["player_key"])
    assert first_starters == second_starters
    assert first.captain_player_key == second.captain_player_key == "p01"
    assert first.vice_captain_player_key == second.vice_captain_player_key == "p03"


def test_decision_regret_is_hindsight_paired_against_last5() -> None:
    frame = _prediction_frame()
    frame["season"] = "2025-26"
    frame["gameweek"] = 4
    frame["squad_id"] = "s1"
    frame["actual_points_gw"] = frame["xpts"]
    frame["actual_minutes_gw"] = 90
    frame["pred_last5_points"] = -frame["xpts"]

    report = decision_regret_backtest(
        frame,
        champion_prediction="xpts",
        baselines={"last5": "pred_last5_points"},
        columns=PredictionColumns(),
    )

    assert len(report.by_gameweek) == 2
    champion = report.by_gameweek.set_index("policy").loc["champion"]
    baseline = report.by_gameweek.set_index("policy").loc["last5"]
    assert champion["xi_regret"] == 0.0
    assert champion["captain_regret"] == 0.0
    assert baseline["xi_regret"] > champion["xi_regret"]
    assert set(report.paired_deltas["baseline_policy"]) == {"last5"}
