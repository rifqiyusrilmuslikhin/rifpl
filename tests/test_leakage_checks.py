"""Adversarial acceptance tests for the Sprint 5 leakage release gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from fpl_model.evaluation import (
    LeakageError,
    assert_chronological_fold,
    assert_dgw_deadline_anchoring,
    assert_feature_columns_safe,
    assert_feature_timestamps_predeadline,
    assert_fold_local_fit,
    assert_no_suspicious_raw_features,
    audit_feature_frame,
    find_suspicious_raw_features,
)

DEADLINE = datetime(2025, 8, 16, 10, tzinfo=UTC)


def _audit_frame(rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": ["2025-26"] * rows,
            "gameweek": [index // 3 + 1 for index in range(rows)],
            "player_key": [f"player-{index}" for index in range(rows)],
            "deadline_utc": [DEADLINE + timedelta(days=index // 3 * 7) for index in range(rows)],
            "snapshot_captured_at_utc": [
                DEADLINE + timedelta(days=index // 3 * 7, hours=-2) for index in range(rows)
            ],
            "feature_cutoff_utc": [
                DEADLINE + timedelta(days=index // 3 * 7, hours=-1) for index in range(rows)
            ],
            "minutes_mean_last5": [10, 70, 20, 65, 30, 60, 25, 55, 35, 50, 40, 45],
            "fixture_count": [1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2],
            "actual_minutes_gw": [0, 90, 15, 80, 20, 70, 30, 60, 40, 50, 45, 35],
            "actual_points_gw": [1, 3, 8, 2, 11, 0, 4, 7, 2, 6, 1, 9],
        }
    )


def _fold_frame(season: str, deadline: datetime, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [season, season],
            "gameweek": [1, 1],
            "player_key": [f"{prefix}-a", f"{prefix}-b"],
            "deadline_utc": [deadline, deadline],
        }
    )


def test_honest_feature_frame_passes_timestamp_denylist_and_proxy_audit() -> None:
    frame = _audit_frame()

    audit_feature_frame(frame, ("minutes_mean_last5", "fixture_count"))


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(microseconds=1)])
def test_feature_timestamp_at_or_after_deadline_is_rejected(offset: timedelta) -> None:
    frame = _audit_frame()
    frame.loc[4, "news_available_at_utc"] = frame.loc[4, "deadline_utc"] + offset

    with pytest.raises(LeakageError, match="strictly before"):
        assert_feature_timestamps_predeadline(frame)


def test_explicit_post_deadline_provenance_column_is_rejected() -> None:
    frame = _audit_frame()
    frame["custom_source_time"] = frame["deadline_utc"] - timedelta(minutes=1)
    frame.loc[0, "custom_source_time"] = frame.loc[0, "deadline_utc"]

    with pytest.raises(LeakageError, match="custom_source_time"):
        assert_feature_timestamps_predeadline(frame, timestamp_columns=("custom_source_time",))


def test_missing_required_feature_provenance_is_rejected() -> None:
    frame = _audit_frame()
    frame.loc[0, "snapshot_captured_at_utc"] = pd.NaT

    with pytest.raises(LeakageError, match="must not be missing"):
        assert_feature_timestamps_predeadline(frame)


@pytest.mark.parametrize(
    "injected_feature",
    ["actual_minutes_gw", "target_gw_minutes", "event_live_total_points", "minutes"],
)
def test_actual_target_and_current_gw_outcomes_are_denylisted(
    injected_feature: str,
) -> None:
    with pytest.raises(LeakageError, match="target/current-GW"):
        assert_feature_columns_safe(("minutes_mean_last5", injected_feature))


def test_suspicious_disguised_target_proxy_is_detected() -> None:
    frame = _audit_frame()
    frame["mystery_availability_signal"] = frame["actual_minutes_gw"] * 2 + 7

    findings = find_suspicious_raw_features(
        frame,
        ("minutes_mean_last5", "mystery_availability_signal"),
    )
    assert [(item.feature, item.target) for item in findings] == [
        ("mystery_availability_signal", "actual_minutes_gw")
    ]
    with pytest.raises(LeakageError, match="suspicious direct target correlation"):
        assert_no_suspicious_raw_features(
            frame,
            ("minutes_mean_last5", "mystery_availability_signal"),
        )


def test_honest_chronological_fold_passes() -> None:
    train = _fold_frame("2024-25", DEADLINE - timedelta(days=100), "train")
    test = _fold_frame("2025-26", DEADLINE, "test")

    assert_chronological_fold(train, test)


def test_later_season_row_in_earlier_fold_is_rejected() -> None:
    test = _fold_frame("2024-25", DEADLINE, "test")
    train = _fold_frame("2023-24", DEADLINE - timedelta(days=100), "train")
    injected = _fold_frame("2025-26", DEADLINE + timedelta(days=300), "leaked")
    train = pd.concat([train, injected.iloc[[0]]], ignore_index=True)

    with pytest.raises(LeakageError, match="season later"):
        assert_chronological_fold(train, test)


def test_dgw_fixtures_share_one_deadline_locked_history_state() -> None:
    contexts = [
        {
            "season": "2025-26",
            "gameweek": 4,
            "player_key": "player-a",
            "fixture_id": fixture_id,
            "deadline_utc": DEADLINE,
            "snapshot_captured_at_utc": DEADLINE - timedelta(hours=2),
            "feature_cutoff_utc": DEADLINE - timedelta(hours=1),
            "context_anchor_id": "2025-26:gw4:player-a",
            "team_id_at_deadline": 1,
            "history_points_mean_last5": 4.2,
        }
        for fixture_id in (9001, 9002)
    ]

    assert_dgw_deadline_anchoring(contexts, history_feature_columns=("history_points_mean_last5",))


def test_second_dgw_fixture_seeing_first_fixture_is_rejected() -> None:
    contexts = [
        {
            "season": "2025-26",
            "gameweek": 4,
            "player_key": "player-a",
            "fixture_id": fixture_id,
            "deadline_utc": DEADLINE,
            "snapshot_captured_at_utc": DEADLINE - timedelta(hours=2),
            "feature_cutoff_utc": DEADLINE - timedelta(hours=1),
            "context_anchor_id": "2025-26:gw4:player-a",
            "team_id_at_deadline": 1,
            "history_points_mean_last5": history_points,
        }
        for fixture_id, history_points in ((9001, 4.2), (9002, 7.8))
    ]

    with pytest.raises(LeakageError, match="history_points_mean_last5"):
        assert_dgw_deadline_anchoring(
            contexts, history_feature_columns=("history_points_mean_last5",)
        )


def test_fold_local_preprocessing_passes_with_training_rows_only() -> None:
    train = pd.DataFrame({"form": [1.0, 2.0, 3.0]}, index=["train-1", "train-2", "train-3"])
    test = pd.DataFrame({"form": [100.0]}, index=["test-1"])
    scaler = StandardScaler().fit(train)

    assert_fold_local_fit(train.index, train.index, held_out_row_keys=test.index)
    assert scaler.mean_[0] == pytest.approx(2.0)


def test_preprocessing_fitted_using_test_period_is_rejected() -> None:
    train = pd.DataFrame({"form": [1.0, 2.0, 3.0]}, index=["train-1", "train-2", "train-3"])
    test = pd.DataFrame({"form": [100.0]}, index=["test-1"])
    leaked_fit_frame = pd.concat([train, test])

    # Deliberately perform the prohibited fit, then submit its retained fit-row provenance.
    StandardScaler().fit(leaked_fit_frame)
    with pytest.raises(LeakageError, match="held-out rows"):
        assert_fold_local_fit(
            train.index,
            leaked_fit_frame.index,
            held_out_row_keys=test.index,
        )


def test_duplicate_canonical_feature_keys_are_rejected() -> None:
    frame = _audit_frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(LeakageError, match="duplicate canonical keys"):
        audit_feature_frame(duplicate, ("minutes_mean_last5", "fixture_count"))
