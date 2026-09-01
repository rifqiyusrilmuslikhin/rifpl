"""Walk-forward evaluation, metrics, and leakage-check boundaries."""

from fpl_model.evaluation.leakage_check import (
    CANONICAL_KEY_COLUMNS,
    CURRENT_GAMEWEEK_RAW_COLUMNS,
    TARGET_COLUMNS,
    LeakageError,
    SuspiciousCorrelation,
    assert_chronological_fold,
    assert_dgw_deadline_anchoring,
    assert_feature_columns_safe,
    assert_feature_timestamps_predeadline,
    assert_fold_local_fit,
    assert_no_suspicious_raw_features,
    audit_feature_frame,
    find_suspicious_raw_features,
)

__all__ = [
    "CANONICAL_KEY_COLUMNS",
    "CURRENT_GAMEWEEK_RAW_COLUMNS",
    "TARGET_COLUMNS",
    "LeakageError",
    "SuspiciousCorrelation",
    "assert_chronological_fold",
    "assert_dgw_deadline_anchoring",
    "assert_feature_columns_safe",
    "assert_feature_timestamps_predeadline",
    "assert_fold_local_fit",
    "assert_no_suspicious_raw_features",
    "audit_feature_frame",
    "find_suspicious_raw_features",
]
