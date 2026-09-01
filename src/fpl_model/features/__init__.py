"""Point-in-time feature contracts and gameweek aggregation boundaries."""

from fpl_model.features.baseline import BaselineFeatureBuilder, FeatureBuildError
from fpl_model.features.contract import (
    BASELINE_FEATURE_CONTRACT,
    BASELINE_FEATURE_NAMES,
    FeatureContract,
    FeatureContractError,
    FeatureDefinition,
    load_feature_contract,
)
from fpl_model.features.reporting import (
    REQUIRED_SPOT_CHECK_CASES,
    build_coverage_report,
    write_coverage_report,
    write_spot_check_report,
)

__all__ = [
    "BASELINE_FEATURE_CONTRACT",
    "BASELINE_FEATURE_NAMES",
    "REQUIRED_SPOT_CHECK_CASES",
    "BaselineFeatureBuilder",
    "FeatureBuildError",
    "FeatureContract",
    "FeatureContractError",
    "FeatureDefinition",
    "build_coverage_report",
    "load_feature_contract",
    "write_coverage_report",
    "write_spot_check_report",
]
