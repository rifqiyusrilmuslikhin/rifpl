"""Machine-readable contract for the fixed Sprint 4 baseline feature set."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


class FeatureContractError(ValueError):
    """Raised when the checked-in feature contract is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    dtype: str
    source: str
    availability_cutoff: str
    window: str
    aggregation: str
    missing_rule: str
    hypothesis_reference: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FeatureDefinition:
        expected = tuple(cls.__dataclass_fields__)
        if set(value) != set(expected):
            raise FeatureContractError(
                f"feature contract fields must be exactly {expected!r}; got {tuple(value)!r}"
            )
        fields: dict[str, str] = {}
        for name in expected:
            item = value[name]
            if not isinstance(item, str) or not item.strip():
                raise FeatureContractError(f"feature field {name!r} must be a non-empty string")
            fields[name] = item.strip()
        return cls(**fields)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class FeatureContract:
    contract_version: str
    per90_epsilon_minutes: float
    per90_minimum_minutes: int
    features: tuple[FeatureDefinition, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "feature_count": len(self.features),
            "per90_epsilon_minutes": self.per90_epsilon_minutes,
            "per90_minimum_minutes": self.per90_minimum_minutes,
            "features": [feature.to_dict() for feature in self.features],
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination.resolve()


def load_feature_contract(path: str | Path | None = None) -> FeatureContract:
    """Load and strictly validate the checked-in contract (or an explicit contract file)."""
    if path is None:
        resource = files("fpl_model.features").joinpath("baseline_contract.json")
        payload = json.loads(resource.read_text(encoding="utf-8"))
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FeatureContractError("feature contract root must be an object")
    required = {
        "contract_version",
        "feature_count",
        "per90_epsilon_minutes",
        "per90_minimum_minutes",
        "features",
    }
    if set(payload) != required:
        raise FeatureContractError(
            f"feature contract root fields must be exactly {sorted(required)}"
        )
    features_value = payload["features"]
    if not isinstance(features_value, list):
        raise FeatureContractError("features must be a list")
    features = tuple(FeatureDefinition.from_mapping(value) for value in features_value)
    names = tuple(feature.name for feature in features)
    if len(features) != 46 or payload["feature_count"] != 46:
        raise FeatureContractError("baseline contract must contain exactly 46 features")
    if len(set(names)) != len(names):
        raise FeatureContractError("baseline feature names must be unique")
    version = payload["contract_version"]
    epsilon = payload["per90_epsilon_minutes"]
    minimum = payload["per90_minimum_minutes"]
    if not isinstance(version, str) or not version:
        raise FeatureContractError("contract_version must be a non-empty string")
    if not isinstance(epsilon, int | float) or isinstance(epsilon, bool) or epsilon <= 0:
        raise FeatureContractError("per90_epsilon_minutes must be positive")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0:
        raise FeatureContractError("per90_minimum_minutes must be a positive integer")
    return FeatureContract(version, float(epsilon), minimum, features)


BASELINE_FEATURE_CONTRACT = load_feature_contract()
BASELINE_FEATURE_NAMES = BASELINE_FEATURE_CONTRACT.feature_names
