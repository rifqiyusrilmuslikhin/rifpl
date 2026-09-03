"""Predeclared discovery and untouched-confirmation runners for champion stabilization.

The module deliberately separates model selection from confirmation. Discovery runs may compare
source arms, grouped ablations, and the checked-in small grid. A confirmation run accepts only a
checksummed :class:`SelectionFreeze`, and therefore cannot receive a list of alternatives.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.evaluation.artifacts import RetainedPredictions, retained_predictions
from fpl_model.evaluation.harness import FoldPredictor, WalkForwardHarness
from fpl_model.evaluation.windows import EvaluationConfigError, WindowRole
from fpl_model.features.contract import BASELINE_FEATURE_NAMES
from fpl_model.models.participation import (
    ENSEMBLE_COLUMNS,
    ENSEMBLE_SOURCE_COLUMNS,
    PARTICIPATION_TARGETS,
    PREDICTION_COLUMNS,
    ParticipationAwarePredictor,
    ParticipationConfig,
)

DEFAULT_STABILIZATION_CONFIG = Path("config/champion_stabilization.toml")
CHAMPION_SEEDS = (42, 7, 2026, 31415, 27182)
FINALIST_PREDICTIONS = frozenset({"xpts_direct", "xpts_conditional", "xpts_blend"})
ENSEMBLE_BY_FINALIST = {
    "xpts_direct": "xpts_direct_ensemble",
    "xpts_conditional": "xpts_conditional_ensemble",
    "xpts_blend": "xpts_blend_ensemble",
}
UNDERSTAT_ELIGIBILITY_COLUMN = "understat_complete"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ChampionStabilizationError(ValueError):
    """Raised when the stabilization protocol would become ambiguous or non-causal."""


@dataclass(frozen=True, slots=True)
class FeatureGroup:
    id: str
    hypothesis: str
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.hypothesis.strip() or not self.features:
            raise ChampionStabilizationError("feature groups require id, hypothesis, and features")
        if len(self.features) != len(set(self.features)):
            raise ChampionStabilizationError(f"feature group {self.id!r} contains duplicates")


@dataclass(frozen=True, slots=True)
class HyperparameterArm:
    id: str
    config: ParticipationConfig

    def __post_init__(self) -> None:
        if not self.id:
            raise ChampionStabilizationError("hyperparameter arm id must not be empty")


@dataclass(frozen=True, slots=True)
class ChampionStabilizationPlan:
    schema_version: str
    seeds: tuple[int, ...]
    discovery_windows: tuple[str, ...]
    confirmation_window: str
    primary_metric: str
    bootstrap_samples: int
    bootstrap_seed: int
    minimum_spearman_delta: float
    maximum_mae_regression: float
    maximum_absolute_bias_regression: float
    require_nonnegative_paired_ci: bool
    understat_features: tuple[str, ...]
    feature_groups: tuple[FeatureGroup, ...]
    hyperparameter_grid: tuple[HyperparameterArm, ...]

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ChampionStabilizationError("stabilization schema_version must not be empty")
        if self.seeds != CHAMPION_SEEDS:
            raise ChampionStabilizationError(
                f"champion finalist must use the five predeclared seeds {CHAMPION_SEEDS!r}"
            )
        if not self.discovery_windows or len(self.discovery_windows) != len(
            set(self.discovery_windows)
        ):
            raise ChampionStabilizationError("discovery windows must be non-empty and unique")
        if not self.confirmation_window or self.confirmation_window in self.discovery_windows:
            raise ChampionStabilizationError("confirmation must be distinct from discovery")
        if self.primary_metric != "ndcg_at_10":
            raise ChampionStabilizationError("Sprint 11 fixes NDCG@10 as the primary metric")
        if self.bootstrap_samples < 1:
            raise ChampionStabilizationError("bootstrap_samples must be positive")
        if any(
            value < 0
            for value in (self.maximum_mae_regression, self.maximum_absolute_bias_regression)
        ):
            raise ChampionStabilizationError("guardrail regression tolerances must be non-negative")

        feature_set = set(BASELINE_FEATURE_NAMES)
        understat = set(self.understat_features)
        if (
            not understat
            or len(understat) != len(self.understat_features)
            or not understat.issubset(feature_set)
        ):
            raise ChampionStabilizationError(
                "understat_features must be a non-empty unique subset of the baseline contract"
            )
        group_ids = [group.id for group in self.feature_groups]
        grouped = [feature for group in self.feature_groups for feature in group.features]
        if len(group_ids) != len(set(group_ids)):
            raise ChampionStabilizationError("feature group ids must be unique")
        if set(grouped) != feature_set:
            raise ChampionStabilizationError(
                "hypothesis groups must collectively cover all 46 baseline features"
            )
        understat_groups = [
            group for group in self.feature_groups if set(group.features) == understat
        ]
        if len(understat_groups) != 1:
            raise ChampionStabilizationError(
                "one feature group must exactly match the FPL-versus-Understat source arm"
            )

        grid_ids = [arm.id for arm in self.hyperparameter_grid]
        if not 2 <= len(grid_ids) <= 6 or len(grid_ids) != len(set(grid_ids)):
            raise ChampionStabilizationError("small grid must contain 2-6 uniquely named arms")
        if grid_ids[0] != "baseline" or self.hyperparameter_grid[0].config != ParticipationConfig():
            raise ChampionStabilizationError(
                "the first grid arm must reproduce the frozen Sprint 9 baseline"
            )

    @property
    def plan_hash(self) -> str:
        return _json_hash(self.to_dict())

    @property
    def fpl_only_features(self) -> tuple[str, ...]:
        understat = set(self.understat_features)
        return tuple(feature for feature in BASELINE_FEATURE_NAMES if feature not in understat)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seeds": list(self.seeds),
            "discovery_windows": list(self.discovery_windows),
            "confirmation_window": self.confirmation_window,
            "primary_metric": self.primary_metric,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "guardrails": {
                "minimum_spearman_delta": self.minimum_spearman_delta,
                "maximum_mae_regression": self.maximum_mae_regression,
                "maximum_absolute_bias_regression": self.maximum_absolute_bias_regression,
                "require_nonnegative_paired_ci": self.require_nonnegative_paired_ci,
            },
            "understat_features": list(self.understat_features),
            "feature_groups": [asdict(group) for group in self.feature_groups],
            "hyperparameter_grid": [
                {"id": arm.id, **asdict(arm.config)} for arm in self.hyperparameter_grid
            ],
        }


def load_champion_stabilization_plan(
    path: str | Path = DEFAULT_STABILIZATION_CONFIG,
) -> ChampionStabilizationPlan:
    """Load and strictly validate the predeclared Sprint 11 experiment surface."""
    with Path(path).open("rb") as config_file:
        payload = tomllib.load(config_file)
    guardrails = payload["guardrails"]
    groups = tuple(
        FeatureGroup(item["id"], item["hypothesis"], tuple(item["features"]))
        for item in payload["feature_groups"]
    )
    grid = tuple(
        HyperparameterArm(
            item["id"],
            ParticipationConfig(**{key: value for key, value in item.items() if key != "id"}),
        )
        for item in payload["hyperparameter_grid"]
    )
    return ChampionStabilizationPlan(
        schema_version=payload["schema_version"],
        seeds=tuple(payload["seeds"]),
        discovery_windows=tuple(payload["discovery_windows"]),
        confirmation_window=payload["confirmation_window"],
        primary_metric=payload["primary_metric"],
        bootstrap_samples=payload["bootstrap_samples"],
        bootstrap_seed=payload["bootstrap_seed"],
        minimum_spearman_delta=float(guardrails["minimum_spearman_delta"]),
        maximum_mae_regression=float(guardrails["maximum_mae_regression"]),
        maximum_absolute_bias_regression=float(guardrails["maximum_absolute_bias_regression"]),
        require_nonnegative_paired_ci=guardrails["require_nonnegative_paired_ci"],
        understat_features=tuple(payload["understat_features"]),
        feature_groups=groups,
        hyperparameter_grid=grid,
    )


@dataclass(frozen=True, slots=True)
class FinalistDeclaration:
    """Trace from a prior discovery decision to the one architecture stabilized here."""

    prediction: str
    evidence_sha256: str
    discovery_windows: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = self.prediction.removesuffix("_ensemble")
        if normalized not in FINALIST_PREDICTIONS:
            raise ChampionStabilizationError(
                f"finalist prediction must be one of {sorted(FINALIST_PREDICTIONS)!r}"
            )
        if not _SHA256_PATTERN.fullmatch(self.evidence_sha256):
            raise ChampionStabilizationError("finalist evidence must be a lowercase SHA-256")
        if not self.discovery_windows:
            raise ChampionStabilizationError("finalist declaration requires discovery windows")
        object.__setattr__(self, "prediction", normalized)

    @classmethod
    def from_report(
        cls,
        path: str | Path,
        *,
        discovery_windows: Sequence[str],
        prediction: str | None = None,
    ) -> FinalistDeclaration:
        """Verify a retained discovery report and derive its selected finalist."""
        source = Path(path)
        raw = source.read_bytes()
        payload = json.loads(raw)
        gate = payload.get("promotion_gate", {})
        selected = gate.get("selected_on_discovery")
        if selected is None and gate.get("discovery_result") is True:
            selected = "xpts_direct"
        if selected is None and str(payload.get("decision", "")).startswith("KEEP_DIRECT"):
            selected = "xpts_direct"
        if selected is None:
            raise ChampionStabilizationError(
                "source report does not retain a finalist that passed discovery"
            )
        normalized = str(selected).removesuffix("_ensemble")
        if prediction is not None and prediction.removesuffix("_ensemble") != normalized:
            raise ChampionStabilizationError(
                "declared finalist disagrees with the retained discovery report"
            )
        return cls(normalized, hashlib.sha256(raw).hexdigest(), tuple(discovery_windows))


@dataclass(frozen=True, slots=True)
class StabilizationArm:
    id: str
    category: str
    hypothesis: str
    feature_columns: tuple[str, ...]
    hyperparameter_id: str
    config: ParticipationConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "hypothesis": self.hypothesis,
            "feature_columns": list(self.feature_columns),
            "feature_count": len(self.feature_columns),
            "hyperparameter_id": self.hyperparameter_id,
            "hyperparameters": asdict(self.config),
        }


def build_stabilization_arms(plan: ChampionStabilizationPlan) -> tuple[StabilizationArm, ...]:
    """Construct the complete experiment list without consulting any model result."""
    baseline_config = plan.hyperparameter_grid[0].config
    arms = [
        StabilizationArm(
            "full/baseline",
            "finalist",
            "Stabilize the prior finalist using all 46 baseline features.",
            BASELINE_FEATURE_NAMES,
            "baseline",
            baseline_config,
        ),
        StabilizationArm(
            "fpl_only/baseline",
            "source_comparison",
            "Test whether nullable Understat enrichment adds operational value over FPL-only data.",
            plan.fpl_only_features,
            "baseline",
            baseline_config,
        ),
    ]
    understat = set(plan.understat_features)
    for group in plan.feature_groups:
        if set(group.features) == understat:
            # The FPL-only arm is exactly this ablation and is trained only once.
            continue
        removed = set(group.features)
        features = tuple(feature for feature in BASELINE_FEATURE_NAMES if feature not in removed)
        arms.append(
            StabilizationArm(
                f"ablation/drop_{group.id}",
                "grouped_ablation",
                group.hypothesis,
                features,
                "baseline",
                baseline_config,
            )
        )
    for grid_arm in plan.hyperparameter_grid[1:]:
        arms.append(
            StabilizationArm(
                f"grid/{grid_arm.id}",
                "hyperparameter_grid",
                f"Predeclared small-grid configuration {grid_arm.id!r}.",
                BASELINE_FEATURE_NAMES,
                grid_arm.id,
                grid_arm.config,
            )
        )
    identities = [(arm.feature_columns, arm.config) for arm in arms]
    if len(identities) != len(set(identities)):
        raise ChampionStabilizationError("predeclared study contains a duplicate experiment arm")
    return tuple(arms)


PredictorFactory = Callable[[StabilizationArm, int, object], FoldPredictor]


@dataclass(frozen=True, slots=True)
class DiscoveryStudy:
    plan: ChampionStabilizationPlan
    finalist: FinalistDeclaration
    arms: tuple[StabilizationArm, ...]
    artifacts: Mapping[str, RetainedPredictions]
    data_manifest_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = {arm.id for arm in self.arms}
        if set(self.artifacts) != expected:
            raise ChampionStabilizationError("discovery artifacts do not match predeclared arms")
        if any(
            not _SHA256_PATTERN.fullmatch(str(value))
            for value in self.data_manifest_hashes.values()
        ):
            raise ChampionStabilizationError("data manifest hashes must be lowercase SHA-256")

    def write(self, directory: str | Path) -> Path:
        """Retain every arm plus a checksummed study index in a new immutable directory."""
        destination = Path(directory)
        if destination.exists():
            raise FileExistsError(f"frozen discovery directory already exists: {destination}")
        destination.mkdir(parents=True)
        artifact_rows = []
        for arm in self.arms:
            filename = arm.id.replace("/", "__") + ".parquet"
            artifact_path, metadata_path = self.artifacts[arm.id].write(
                destination / filename,
                metadata={
                    "phase": "discovery",
                    "arm": arm.to_dict(),
                    "stabilization_plan_sha256": self.plan.plan_hash,
                    "finalist_evidence_sha256": self.finalist.evidence_sha256,
                    "data_manifest_hashes": dict(self.data_manifest_hashes),
                },
            )
            artifact_rows.append(
                {
                    "arm_id": arm.id,
                    "artifact": artifact_path.name,
                    "artifact_sha256": _file_hash(artifact_path),
                    "metadata": metadata_path.name,
                    "metadata_sha256": _file_hash(metadata_path),
                }
            )
        index = {
            "schema_version": "1.0.0",
            "phase": "discovery",
            "stabilization_plan_sha256": self.plan.plan_hash,
            "finalist": asdict(self.finalist),
            "data_manifest_hashes": dict(self.data_manifest_hashes),
            "arms": [arm.to_dict() for arm in self.arms],
            "artifacts": artifact_rows,
        }
        (destination / "study.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination.resolve()


def run_champion_discovery(
    frame: pd.DataFrame,
    harness: WalkForwardHarness,
    finalist: FinalistDeclaration,
    *,
    plan: ChampionStabilizationPlan | None = None,
    context_columns: Sequence[str] = (),
    eligibility_column: str = "eligibility",
    understat_eligibility_column: str | None = None,
    data_manifest_hashes: Mapping[str, str] | None = None,
    predictor_factory: PredictorFactory | None = None,
) -> DiscoveryStudy:
    """Run five-seed finalist, source arms, grouped ablations, and grid on discovery only."""
    policy = plan or load_champion_stabilization_plan()
    if tuple(finalist.discovery_windows) != policy.discovery_windows:
        raise ChampionStabilizationError(
            "finalist evidence windows must exactly match the predeclared discovery windows"
        )
    harness.splitter.plan.assert_selection_allowed(policy.discovery_windows)
    _validate_window_roles(harness, policy.discovery_windows, WindowRole.DISCOVERY)
    prepared = _with_understat_eligibility(
        frame,
        policy,
        source_column=understat_eligibility_column,
    )
    arms = build_stabilization_arms(policy)
    retained_context = tuple(
        dict.fromkeys(
            [
                *context_columns,
                UNDERSTAT_ELIGIBILITY_COLUMN,
                *(
                    column
                    for column in ("position", "fixture_count", "status_risk_ordinal")
                    if column in frame
                ),
            ]
        )
    )
    artifacts = {
        arm.id: _run_arm(
            prepared,
            harness,
            arm,
            policy,
            window_names=policy.discovery_windows,
            context_columns=retained_context,
            eligibility_column=eligibility_column,
            predictor_factory=predictor_factory,
        )
        for arm in arms
    }
    return DiscoveryStudy(
        policy,
        finalist,
        arms,
        artifacts,
        dict(data_manifest_hashes or {}),
    )


@dataclass(frozen=True, slots=True)
class SelectionFreeze:
    schema_version: str
    created_from_discovery_sha256: str
    finalist_evidence_sha256: str
    stabilization_plan_sha256: str
    evaluation_plan_version: str
    discovery_windows: tuple[str, ...]
    confirmation_window: str
    selected_arm: str
    selected_prediction: str
    feature_columns: tuple[str, ...]
    hyperparameter_id: str
    hyperparameters: Mapping[str, Any]
    seeds: tuple[int, ...]
    selection_sha256: str

    def __post_init__(self) -> None:
        if self.selected_prediction not in FINALIST_PREDICTIONS:
            raise ChampionStabilizationError("selection freeze has an unknown prediction head")
        if self.seeds != CHAMPION_SEEDS:
            raise ChampionStabilizationError("selection freeze does not retain all five seeds")
        if not self.feature_columns or not set(self.feature_columns).issubset(
            BASELINE_FEATURE_NAMES
        ):
            raise ChampionStabilizationError("selection freeze contains invalid features")
        if self.selection_sha256 != _json_hash(self._unsigned_dict()):
            raise ChampionStabilizationError("selection freeze checksum mismatch")

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_from_discovery_sha256": self.created_from_discovery_sha256,
            "finalist_evidence_sha256": self.finalist_evidence_sha256,
            "stabilization_plan_sha256": self.stabilization_plan_sha256,
            "evaluation_plan_version": self.evaluation_plan_version,
            "discovery_windows": list(self.discovery_windows),
            "confirmation_window": self.confirmation_window,
            "selected_arm": self.selected_arm,
            "selected_prediction": self.selected_prediction,
            "feature_columns": list(self.feature_columns),
            "hyperparameter_id": self.hyperparameter_id,
            "hyperparameters": dict(self.hyperparameters),
            "seeds": list(self.seeds),
        }

    def to_dict(self) -> dict[str, Any]:
        return self._unsigned_dict() | {"selection_sha256": self.selection_sha256}

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"selection freeze already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination.resolve()

    @classmethod
    def read_json(cls, path: str | Path) -> SelectionFreeze:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            schema_version=payload["schema_version"],
            created_from_discovery_sha256=payload["created_from_discovery_sha256"],
            finalist_evidence_sha256=payload["finalist_evidence_sha256"],
            stabilization_plan_sha256=payload["stabilization_plan_sha256"],
            evaluation_plan_version=payload["evaluation_plan_version"],
            discovery_windows=tuple(payload["discovery_windows"]),
            confirmation_window=payload["confirmation_window"],
            selected_arm=payload["selected_arm"],
            selected_prediction=payload["selected_prediction"],
            feature_columns=tuple(payload["feature_columns"]),
            hyperparameter_id=payload["hyperparameter_id"],
            hyperparameters=payload["hyperparameters"],
            seeds=tuple(payload["seeds"]),
            selection_sha256=payload["selection_sha256"],
        )


def make_selection_freeze(
    *,
    discovery_report: Mapping[str, Any],
    study: DiscoveryStudy,
    evaluation_plan_version: str,
) -> SelectionFreeze:
    """Freeze the discovery-only winner; report content becomes part of the seal."""
    if discovery_report.get("phase") != "discovery":
        raise ChampionStabilizationError("selection may only be frozen from a discovery report")
    selected_id = discovery_report.get("selection", {}).get("selected_arm")
    by_id = {arm.id: arm for arm in study.arms}
    if selected_id not in by_id:
        raise ChampionStabilizationError("discovery report selected an unknown arm")
    report_plan_hash = discovery_report.get("stabilization_plan_sha256")
    if report_plan_hash != study.plan.plan_hash:
        raise ChampionStabilizationError("discovery report and study plan hashes disagree")
    report_hash = _json_hash(discovery_report)
    arm = by_id[str(selected_id)]
    unsigned = {
        "schema_version": "1.0.0",
        "created_from_discovery_sha256": report_hash,
        "finalist_evidence_sha256": study.finalist.evidence_sha256,
        "stabilization_plan_sha256": study.plan.plan_hash,
        "evaluation_plan_version": evaluation_plan_version,
        "discovery_windows": list(study.plan.discovery_windows),
        "confirmation_window": study.plan.confirmation_window,
        "selected_arm": arm.id,
        "selected_prediction": study.finalist.prediction,
        "feature_columns": list(arm.feature_columns),
        "hyperparameter_id": arm.hyperparameter_id,
        "hyperparameters": asdict(arm.config),
        "seeds": list(study.plan.seeds),
    }
    return SelectionFreeze(
        schema_version="1.0.0",
        created_from_discovery_sha256=report_hash,
        finalist_evidence_sha256=study.finalist.evidence_sha256,
        stabilization_plan_sha256=study.plan.plan_hash,
        evaluation_plan_version=evaluation_plan_version,
        discovery_windows=study.plan.discovery_windows,
        confirmation_window=study.plan.confirmation_window,
        selected_arm=arm.id,
        selected_prediction=study.finalist.prediction,
        feature_columns=arm.feature_columns,
        hyperparameter_id=arm.hyperparameter_id,
        hyperparameters=asdict(arm.config),
        seeds=study.plan.seeds,
        selection_sha256=_json_hash(unsigned),
    )


def run_untouched_confirmation(
    frame: pd.DataFrame,
    harness: WalkForwardHarness,
    selection: SelectionFreeze,
    *,
    plan: ChampionStabilizationPlan | None = None,
    artifact_path: str | Path | None = None,
    context_columns: Sequence[str] = (),
    eligibility_column: str = "eligibility",
    understat_eligibility_column: str | None = None,
    predictor_factory: PredictorFactory | None = None,
) -> RetainedPredictions:
    """Evaluate exactly one frozen configuration once on the untouched window."""
    policy = plan or load_champion_stabilization_plan()
    if selection.stabilization_plan_sha256 != policy.plan_hash:
        raise ChampionStabilizationError("selection was frozen under a different experiment plan")
    if selection.evaluation_plan_version != harness.splitter.plan.version:
        raise ChampionStabilizationError("selection and evaluation plan versions disagree")
    if selection.confirmation_window != policy.confirmation_window:
        raise ChampionStabilizationError("selection names a different confirmation window")
    _validate_window_roles(harness, (selection.confirmation_window,), WindowRole.CONFIRMATION)
    if artifact_path is not None:
        path = Path(artifact_path)
        if path.suffix.casefold() != ".parquet":
            raise ChampionStabilizationError("confirmation artifact path must end in .parquet")
        if path.exists() or path.with_suffix(".metadata.json").exists():
            raise FileExistsError(f"untouched confirmation artifact already exists: {path}")

    prepared = _with_understat_eligibility(
        frame,
        policy,
        source_column=understat_eligibility_column,
    )
    arm = StabilizationArm(
        selection.selected_arm,
        "frozen_confirmation",
        "Exactly the configuration sealed after discovery.",
        selection.feature_columns,
        selection.hyperparameter_id,
        ParticipationConfig(**dict(selection.hyperparameters)),
    )
    retained_context = tuple(
        dict.fromkeys(
            [
                *context_columns,
                UNDERSTAT_ELIGIBILITY_COLUMN,
                *(
                    column
                    for column in ("position", "fixture_count", "status_risk_ordinal")
                    if column in frame
                ),
            ]
        )
    )
    artifact = _run_arm(
        prepared,
        harness,
        arm,
        policy,
        window_names=(selection.confirmation_window,),
        context_columns=retained_context,
        eligibility_column=eligibility_column,
        predictor_factory=predictor_factory,
    )
    if artifact_path is not None:
        artifact.write(
            artifact_path,
            metadata={
                "phase": "confirmation",
                "selection_sha256": selection.selection_sha256,
                "selected_arm": selection.selected_arm,
                "selected_prediction": selection.selected_prediction,
            },
        )
    return artifact


def _run_arm(
    frame: pd.DataFrame,
    harness: WalkForwardHarness,
    arm: StabilizationArm,
    plan: ChampionStabilizationPlan,
    *,
    window_names: Sequence[str],
    context_columns: tuple[str, ...],
    eligibility_column: str,
    predictor_factory: PredictorFactory | None,
) -> RetainedPredictions:
    def factory(seed: int, fold: object) -> FoldPredictor:
        if predictor_factory is not None:
            return predictor_factory(arm, seed, fold)
        return ParticipationAwarePredictor(
            seed=seed,
            config=arm.config,
            expected_feature_columns=arm.feature_columns,
        )

    individual = harness.run(
        frame,
        factory,
        feature_columns=arm.feature_columns,
        target_columns=PARTICIPATION_TARGETS,
        prediction_columns=PREDICTION_COLUMNS,
        seeds=plan.seeds,
        baseline_columns=("points_mean_last5",),
        eligibility_column=eligibility_column,
        context_columns=context_columns,
        window_names=window_names,
    )
    return _with_five_seed_ensembles(individual, seeds=plan.seeds)


def _with_five_seed_ensembles(
    artifact: RetainedPredictions,
    *,
    seeds: tuple[int, ...],
) -> RetainedPredictions:
    frame = artifact.frame
    identity = [*artifact.key_columns, "fold"]
    expected = set(seeds)
    observed = frame.groupby(identity, sort=False, dropna=False)["seed"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    if not observed.map(lambda values: values == expected).all():
        raise ChampionStabilizationError("every finalist row must retain all five seeds")
    for source, destination in zip(ENSEMBLE_SOURCE_COLUMNS, ENSEMBLE_COLUMNS, strict=True):
        mean = frame.groupby(identity, sort=False, dropna=False)[source].transform("mean")
        if not np.isfinite(mean.to_numpy(dtype=float)).all():
            raise ChampionStabilizationError(f"ensemble {destination!r} is non-finite")
        frame[destination] = mean.to_numpy(dtype=float)
    if (frame["p_minutes_60_ensemble"] > frame["p_play_any_ensemble"] + 1e-12).any():
        raise ChampionStabilizationError("probability coherence was lost during ensembling")
    lower = 60.0 * frame["p_minutes_60_ensemble"]
    upper = frame["gw_max_minutes"] * frame["p_play_any_ensemble"]
    if (frame["expected_minutes_ensemble"] < lower - 1e-9).any() or (
        frame["expected_minutes_ensemble"] > upper + 1e-9
    ).any():
        raise ChampionStabilizationError("minutes coherence was lost during ensembling")
    return retained_predictions(
        frame,
        prediction_columns=(*artifact.prediction_columns, *ENSEMBLE_COLUMNS),
        target_columns=artifact.target_columns,
        baseline_columns=artifact.baseline_columns,
        key_columns=artifact.key_columns,
    )


def collapse_five_seed_ensemble(
    artifact: RetainedPredictions,
    *,
    seeds: Sequence[int] = CHAMPION_SEEDS,
) -> pd.DataFrame:
    """Return one audited row per player-GW/fold from a five-seed artifact."""
    frame = artifact.frame
    identity = [*artifact.key_columns, "fold"]
    expected = set(seeds)
    observed = frame.groupby(identity, dropna=False)["seed"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    if not observed.map(lambda values: values == expected).all():
        raise ChampionStabilizationError("ensemble collapse requires every declared seed")
    invariant = [
        *ENSEMBLE_COLUMNS,
        *artifact.target_columns,
        *artifact.baseline_columns,
        "eligibility",
    ]
    for column in dict.fromkeys(invariant):
        variation = frame.groupby(identity, dropna=False)[column].nunique(dropna=False)
        if variation.gt(1).any():
            raise ChampionStabilizationError(
                f"ensemble invariant {column!r} differs across seed rows"
            )
    return (
        frame.sort_values([*identity, "seed"])
        .drop_duplicates(identity, keep="first")
        .reset_index(drop=True)
    )


def _with_understat_eligibility(
    frame: pd.DataFrame,
    plan: ChampionStabilizationPlan,
    *,
    source_column: str | None,
) -> pd.DataFrame:
    missing = sorted(set(plan.understat_features).difference(frame.columns))
    if missing:
        raise ChampionStabilizationError(f"model frame is missing Understat columns {missing!r}")
    prepared = frame.copy()
    complete = prepared.loc[:, list(plan.understat_features)].notna().all(axis=1)
    if source_column is not None:
        if source_column not in prepared:
            raise ChampionStabilizationError(
                f"Understat eligibility column {source_column!r} is missing"
            )
        supplied = prepared[source_column]
        if supplied.isna().any() or not supplied.isin((True, False)).all():
            raise ChampionStabilizationError("Understat eligibility must be a complete boolean")
        supplied = supplied.astype(bool)
        if (supplied & ~complete).any():
            raise ChampionStabilizationError(
                "a row cannot be Understat-eligible while required signals are missing"
            )
        complete = supplied
    prepared[UNDERSTAT_ELIGIBILITY_COLUMN] = complete.astype(bool)
    return prepared


def _validate_window_roles(
    harness: WalkForwardHarness,
    names: Sequence[str],
    expected: WindowRole,
) -> None:
    for name in names:
        try:
            window = harness.splitter.plan.window(name)
        except EvaluationConfigError as error:
            raise ChampionStabilizationError(str(error)) from error
        if window.role is not expected:
            raise ChampionStabilizationError(
                f"window {name!r} must have role={expected.value!r}, got {window.role.value!r}"
            )


def _json_hash(value: Mapping[str, Any] | dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=list).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "CHAMPION_SEEDS",
    "DEFAULT_STABILIZATION_CONFIG",
    "ENSEMBLE_BY_FINALIST",
    "FINALIST_PREDICTIONS",
    "UNDERSTAT_ELIGIBILITY_COLUMN",
    "ChampionStabilizationError",
    "ChampionStabilizationPlan",
    "DiscoveryStudy",
    "FeatureGroup",
    "FinalistDeclaration",
    "HyperparameterArm",
    "SelectionFreeze",
    "StabilizationArm",
    "build_stabilization_arms",
    "collapse_five_seed_ensemble",
    "load_champion_stabilization_plan",
    "make_selection_freeze",
    "run_champion_discovery",
    "run_untouched_confirmation",
]
