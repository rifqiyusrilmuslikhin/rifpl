"""Discovery selection, confirmation gates, and immutable champion evidence bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import tomllib
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.evaluation.artifacts import RetainedPredictions
from fpl_model.evaluation.comparison import (
    assert_exact_same_rows,
    compare_ranking_predictions,
)
from fpl_model.evaluation.metrics import (
    probability_metrics_by_gameweek,
    ranking_metrics_by_gameweek,
)
from fpl_model.evaluation.windows import load_evaluation_plan
from fpl_model.features.contract import (
    BASELINE_FEATURE_CONTRACT,
    BASELINE_FEATURE_NAMES,
    load_feature_contract,
)
from fpl_model.models.champion_stabilization import (
    CHAMPION_SEEDS,
    ENSEMBLE_BY_FINALIST,
    UNDERSTAT_ELIGIBILITY_COLUMN,
    DiscoveryStudy,
    SelectionFreeze,
    StabilizationArm,
    collapse_five_seed_ensemble,
    load_champion_stabilization_plan,
)

_METRICS = ("ndcg_at_10", "spearman", "mae", "mean_bias")
_HIGHER_IS_BETTER = frozenset({"ndcg_at_10", "spearman"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_HASH_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class ChampionReportError(ValueError):
    """Raised when retained evidence cannot support a champion decision."""


@dataclass(frozen=True, slots=True)
class FrozenDiscoveryReport:
    schema_version: str
    phase: str
    stabilization_plan_sha256: str
    finalist: dict[str, Any]
    experiment: dict[str, Any]
    arms: list[dict[str, Any]]
    individual_seed_metrics: dict[str, dict[str, dict[str, float]]]
    ensemble_metrics: dict[str, dict[str, dict[str, float]]]
    seed_noise: list[dict[str, Any]]
    source_comparison: dict[str, Any]
    grouped_ablations: dict[str, Any]
    hyperparameter_grid: dict[str, Any]
    probability_calibration: dict[str, Any]
    selection: dict[str, Any]
    negative_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def write_json(self, path: str | Path) -> Path:
        return _write_immutable_json(path, self.to_dict(), "discovery report")


def regenerate_champion_discovery_report(study: DiscoveryStudy) -> FrozenDiscoveryReport:
    """Select one arm using discovery evidence only and retain every negative result."""
    plan = study.plan
    raw_prediction = study.finalist.prediction
    ensemble_prediction = ENSEMBLE_BY_FINALIST[raw_prediction]
    individual: dict[str, dict[str, dict[str, float]]] = {}
    ensemble_metrics: dict[str, dict[str, dict[str, float]]] = {}
    calibration: dict[str, Any] = {}
    noise_rows: list[dict[str, Any]] = []
    collapsed: dict[str, pd.DataFrame] = {}

    for arm in study.arms:
        artifact = study.artifacts[arm.id]
        frame = artifact.frame
        _assert_discovery_folds(frame, plan.discovery_windows, arm.id)
        if set(int(seed) for seed in frame["seed"].unique()) != set(CHAMPION_SEEDS):
            raise ChampionReportError(f"arm {arm.id!r} does not retain all five seeds")
        scored = frame.loc[frame["eligibility"]].copy()
        if scored.empty:
            raise ChampionReportError(f"arm {arm.id!r} has no eligible discovery rows")
        for (fold, seed), group in scored.groupby(["fold", "seed"], sort=True):
            label = f"{fold}/seed-{int(seed)}"
            individual.setdefault(arm.id, {})[label] = ranking_metrics_by_gameweek(
                group,
                actual_column="y_points",
                prediction_column=raw_prediction,
            ).summary
        noise_rows.extend(_seed_noise(scored, arm=arm.id, prediction=raw_prediction))

        collapsed_frame = collapse_five_seed_ensemble(artifact)
        collapsed[arm.id] = collapsed_frame
        eligible = collapsed_frame.loc[collapsed_frame["eligibility"]].copy()
        ensemble_metrics[arm.id] = {}
        calibration[arm.id] = {}
        for fold, group in eligible.groupby("fold", sort=True):
            fold_name = str(fold)
            model_report = ranking_metrics_by_gameweek(
                group,
                actual_column="y_points",
                prediction_column=ensemble_prediction,
            )
            baseline_report = ranking_metrics_by_gameweek(
                group,
                actual_column="y_points",
                prediction_column="points_mean_last5",
            )
            ensemble_metrics[arm.id][fold_name] = {
                "model": model_report.summary,
                "last5": baseline_report.summary,
            }
            calibration[arm.id][fold_name] = _probability_evidence(group)

    _assert_all_arms_same_rows(collapsed)
    base_id = "full/baseline"
    fpl_only_id = "fpl_only/baseline"
    source = _source_comparison(
        full=collapsed[base_id],
        fpl_only=collapsed[fpl_only_id],
        prediction=ensemble_prediction,
        study=study,
    )
    candidate_evidence = {
        arm.id: _candidate_gate(
            candidate=collapsed[arm.id],
            baseline=collapsed[base_id],
            candidate_id=arm.id,
            baseline_id=base_id,
            prediction=ensemble_prediction,
            seed_noise=noise_rows,
            study=study,
            exact_source_comparison=source if arm.id == fpl_only_id else None,
        )
        for arm in study.arms
        if arm.id != base_id
    }
    passed = [
        (evidence["mean_ndcg_delta"], arm_id)
        for arm_id, evidence in candidate_evidence.items()
        if evidence["passed"]
    ]
    selected_id = max(passed, default=(0.0, base_id))[1]
    selected_last5_gate = _discovery_last5_gate(
        collapsed[selected_id],
        selected_id=selected_id,
        prediction=ensemble_prediction,
        seed_noise=noise_rows,
        study=study,
    )
    selection = {
        "policy": (
            "Choose on discovery only. A challenger must improve NDCG beyond five-seed "
            "movement and the paired-GW interval, while preserving Spearman, MAE, and bias."
        ),
        "baseline_arm": base_id,
        "candidate_gates": candidate_evidence,
        "selected_arm": selected_id,
        "selected_prediction": raw_prediction,
        "promotion_vs_last5": selected_last5_gate,
        "selection_windows": list(plan.discovery_windows),
        "confirmation_observed": False,
    }

    ablations = _ablation_results(study.arms, candidate_evidence, source, selected_id)
    grid = {
        arm.id: candidate_evidence[arm.id]
        for arm in study.arms
        if arm.category == "hyperparameter_grid"
    }
    negatives = _negative_results(study.arms, candidate_evidence, ablations, source, selected_id)
    return FrozenDiscoveryReport(
        schema_version="1.0.0",
        phase="discovery",
        stabilization_plan_sha256=plan.plan_hash,
        finalist=asdict(study.finalist),
        experiment={
            "hypothesis": "Stabilize the discovery finalist without adding a new architecture.",
            "data_manifest_hashes": dict(study.data_manifest_hashes),
            "feature_contract_version": BASELINE_FEATURE_CONTRACT.contract_version,
            "feature_contract_count": len(BASELINE_FEATURE_NAMES),
            "discovery_windows": list(plan.discovery_windows),
            "confirmation_window": plan.confirmation_window,
            "seeds": list(plan.seeds),
            "primary_metric": plan.primary_metric,
            "guardrails": {
                "minimum_spearman_delta": plan.minimum_spearman_delta,
                "maximum_mae_regression": plan.maximum_mae_regression,
                "maximum_absolute_bias_regression": plan.maximum_absolute_bias_regression,
                "require_nonnegative_paired_ci": plan.require_nonnegative_paired_ci,
            },
            "bootstrap_samples": plan.bootstrap_samples,
            "bootstrap_seed": plan.bootstrap_seed,
        },
        arms=[arm.to_dict() for arm in study.arms],
        individual_seed_metrics=individual,
        ensemble_metrics=ensemble_metrics,
        seed_noise=noise_rows,
        source_comparison=source,
        grouped_ablations=ablations,
        hyperparameter_grid=grid,
        probability_calibration=calibration,
        selection=selection,
        negative_results=negatives,
    )


def freeze_champion_selection(
    report: FrozenDiscoveryReport,
    study: DiscoveryStudy,
    *,
    evaluation_plan_version: str,
) -> SelectionFreeze:
    """Public typed wrapper around the discovery-only selection seal."""
    from fpl_model.models.champion_stabilization import make_selection_freeze

    return make_selection_freeze(
        discovery_report=report.to_dict(),
        study=study,
        evaluation_plan_version=evaluation_plan_version,
    )


@dataclass(frozen=True, slots=True)
class PromotionAuditInputs:
    """Files and hashes that must be pinned before a model may be promoted."""

    model_artifact_path: Path
    confirmation_artifact_path: Path
    finalist_report_path: Path
    data_manifest_hashes: Mapping[str, str]
    git_commit: str
    feature_contract_path: Path = Path("src/fpl_model/features/baseline_contract.json")
    evaluation_config_path: Path = Path("config/evaluation_windows.toml")
    stabilization_config_path: Path = Path("config/champion_stabilization.toml")
    project_file_path: Path = Path("pyproject.toml")
    python_version_path: Path = Path(".python-version")

    def __post_init__(self) -> None:
        for name in (
            "model_artifact_path",
            "confirmation_artifact_path",
            "finalist_report_path",
            "feature_contract_path",
            "evaluation_config_path",
            "stabilization_config_path",
            "project_file_path",
            "python_version_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(self, "data_manifest_hashes", dict(self.data_manifest_hashes))


@dataclass(frozen=True, slots=True)
class FrozenPromotionReport:
    schema_version: str
    phase: str
    selection: dict[str, Any]
    discovery_evidence_sha256: str
    confirmation_metrics: dict[str, Any]
    seed_noise: list[dict[str, Any]]
    paired_confidence_intervals: dict[str, Any]
    calibration: dict[str, Any]
    promotion_gate: dict[str, Any]
    audit: dict[str, Any]
    environment: dict[str, Any]
    negative_results: list[dict[str, Any]]
    decision: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def write_json(self, path: str | Path) -> Path:
        return _write_immutable_json(path, self.to_dict(), "promotion report")


def regenerate_champion_promotion_report(
    discovery_report: FrozenDiscoveryReport,
    selection: SelectionFreeze,
    audit_inputs: PromotionAuditInputs,
) -> FrozenPromotionReport:
    """Open confirmation once, audit reproducibility, and fail closed on every missing gate."""
    discovery_payload = discovery_report.to_dict()
    discovery_hash = _json_hash(discovery_payload)
    if discovery_hash != selection.created_from_discovery_sha256:
        raise ChampionReportError("selection is not sealed to this discovery report")
    if selection.selected_arm != discovery_report.selection["selected_arm"]:
        raise ChampionReportError("selection arm differs from the discovery report")
    try:
        confirmation = RetainedPredictions.read(audit_inputs.confirmation_artifact_path)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise ChampionReportError("confirmation artifact failed integrity validation") from error
    frame = confirmation.frame
    if set(frame["fold"].astype(str)) != {selection.confirmation_window}:
        raise ChampionReportError("confirmation artifact contains a non-confirmation fold")
    if set(int(seed) for seed in frame["seed"].unique()) != set(selection.seeds):
        raise ChampionReportError("confirmation artifact does not contain the frozen five seeds")
    prediction = ENSEMBLE_BY_FINALIST[selection.selected_prediction]
    ensemble = collapse_five_seed_ensemble(confirmation)
    scored = ensemble.loc[ensemble["eligibility"]].copy()
    if scored.empty:
        raise ChampionReportError("confirmation contains no eligible rows")
    model_report = ranking_metrics_by_gameweek(
        scored,
        actual_column="y_points",
        prediction_column=prediction,
    )
    baseline_report = ranking_metrics_by_gameweek(
        scored,
        actual_column="y_points",
        prediction_column="points_mean_last5",
    )
    confidence_intervals = {
        metric: _paired_payload(
            scored,
            scored,
            metric=metric,
            candidate_prediction=prediction,
            baseline_prediction="points_mean_last5",
            samples=_plan_value(discovery_report, "bootstrap_samples", default=2_000),
            seed=_plan_value(discovery_report, "bootstrap_seed", default=42),
        )
        for metric in ("ndcg_at_10", "spearman", "mae", "mean_bias")
    }
    confirmation_noise = _seed_noise(
        frame.loc[frame["eligibility"]],
        arm=selection.selected_arm,
        prediction=selection.selected_prediction,
    )
    confirmation_calibration = _probability_evidence(scored)
    environment = capture_environment(audit_inputs.project_file_path)
    audit = _reproducibility_audit(
        selection,
        discovery_report,
        audit_inputs,
        confirmation,
        environment,
    )
    gate = _promotion_gate(
        discovery_report,
        selection,
        model_summary=model_report.summary,
        baseline_summary=baseline_report.summary,
        confidence=confidence_intervals,
        seed_noise=confirmation_noise,
        confirmation_calibration=confirmation_calibration,
        audit=audit,
    )
    decision = "PROMOTE" if gate["passed"] else "DO_NOT_PROMOTE"
    failed = [name for name, passed in gate["checks"].items() if not passed]
    reason = (
        "Frozen finalist cleared discovery, confirmation, calibration, uncertainty, "
        "and audit gates."
        if not failed
        else f"Promotion blocked by gates: {', '.join(failed)}."
    )
    negatives = list(discovery_report.negative_results)
    if failed:
        negatives.append(
            {
                "experiment_id": "untouched_confirmation",
                "decision": "REJECT",
                "reason": reason,
                "failed_checks": failed,
            }
        )
    return FrozenPromotionReport(
        schema_version="1.0.0",
        phase="promotion",
        selection=selection.to_dict(),
        discovery_evidence_sha256=discovery_hash,
        confirmation_metrics={
            "model": model_report.summary,
            "last5": baseline_report.summary,
            "by_gameweek": model_report.by_gameweek.to_dict(orient="records"),
        },
        seed_noise=confirmation_noise,
        paired_confidence_intervals=confidence_intervals,
        calibration=confirmation_calibration,
        promotion_gate=gate,
        audit=audit,
        environment=environment,
        negative_results=negatives,
        decision=decision,
        reason=reason,
    )


def capture_environment(project_file_path: str | Path = "pyproject.toml") -> dict[str, Any]:
    """Capture exact runtime versions and verify every declared dependency is equality-pinned."""
    project_path = Path(project_file_path)
    with project_path.open("rb") as project_file:
        payload = tomllib.load(project_file)
    declared = [
        *payload.get("project", {}).get("dependencies", []),
        *payload.get("project", {}).get("optional-dependencies", {}).get("dev", []),
    ]
    rows = []
    all_pinned = True
    all_match = True
    for requirement in declared:
        if "==" not in requirement:
            rows.append(
                {
                    "requirement": requirement,
                    "declared_version": None,
                    "installed_version": None,
                    "pinned": False,
                    "matches": False,
                }
            )
            all_pinned = False
            all_match = False
            continue
        distribution, declared_version = requirement.split("==", maxsplit=1)
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        matches = installed == declared_version
        rows.append(
            {
                "requirement": requirement,
                "declared_version": declared_version,
                "installed_version": installed,
                "pinned": True,
                "matches": matches,
            }
        )
        all_match &= matches
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "project_file": str(project_path),
        "project_file_sha256": _file_hash(project_path),
        "dependencies": rows,
        "all_dependencies_equality_pinned": all_pinned,
        "installed_versions_match": all_match,
    }


def pin_champion_bundle(
    report: FrozenPromotionReport,
    selection: SelectionFreeze,
    audit_inputs: PromotionAuditInputs,
    destination: str | Path,
) -> Path:
    """Copy the promoted model and complete evidence into one immutable checksummed bundle."""
    if report.decision != "PROMOTE" or not report.promotion_gate.get("passed"):
        raise ChampionReportError("only a report that passed every gate may be pinned")
    if report.selection["selection_sha256"] != selection.selection_sha256:
        raise ChampionReportError("promotion report and selection freeze disagree")
    _verify_audit_sources(report.audit, audit_inputs)
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"champion bundle already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # ``tempfile.mkdtemp`` applies mode 0700, which can produce an unreadable ACL in some Windows
    # Colab/local-drive bridges. An atomic, randomly named normal directory has the same staging
    # semantics without altering inherited ACLs.
    staging = target.parent / f".{target.name}-{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        sources = {
            "model" + audit_inputs.model_artifact_path.suffix: audit_inputs.model_artifact_path,
            "confirmation.parquet": audit_inputs.confirmation_artifact_path,
            "confirmation.metadata.json": audit_inputs.confirmation_artifact_path.with_suffix(
                ".metadata.json"
            ),
            "finalist_report.json": audit_inputs.finalist_report_path,
            "baseline_contract.json": audit_inputs.feature_contract_path,
            "evaluation_windows.toml": audit_inputs.evaluation_config_path,
            "champion_stabilization.toml": audit_inputs.stabilization_config_path,
            "pyproject.toml": audit_inputs.project_file_path,
            ".python-version": audit_inputs.python_version_path,
        }
        for name, source in sources.items():
            shutil.copy2(source, staging / name)
        _write_json(staging / "selection.json", selection.to_dict())
        _write_json(staging / "promotion_report.json", report.to_dict())
        _write_json(staging / "environment.lock.json", report.environment)
        files = [
            {
                "path": path.relative_to(staging).as_posix(),
                "sha256": _file_hash(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(staging.iterdir(), key=lambda item: item.name)
            if path.is_file()
        ]
        manifest = {
            "schema_version": "1.0.0",
            "selection_sha256": selection.selection_sha256,
            "git_commit": audit_inputs.git_commit,
            "data_manifest_hashes": dict(audit_inputs.data_manifest_hashes),
            "files": files,
        }
        _write_json(staging / "champion.manifest.json", manifest)
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target.resolve()


def write_negative_results_markdown(
    report: FrozenDiscoveryReport | FrozenPromotionReport,
    path: str | Path,
) -> Path:
    """Retain human-readable reject/inconclusive results without replacing the JSON evidence."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"negative-results report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Champion stabilization negative results", ""]
    if not report.negative_results:
        lines.extend(["No negative or inconclusive result was recorded.", ""])
    for item in report.negative_results:
        lines.extend(
            [
                f"## {item.get('experiment_id', 'unnamed experiment')}",
                "",
                f"- Decision: `{item.get('decision', 'INCONCLUSIVE')}`",
                f"- Reason: {item.get('reason', 'No reason retained.')}",
                "",
            ]
        )
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination.resolve()


def _source_comparison(
    *,
    full: pd.DataFrame,
    fpl_only: pd.DataFrame,
    prediction: str,
    study: DiscoveryStudy,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "policy": (
            "Compare FPL+Understat minus FPL-only on identical operational rows and again on "
            "the identical rows with complete audited Understat coverage."
        ),
        "direction": "fpl_plus_understat_minus_fpl_only",
        "coverage": [],
        "cohorts": {},
    }
    for (fold, season), group in full.loc[full["eligibility"]].groupby(
        ["fold", "season"], sort=True
    ):
        complete = group[UNDERSTAT_ELIGIBILITY_COLUMN].astype(bool)
        output["coverage"].append(
            {
                "fold": str(fold),
                "season": str(season),
                "eligible_rows": len(group),
                "understat_complete_rows": int(complete.sum()),
                "understat_missing_rows": int((~complete).sum()),
                "understat_coverage": float(complete.mean()),
            }
        )
    masks = {
        "operational_full_coverage": full["eligibility"].astype(bool),
        "exact_understat_eligible_rows": full["eligibility"].astype(bool)
        & full[UNDERSTAT_ELIGIBILITY_COLUMN].astype(bool),
    }
    for cohort, mask in masks.items():
        candidate = full.loc[mask].copy()
        baseline_keys = pd.MultiIndex.from_frame(
            candidate[["season", "gameweek", "player_key", "fold"]]
        )
        source_keys = pd.MultiIndex.from_frame(
            fpl_only[["season", "gameweek", "player_key", "fold"]]
        )
        baseline = fpl_only.loc[source_keys.isin(baseline_keys)].copy()
        output["cohorts"][cohort] = {
            "row_count": len(candidate),
            "folds": _comparisons_by_fold(
                candidate,
                baseline,
                prediction=prediction,
                study=study,
            )
            if not candidate.empty
            else {},
            "available": not candidate.empty,
        }
    return output


def _candidate_gate(
    *,
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate_id: str,
    baseline_id: str,
    prediction: str,
    seed_noise: list[dict[str, Any]],
    study: DiscoveryStudy,
    exact_source_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    fold_results: dict[str, Any] = {}
    for fold in study.plan.discovery_windows:
        candidate_fold = candidate.loc[candidate["fold"].eq(fold) & candidate["eligibility"]]
        baseline_fold = baseline.loc[baseline["fold"].eq(fold) & baseline["eligibility"]]
        candidate_summary = ranking_metrics_by_gameweek(
            candidate_fold,
            actual_column="y_points",
            prediction_column=prediction,
        ).summary
        baseline_summary = ranking_metrics_by_gameweek(
            baseline_fold,
            actual_column="y_points",
            prediction_column=prediction,
        ).summary
        ndcg = _paired_payload(
            candidate_fold,
            baseline_fold,
            metric="ndcg_at_10",
            candidate_prediction=prediction,
            baseline_prediction=prediction,
            samples=study.plan.bootstrap_samples,
            seed=study.plan.bootstrap_seed,
        )
        noise = max(
            _noise_floor(seed_noise, candidate_id, fold),
            _noise_floor(seed_noise, baseline_id, fold),
        )
        deltas = {
            "ndcg_at_10": candidate_summary["ndcg_at_10"] - baseline_summary["ndcg_at_10"],
            "spearman": candidate_summary["spearman"] - baseline_summary["spearman"],
            "mae": candidate_summary["mae"] - baseline_summary["mae"],
            "absolute_bias": abs(candidate_summary["mean_bias"])
            - abs(baseline_summary["mean_bias"]),
        }
        checks = {
            "ndcg_exceeds_seed_noise": deltas["ndcg_at_10"] > noise,
            "paired_ci_nonnegative": (
                ndcg["interval_low"] >= 0 if study.plan.require_nonnegative_paired_ci else True
            ),
            "spearman_guardrail": deltas["spearman"] >= study.plan.minimum_spearman_delta,
            "mae_guardrail": deltas["mae"] <= study.plan.maximum_mae_regression,
            "absolute_bias_guardrail": deltas["absolute_bias"]
            <= study.plan.maximum_absolute_bias_regression,
        }
        fold_results[fold] = {
            "candidate": candidate_summary,
            "baseline": baseline_summary,
            "deltas": deltas,
            "seed_noise_floor": noise,
            "ndcg_paired": ndcg,
            "checks": checks,
            "passed": all(checks.values()),
        }
    exact_check = True
    if exact_source_comparison is not None:
        exact = exact_source_comparison["cohorts"]["exact_understat_eligible_rows"]
        if exact["available"]:
            # Candidate is FPL-only, while the stored comparison direction is full minus FPL-only.
            exact_check = all(
                fold["ndcg_at_10"]["mean_delta"] <= 0 for fold in exact["folds"].values()
            )
        else:
            exact_check = False
        fold_results["exact_source_guardrail"] = exact_check
    return {
        "candidate_arm": candidate_id,
        "baseline_arm": baseline_id,
        "folds": fold_results,
        "mean_ndcg_delta": float(
            np.mean(
                [
                    row["deltas"]["ndcg_at_10"]
                    for row in fold_results.values()
                    if isinstance(row, dict)
                ]
            )
        ),
        "exact_source_guardrail": exact_check,
        "passed": all(row["passed"] for row in fold_results.values() if isinstance(row, dict))
        and exact_check,
    }


def _ablation_results(
    arms: Sequence[StabilizationArm],
    gates: Mapping[str, Any],
    source: dict[str, Any],
    selected_id: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for arm in arms:
        if arm.category == "grouped_ablation":
            group = arm.id.removeprefix("ablation/drop_")
            gate = gates[arm.id]
            results[group] = {
                "hypothesis": arm.hypothesis,
                "ablation_arm": arm.id,
                "decision": "DROP_GROUP" if gate["passed"] else "KEEP_GROUP_OR_INCONCLUSIVE",
                "selected": arm.id == selected_id,
                "evidence": gate,
            }
    exact = source["cohorts"]["exact_understat_eligible_rows"]
    operational = source["cohorts"]["operational_full_coverage"]
    full_positive = bool(exact["available"] and operational["available"]) and all(
        row["ndcg_at_10"]["mean_delta"] > 0
        for cohort in (exact, operational)
        for row in cohort["folds"].values()
    )
    results["understat_signal"] = {
        "hypothesis": "Audited Understat signals add value beyond FPL-only history.",
        "ablation_arm": "fpl_only/baseline",
        "decision": "KEEP_GROUP" if full_positive else "INCONCLUSIVE_OR_DROP",
        "selected": selected_id == "fpl_only/baseline",
        "evidence": source,
    }
    return results


def _negative_results(
    arms: Sequence[StabilizationArm],
    gates: Mapping[str, Any],
    ablations: Mapping[str, Any],
    source: Mapping[str, Any],
    selected_id: str,
) -> list[dict[str, Any]]:
    negatives = []
    for arm in arms:
        if arm.id == "full/baseline" or arm.id == selected_id:
            continue
        gate = gates[arm.id]
        negatives.append(
            {
                "experiment_id": arm.id,
                "hypothesis": arm.hypothesis,
                "decision": "REJECT" if not gate["passed"] else "NOT_SELECTED",
                "reason": (
                    "Did not clear every discovery noise, uncertainty, and guardrail check."
                    if not gate["passed"]
                    else "Cleared its gate but another predeclared arm had higher mean NDCG gain."
                ),
            }
        )
    if ablations["understat_signal"]["decision"] != "KEEP_GROUP":
        negatives.append(
            {
                "experiment_id": "source/fpl_plus_understat",
                "hypothesis": ablations["understat_signal"]["hypothesis"],
                "decision": "INCONCLUSIVE",
                "reason": (
                    "FPL+Understat was not directionally better in every discovery fold on both "
                    "the exact eligible and full operational cohorts."
                ),
                "coverage": {
                    name: cohort["row_count"] for name, cohort in source["cohorts"].items()
                },
            }
        )
    return negatives


def _probability_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    heads = {
        "play_any": ("y_play_any", "p_play_any_ensemble", "p_play_any_historical"),
        "minutes_60": (
            "y_minutes_60",
            "p_minutes_60_ensemble",
            "p_minutes_60_historical",
        ),
    }
    output = {}
    for name, (target, model, baseline) in heads.items():
        model_report = probability_metrics_by_gameweek(
            frame,
            target_column=target,
            probability_column=model,
        )
        baseline_report = probability_metrics_by_gameweek(
            frame,
            target_column=target,
            probability_column=baseline,
        )
        output[name] = {
            "model": model_report.summary,
            "historical": baseline_report.summary,
            "brier_delta": model_report.summary["brier"] - baseline_report.summary["brier"],
            "passed": model_report.summary["brier"] <= baseline_report.summary["brier"],
            "expected_calibration_error": model_report.expected_calibration_error,
            "calibration_intercept": model_report.calibration_intercept,
            "calibration_slope": model_report.calibration_slope,
        }
    return output


def _discovery_last5_gate(
    frame: pd.DataFrame,
    *,
    selected_id: str,
    prediction: str,
    seed_noise: Sequence[Mapping[str, Any]],
    study: DiscoveryStudy,
) -> dict[str, Any]:
    folds = {}
    for fold in study.plan.discovery_windows:
        rows = frame.loc[frame["fold"].eq(fold) & frame["eligibility"]]
        model = ranking_metrics_by_gameweek(
            rows, actual_column="y_points", prediction_column=prediction
        ).summary
        baseline = ranking_metrics_by_gameweek(
            rows, actual_column="y_points", prediction_column="points_mean_last5"
        ).summary
        ndcg = _paired_payload(
            rows,
            rows,
            metric="ndcg_at_10",
            candidate_prediction=prediction,
            baseline_prediction="points_mean_last5",
            samples=study.plan.bootstrap_samples,
            seed=study.plan.bootstrap_seed,
        )
        noise = _noise_floor(seed_noise, selected_id, fold)
        deltas = {
            "ndcg_at_10": model["ndcg_at_10"] - baseline["ndcg_at_10"],
            "spearman": model["spearman"] - baseline["spearman"],
            "mae": model["mae"] - baseline["mae"],
            "absolute_bias": abs(model["mean_bias"]) - abs(baseline["mean_bias"]),
        }
        checks = {
            "ndcg_exceeds_seed_noise": deltas["ndcg_at_10"] > noise,
            "paired_ci_nonnegative": (
                ndcg["interval_low"] >= 0 if study.plan.require_nonnegative_paired_ci else True
            ),
            "spearman_guardrail": deltas["spearman"] >= study.plan.minimum_spearman_delta,
            "mae_guardrail": deltas["mae"] <= study.plan.maximum_mae_regression,
            "absolute_bias_guardrail": deltas["absolute_bias"]
            <= study.plan.maximum_absolute_bias_regression,
        }
        folds[fold] = {
            "model": model,
            "last5": baseline,
            "deltas": deltas,
            "seed_noise_floor": noise,
            "ndcg_paired": ndcg,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return {
        "policy": "The frozen discovery winner must also clear the Last-5 promotion baseline.",
        "folds": folds,
        "passed": all(row["passed"] for row in folds.values()),
    }


def _comparisons_by_fold(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    prediction: str,
    study: DiscoveryStudy,
) -> dict[str, Any]:
    output = {}
    for fold in study.plan.discovery_windows:
        left = candidate.loc[candidate["fold"].eq(fold)]
        right = baseline.loc[baseline["fold"].eq(fold)]
        if left.empty or right.empty:
            continue
        output[fold] = {
            metric: _paired_payload(
                left,
                right,
                metric=metric,
                candidate_prediction=prediction,
                baseline_prediction=prediction,
                samples=study.plan.bootstrap_samples,
                seed=study.plan.bootstrap_seed,
            )
            for metric in _METRICS
        }
    return output


def _paired_payload(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    metric: str,
    candidate_prediction: str,
    baseline_prediction: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    comparison = compare_ranking_predictions(
        candidate,
        baseline,
        metric=metric,
        candidate_prediction=candidate_prediction,
        baseline_prediction=baseline_prediction,
        actual_column="y_points",
        bootstrap_samples=samples,
        seed=seed,
    )
    return {
        "mean_delta": comparison.mean_delta,
        "improvement_delta": (
            comparison.mean_delta if metric in _HIGHER_IS_BETTER else -comparison.mean_delta
        ),
        "bootstrap_std": comparison.bootstrap_std,
        "interval_low": comparison.interval_low,
        "interval_high": comparison.interval_high,
        "confidence": comparison.confidence,
        "bootstrap_samples": comparison.bootstrap_samples,
        "by_gameweek": comparison.by_gameweek.to_dict(orient="records"),
    }


def _seed_noise(frame: pd.DataFrame, *, arm: str, prediction: str) -> list[dict[str, Any]]:
    rows = []
    for fold, fold_frame in frame.groupby("fold", sort=True):
        for metric in _METRICS:
            values = []
            for seed, seed_frame in fold_frame.groupby("seed", sort=True):
                report = ranking_metrics_by_gameweek(
                    seed_frame,
                    actual_column="y_points",
                    prediction_column=prediction,
                )
                values.append((int(seed), float(report.summary[metric])))
            numeric = np.asarray([value for _, value in values], dtype=float)
            if not np.isfinite(numeric).all():
                raise ChampionReportError(
                    f"seed metric {metric!r} is non-finite for {arm!r}/{fold!r}"
                )
            if metric == "mean_bias":
                ordered = sorted(values, key=lambda item: abs(item[1]))
            else:
                ordered = sorted(
                    values, key=lambda item: item[1], reverse=metric in _HIGHER_IS_BETTER
                )
            rows.append(
                {
                    "arm": arm,
                    "fold": str(fold),
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std(ddof=1)),
                    "best_seed": ordered[0][0],
                    "best_value": ordered[0][1],
                    "worst_seed": ordered[-1][0],
                    "worst_value": ordered[-1][1],
                    "max_pairwise_change": float(numeric.max() - numeric.min()),
                }
            )
    return rows


def _promotion_gate(
    discovery: FrozenDiscoveryReport,
    selection: SelectionFreeze,
    *,
    model_summary: Mapping[str, float],
    baseline_summary: Mapping[str, float],
    confidence: Mapping[str, Any],
    seed_noise: Sequence[Mapping[str, Any]],
    confirmation_calibration: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    discovery_rows = discovery.ensemble_metrics[selection.selected_arm]
    discovery_calibration = discovery.probability_calibration[selection.selected_arm]
    noise = max(row["max_pairwise_change"] for row in seed_noise if row["metric"] == "ndcg_at_10")
    confirmation_deltas = {
        "ndcg_at_10": model_summary["ndcg_at_10"] - baseline_summary["ndcg_at_10"],
        "spearman": model_summary["spearman"] - baseline_summary["spearman"],
        "mae": model_summary["mae"] - baseline_summary["mae"],
        "absolute_bias": abs(model_summary["mean_bias"]) - abs(baseline_summary["mean_bias"]),
    }
    discovery_passed = bool(discovery.selection["promotion_vs_last5"]["passed"])
    probability_passed = all(
        head["passed"] for fold in discovery_calibration.values() for head in fold.values()
    ) and all(head["passed"] for head in confirmation_calibration.values())
    require_ci = bool(_plan_value(discovery, "require_nonnegative_paired_ci", default=True))
    checks = {
        "selection_was_discovery_only": discovery.selection.get("confirmation_observed") is False,
        "discovery_vs_last5": discovery_passed,
        "confirmation_ndcg_exceeds_seed_noise": confirmation_deltas["ndcg_at_10"] > noise,
        "confirmation_paired_ci_nonnegative": (
            confidence["ndcg_at_10"]["interval_low"] >= 0 if require_ci else True
        ),
        "confirmation_spearman_guardrail": confirmation_deltas["spearman"] >= 0,
        "confirmation_mae_guardrail": confirmation_deltas["mae"] <= 0,
        "confirmation_absolute_bias_guardrail": confirmation_deltas["absolute_bias"] <= 0,
        "participation_calibration": probability_passed,
        "reproducibility_audit": bool(audit.get("passed")),
    }
    return {
        "policy": (
            "Promote only when the frozen arm beats Last-5 on discovery and untouched "
            "confirmation, exceeds five-seed noise, has a nonnegative paired-GW NDCG interval, "
            "preserves guardrails, calibrates participation, and passes every artifact audit."
        ),
        "discovery_folds": discovery_rows,
        "confirmation_deltas": confirmation_deltas,
        "confirmation_seed_noise_floor": noise,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _reproducibility_audit(
    selection: SelectionFreeze,
    discovery: FrozenDiscoveryReport,
    inputs: PromotionAuditInputs,
    confirmation: RetainedPredictions,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    paths = {
        "model_artifact": inputs.model_artifact_path,
        "confirmation_artifact": inputs.confirmation_artifact_path,
        "confirmation_metadata": inputs.confirmation_artifact_path.with_suffix(".metadata.json"),
        "finalist_report": inputs.finalist_report_path,
        "feature_contract": inputs.feature_contract_path,
        "evaluation_config": inputs.evaluation_config_path,
        "stabilization_config": inputs.stabilization_config_path,
        "project_file": inputs.project_file_path,
        "python_version": inputs.python_version_path,
    }
    existing = {name: path.is_file() and path.stat().st_size > 0 for name, path in paths.items()}
    hashes = {name: _file_hash(path) if existing[name] else None for name, path in paths.items()}
    data_hashes_valid = bool(inputs.data_manifest_hashes) and all(
        _SHA256_PATTERN.fullmatch(str(value)) for value in inputs.data_manifest_hashes.values()
    )
    discovery_data_hashes = discovery.experiment.get("data_manifest_hashes", {})
    feature_names_match = tuple(selection.feature_columns) == tuple(
        feature for feature in BASELINE_FEATURE_NAMES if feature in selection.feature_columns
    )
    feature_contract_matches = False
    evaluation_plan_matches = False
    stabilization_plan_matches = False
    python_version_matches = False
    if existing["feature_contract"]:
        contract = load_feature_contract(inputs.feature_contract_path)
        feature_contract_matches = contract.contract_version == discovery.experiment[
            "feature_contract_version"
        ] and set(selection.feature_columns).issubset(contract.feature_names)
    if existing["evaluation_config"]:
        evaluation_plan_matches = (
            load_evaluation_plan(inputs.evaluation_config_path).version
            == selection.evaluation_plan_version
        )
    if existing["stabilization_config"]:
        stabilization_plan_matches = (
            load_champion_stabilization_plan(inputs.stabilization_config_path).plan_hash
            == selection.stabilization_plan_sha256
        )
    if existing["python_version"]:
        requested_python = inputs.python_version_path.read_text(encoding="utf-8").strip()
        current_python = platform.python_version()
        python_version_matches = current_python == requested_python or current_python.startswith(
            requested_python + "."
        )
    checks = {
        "all_required_files_exist": all(existing.values()),
        "finalist_evidence_hash_matches": hashes["finalist_report"]
        == selection.finalist_evidence_sha256,
        "stabilization_plan_hash_matches": selection.stabilization_plan_sha256
        == discovery.stabilization_plan_sha256
        and stabilization_plan_matches,
        "evaluation_plan_version_matches": evaluation_plan_matches,
        "confirmation_metadata_verified": len(confirmation.frame) > 0,
        "data_manifest_hashes_pinned": data_hashes_valid,
        "data_manifest_hashes_match_discovery": bool(discovery_data_hashes)
        and dict(inputs.data_manifest_hashes) == discovery_data_hashes,
        "feature_contract_matches_selection": feature_names_match and feature_contract_matches,
        "git_commit_pinned": bool(_GIT_HASH_PATTERN.fullmatch(inputs.git_commit)),
        "python_version_matches": python_version_matches,
        "dependencies_equality_pinned": bool(environment["all_dependencies_equality_pinned"]),
        "installed_versions_match": bool(environment["installed_versions_match"]),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_hashes": hashes,
        "data_manifest_hashes": dict(inputs.data_manifest_hashes),
        "git_commit": inputs.git_commit,
        "feature_contract_version": "baseline-46",
        "evaluation_plan_version": selection.evaluation_plan_version,
        "selection_sha256": selection.selection_sha256,
    }


def _verify_audit_sources(audit: Mapping[str, Any], inputs: PromotionAuditInputs) -> None:
    expected = audit["source_hashes"]
    current = {
        "model_artifact": inputs.model_artifact_path,
        "confirmation_artifact": inputs.confirmation_artifact_path,
        "confirmation_metadata": inputs.confirmation_artifact_path.with_suffix(".metadata.json"),
        "finalist_report": inputs.finalist_report_path,
        "feature_contract": inputs.feature_contract_path,
        "evaluation_config": inputs.evaluation_config_path,
        "stabilization_config": inputs.stabilization_config_path,
        "project_file": inputs.project_file_path,
        "python_version": inputs.python_version_path,
    }
    for name, path in current.items():
        if not path.is_file() or _file_hash(path) != expected.get(name):
            raise ChampionReportError(f"audited source changed before pinning: {name}")


def _assert_discovery_folds(frame: pd.DataFrame, expected: Sequence[str], arm: str) -> None:
    observed = tuple(sorted(str(value) for value in frame["fold"].unique()))
    if set(observed) != set(expected):
        raise ChampionReportError(
            f"arm {arm!r} includes folds outside discovery: observed={observed!r}"
        )


def _assert_all_arms_same_rows(frames: Mapping[str, pd.DataFrame]) -> None:
    iterator = iter(frames.items())
    base_id, base = next(iterator)
    for arm_id, frame in iterator:
        try:
            assert_exact_same_rows(
                frame,
                base,
                key_columns=("season", "gameweek", "player_key", "fold"),
                target_columns=("y_points", "y_minutes", "y_play_any", "y_minutes_60"),
            )
        except ValueError as error:
            raise ChampionReportError(
                f"arm {arm_id!r} is not exactly paired with {base_id!r}"
            ) from error


def _noise_floor(rows: Sequence[Mapping[str, Any]], arm: str, fold: str) -> float:
    matches = [
        float(row["max_pairwise_change"])
        for row in rows
        if row["arm"] == arm and row["fold"] == fold and row["metric"] == "ndcg_at_10"
    ]
    if len(matches) != 1:
        raise ChampionReportError(f"missing seed-noise evidence for {arm!r}/{fold!r}")
    return matches[0]


def _plan_value(report: FrozenDiscoveryReport, name: str, *, default: Any) -> Any:
    if name in report.experiment:
        return report.experiment[name]
    if name in report.experiment.get("guardrails", {}):
        return report.experiment["guardrails"][name]
    return default


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any], label: str) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"frozen {label} already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, payload)
    return destination.resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


__all__ = [
    "ChampionReportError",
    "FrozenDiscoveryReport",
    "FrozenPromotionReport",
    "PromotionAuditInputs",
    "capture_environment",
    "freeze_champion_selection",
    "pin_champion_bundle",
    "regenerate_champion_discovery_report",
    "regenerate_champion_promotion_report",
    "write_negative_results_markdown",
]
