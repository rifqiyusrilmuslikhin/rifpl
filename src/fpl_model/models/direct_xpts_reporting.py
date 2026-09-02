"""Regenerate Sprint 8 evidence and promotion gates from retained OOF predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.evaluation.artifacts import RetainedPredictions
from fpl_model.evaluation.comparison import PairingError, compare_ranking_predictions
from fpl_model.evaluation.metrics import MetricReport, ranking_metrics_by_gameweek
from fpl_model.models.direct_xpts import DIRECT_XPTS_SEEDS, DirectXptsConfig

PRIMARY_MODEL = "xpts_direct"
PRIMARY_ENSEMBLE = "xpts_direct_ensemble"
PRIMARY_RAW_ENSEMBLE = "xpts_direct_ensemble_raw"
EP_MODEL = "xpts_direct_with_ep_next"
EP_ENSEMBLE = "xpts_direct_with_ep_next_ensemble"
EP_RAW_ENSEMBLE = "xpts_direct_with_ep_next_ensemble_raw"
POINT_BASELINES = ("pred_last5_points", "pred_points_ridge")
RANKING_BASELINES = (*POINT_BASELINES, "rank_price", "ep_next")
PROMOTION_METRICS = ("ndcg_at_10", "spearman", "mae", "mean_bias")


class DirectXptsReportError(ValueError):
    """Raised when retained predictions cannot support the predeclared comparison."""


@dataclass(frozen=True, slots=True)
class FrozenDirectXptsReport:
    schema_version: str
    experiment: dict[str, Any]
    individual_seed_metrics: dict[str, dict[str, dict[str, float]]]
    ensemble_metrics: dict[str, dict[str, dict[str, float]]]
    seed_noise: list[dict[str, Any]]
    paired_comparisons: dict[str, dict[str, dict[str, dict[str, Any]]]]
    coverage: list[dict[str, Any]]
    promotion_gate: dict[str, Any]
    decision: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"frozen direct-xPts report already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination.resolve()


def regenerate_direct_xpts_report(
    artifact: RetainedPredictions,
    *,
    points_target: str = "y_points",
    experiment_id: str = "sprint-8-direct-xpts-v1",
    data_manifest_hash: str | None = None,
    window_roles: dict[str, str] | None = None,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 42,
    runtime_seconds: float | None = None,
) -> FrozenDirectXptsReport:
    """Build metrics, paired GW deltas, seed noise, and a conservative promotion decision."""
    frame = artifact.frame
    required = {
        points_target,
        PRIMARY_MODEL,
        "xpts_direct_raw",
        PRIMARY_ENSEMBLE,
        PRIMARY_RAW_ENSEMBLE,
        *POINT_BASELINES,
        "rank_price",
        "ep_next",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DirectXptsReportError(f"direct-xPts artifact is missing columns {missing!r}")
    if set(int(seed) for seed in frame["seed"].unique()) != set(DIRECT_XPTS_SEEDS):
        raise DirectXptsReportError(f"artifact must retain seeds {DIRECT_XPTS_SEEDS!r}")

    scored = frame.loc[frame["eligibility"]].copy()
    if scored.empty:
        raise DirectXptsReportError("direct-xPts report has no eligible OOF rows")
    has_ep_arm = EP_MODEL in scored.columns
    if has_ep_arm:
        ep_required = {
            "xpts_direct_with_ep_next_raw",
            EP_ENSEMBLE,
            EP_RAW_ENSEMBLE,
        }
        ep_missing = sorted(ep_required.difference(scored.columns))
        if ep_missing:
            raise DirectXptsReportError(
                f"ep_next model arm is incomplete; missing columns {ep_missing!r}"
            )
    model_columns = [PRIMARY_MODEL, "xpts_direct_raw"]
    if has_ep_arm:
        model_columns.extend((EP_MODEL, "xpts_direct_with_ep_next_raw"))

    individual_reports: dict[tuple[str, int, str], MetricReport] = {}
    individual_payload: dict[str, dict[str, dict[str, float]]] = {}
    for (fold, seed), group in scored.groupby(["fold", "seed"], sort=True):
        label = f"{fold}/seed-{int(seed)}"
        individual_payload[label] = {}
        for model in model_columns:
            report = ranking_metrics_by_gameweek(
                group, actual_column=points_target, prediction_column=model
            )
            individual_reports[(str(fold), int(seed), model)] = report
            individual_payload[label][model] = report.summary

    ensemble = _collapse_ensemble(scored, has_ep_arm=has_ep_arm)
    ensemble_models = [PRIMARY_ENSEMBLE, PRIMARY_RAW_ENSEMBLE]
    if has_ep_arm:
        ensemble_models.extend((EP_ENSEMBLE, EP_RAW_ENSEMBLE))
    ensemble_payload: dict[str, dict[str, dict[str, float]]] = {}
    paired: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for fold, group in ensemble.groupby("fold", sort=True):
        fold_name = str(fold)
        ensemble_payload[fold_name] = {}
        paired[fold_name] = {}
        for model in ensemble_models:
            report = ranking_metrics_by_gameweek(
                group, actual_column=points_target, prediction_column=model
            )
            ensemble_payload[fold_name][model] = report.summary
        for baseline in RANKING_BASELINES:
            baseline_group = group.loc[group[baseline].notna()]
            if baseline_group.empty:
                continue
            baseline_report = ranking_metrics_by_gameweek(
                baseline_group,
                actual_column=points_target,
                prediction_column=baseline,
            )
            ensemble_payload[fold_name][baseline] = baseline_report.summary

        candidates = [PRIMARY_ENSEMBLE]
        if has_ep_arm:
            candidates.append(EP_ENSEMBLE)
        for candidate in candidates:
            paired[fold_name][candidate] = {}
            for baseline in RANKING_BASELINES:
                comparison_rows = group.loc[group[baseline].notna()].copy()
                if comparison_rows.empty:
                    continue
                metrics = ("ndcg_at_10", "spearman")
                if baseline != "rank_price":
                    metrics += ("mae", "mean_bias")
                paired[fold_name][candidate][baseline] = {
                    metric: _comparison_payload(
                        comparison_rows,
                        metric=metric,
                        candidate_prediction=candidate,
                        baseline_prediction=baseline,
                        actual_column=points_target,
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_seed=bootstrap_seed,
                    )
                    for metric in metrics
                }

    seed_noise = _seed_noise(individual_reports, model_columns)
    coverage = _ep_next_coverage(ensemble)
    roles = window_roles or {str(fold): _infer_window_role(str(fold)) for fold in paired}
    gate, decision, reason = _promotion_gate(
        ensemble_payload,
        seed_noise,
        roles,
    )
    experiment = {
        "experiment_id": experiment_id,
        "hypothesis": "Fixed direct XGBoost improves GW-first ranking over Last-5.",
        "data_manifest_hash": data_manifest_hash,
        "feature_contract": "baseline-46",
        "feature_count": 46,
        "optional_arm": "ep_next" if has_ep_arm else None,
        "folds": sorted(str(fold) for fold in scored["fold"].unique()),
        "seeds": list(DIRECT_XPTS_SEEDS),
        "model": "XGBRegressor",
        "hyperparameters": DirectXptsConfig().model_parameters(DIRECT_XPTS_SEEDS[0])
        | {"random_state": "per-seed"},
        "preprocessing": "train-fold median/mode imputation plus one-hot categorical encoding",
        "early_stopping": "immediately preceding disjoint chronological calibration block",
        "primary_metric": "ndcg_at_10",
        "guardrails": ["spearman", "mae", "mean_bias"],
        "bootstrap_unit": "gameweek",
        "bootstrap_samples": bootstrap_samples,
        "runtime_seconds": runtime_seconds,
    }
    return FrozenDirectXptsReport(
        schema_version="1.0.0",
        experiment=experiment,
        individual_seed_metrics=individual_payload,
        ensemble_metrics=ensemble_payload,
        seed_noise=seed_noise,
        paired_comparisons=paired,
        coverage=coverage,
        promotion_gate=gate,
        decision=decision,
        reason=reason,
    )


def _collapse_ensemble(frame: pd.DataFrame, *, has_ep_arm: bool) -> pd.DataFrame:
    identity = ["season", "gameweek", "player_key", "fold"]
    observed_seeds = frame.groupby(identity, dropna=False)["seed"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    if not observed_seeds.map(lambda values: values == set(DIRECT_XPTS_SEEDS)).all():
        raise DirectXptsReportError("each ensemble row must retain all three declared seeds")
    ensemble_columns = [PRIMARY_ENSEMBLE, PRIMARY_RAW_ENSEMBLE]
    if has_ep_arm:
        ensemble_columns.extend((EP_ENSEMBLE, EP_RAW_ENSEMBLE))
    for column in ensemble_columns:
        variation = frame.groupby(identity, dropna=False)[column].nunique(dropna=False)
        if variation.gt(1).any():
            raise DirectXptsReportError(f"ensemble column {column!r} differs across seed rows")
    return (
        frame.sort_values([*identity, "seed"])
        .drop_duplicates(identity, keep="first")
        .reset_index(drop=True)
    )


def _comparison_payload(
    frame: pd.DataFrame,
    *,
    metric: str,
    candidate_prediction: str,
    baseline_prediction: str,
    actual_column: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    try:
        comparison = compare_ranking_predictions(
            frame,
            frame,
            metric=metric,
            candidate_prediction=candidate_prediction,
            baseline_prediction=baseline_prediction,
            actual_column=actual_column,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    except PairingError as error:
        candidate = ranking_metrics_by_gameweek(
            frame, actual_column=actual_column, prediction_column=candidate_prediction
        ).by_gameweek
        baseline = ranking_metrics_by_gameweek(
            frame, actual_column=actual_column, prediction_column=baseline_prediction
        ).by_gameweek
        paired = candidate[["season", "gameweek", metric]].merge(
            baseline[["season", "gameweek", metric]],
            on=["season", "gameweek"],
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        paired["delta"] = paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]
        if paired["delta"].isna().any():
            return {
                "candidate_minus_baseline": None,
                "improvement_delta": None,
                "bootstrap_std": None,
                "interval_low": None,
                "interval_high": None,
                "confidence": 0.95,
                "bootstrap_samples": 0,
                "unavailable_reason": str(error),
                "by_gameweek": paired.to_dict(orient="records"),
            }
        raise
    higher_is_better = metric in {"ndcg_at_10", "spearman"}
    return {
        "candidate_minus_baseline": comparison.mean_delta,
        "improvement_delta": (
            comparison.mean_delta
            if higher_is_better
            else (-comparison.mean_delta if metric == "mae" else None)
        ),
        "bootstrap_std": comparison.bootstrap_std,
        "interval_low": comparison.interval_low,
        "interval_high": comparison.interval_high,
        "confidence": comparison.confidence,
        "bootstrap_samples": comparison.bootstrap_samples,
        "by_gameweek": comparison.by_gameweek.to_dict(orient="records"),
    }


def _seed_noise(
    reports: dict[tuple[str, int, str], MetricReport],
    models: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    folds = sorted({fold for fold, _, _ in reports})
    for fold in folds:
        for model in models:
            for metric in PROMOTION_METRICS:
                values = [
                    (seed, float(reports[(fold, seed, model)].summary[metric]))
                    for seed in DIRECT_XPTS_SEEDS
                ]
                numeric = np.asarray([value for _, value in values], dtype=float)
                if not np.isfinite(numeric).all():
                    rows.append(
                        {
                            "fold": fold,
                            "model": model,
                            "metric": metric,
                            "seed_count": len(values),
                            "mean": None,
                            "std": None,
                            "best_seed": None,
                            "best_value": None,
                            "worst_seed": None,
                            "worst_value": None,
                            "max_pairwise_change": None,
                            "unavailable_reason": "one or more per-seed metrics are non-finite",
                        }
                    )
                    continue
                if metric == "mean_bias":
                    ordered = sorted(values, key=lambda item: abs(item[1]))
                elif metric in {"ndcg_at_10", "spearman"}:
                    ordered = sorted(values, key=lambda item: item[1], reverse=True)
                else:
                    ordered = sorted(values, key=lambda item: item[1])
                rows.append(
                    {
                        "fold": fold,
                        "model": model,
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


def _ep_next_coverage(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (fold, season), group in frame.groupby(["fold", "season"], sort=True):
        available = group["ep_next"].notna()
        rows.append(
            {
                "fold": str(fold),
                "season": str(season),
                "baseline": "ep_next",
                "row_count": len(group),
                "available_count": int(available.sum()),
                "missing_count": int((~available).sum()),
                "coverage": float(available.mean()),
            }
        )
    return rows


def _infer_window_role(fold: str) -> str:
    normalized = fold.casefold()
    for role in ("discovery", "confirmation", "prospective"):
        if role in normalized:
            return role
    return "unknown"


def _promotion_gate(
    summaries: dict[str, dict[str, dict[str, float]]],
    seed_noise: list[dict[str, Any]],
    roles: dict[str, str],
) -> tuple[dict[str, Any], str, str]:
    noise_lookup = {
        (row["fold"], row["metric"]): row["max_pairwise_change"]
        for row in seed_noise
        if row["model"] == PRIMARY_MODEL
    }
    evidence: dict[str, Any] = {}
    for fold in sorted(summaries):
        candidate = summaries[fold][PRIMARY_ENSEMBLE]
        baseline = summaries[fold].get("pred_last5_points")
        if baseline is None:
            raise DirectXptsReportError(f"fold {fold!r} lacks the Last-5 promotion baseline")
        ndcg_delta = candidate["ndcg_at_10"] - baseline["ndcg_at_10"]
        spearman_delta = candidate["spearman"] - baseline["spearman"]
        mae_delta = candidate["mae"] - baseline["mae"]
        absolute_bias_delta = abs(candidate["mean_bias"]) - abs(baseline["mean_bias"])
        noise_floor = noise_lookup[(fold, "ndcg_at_10")]
        checks = {
            "ndcg_improves": ndcg_delta > 0,
            "ndcg_exceeds_seed_noise": ndcg_delta > noise_floor,
            "spearman_guardrail": spearman_delta >= 0,
            "mae_guardrail": mae_delta <= 0,
            "absolute_bias_guardrail": absolute_bias_delta <= 0,
        }
        evidence[fold] = {
            "role": roles.get(fold, "unknown"),
            "candidate": {metric: candidate[metric] for metric in PROMOTION_METRICS},
            "last5": {metric: baseline[metric] for metric in PROMOTION_METRICS},
            "deltas": {
                "ndcg_at_10": ndcg_delta,
                "spearman": spearman_delta,
                "mae": mae_delta,
                "absolute_bias": absolute_bias_delta,
            },
            "seed_noise_floor": noise_floor,
            "checks": checks,
            "passed": all(checks.values()),
        }

    discovery = [value for value in evidence.values() if value["role"] == "discovery"]
    confirmation = [value for value in evidence.values() if value["role"] == "confirmation"]
    gate = {
        "policy": (
            "Beat Last-5 on NDCG beyond max seed movement, with non-worse Spearman, MAE, "
            "and absolute bias, in every discovery and confirmation block."
        ),
        "folds": evidence,
        "discovery_result": bool(discovery) and all(item["passed"] for item in discovery),
        "confirmation_result": (
            "not_available"
            if not confirmation
            else ("passed" if all(item["passed"] for item in confirmation) else "failed")
        ),
    }
    if discovery and not gate["discovery_result"]:
        return gate, "REJECT", "Direct xPts failed a discovery promotion gate; retain Last-5."
    if not discovery:
        return gate, "INCONCLUSIVE", "No discovery window was available; retain Last-5."
    if not confirmation:
        return (
            gate,
            "INCONCLUSIVE",
            "Discovery passed but confirmation is unavailable; retain Last-5.",
        )
    if gate["confirmation_result"] != "passed":
        return gate, "REJECT", "Direct xPts did not repeat on confirmation; retain Last-5."
    return gate, "KEEP", "Direct xPts cleared discovery, seed-noise, and confirmation gates."


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
