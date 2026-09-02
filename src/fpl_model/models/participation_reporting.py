"""Regenerate Sprint 9 reliability, coherence, cohort, and promotion evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.evaluation.artifacts import RetainedPredictions
from fpl_model.evaluation.cohorts import cohort_frame, diagnostic_cohorts
from fpl_model.evaluation.comparison import (
    PairingError,
    bootstrap_gameweek_deltas,
    compare_ranking_predictions,
)
from fpl_model.evaluation.metrics import (
    ProbabilityReport,
    probability_metrics_by_gameweek,
    ranking_metrics_by_gameweek,
)
from fpl_model.models.participation import PARTICIPATION_SEEDS, ParticipationConfig

PROBABILITY_MODELS = {
    "p_play_any_raw_ensemble": "y_play_any",
    "p_play_any_calibrated_ensemble": "y_play_any",
    "p_play_any_ensemble": "y_play_any",
    "p_play_any_historical": "y_play_any",
    "p_minutes_60_raw_ensemble": "y_minutes_60",
    "p_minutes_60_calibrated_ensemble": "y_minutes_60",
    "p_minutes_60_ensemble": "y_minutes_60",
    "p_minutes_60_historical": "y_minutes_60",
}
XPTS_MODELS = (
    "xpts_direct_ensemble",
    "xpts_conditional_ensemble",
    "xpts_blend_ensemble",
)
PRIMARY_METRICS = ("ndcg_at_10", "spearman", "mae", "mean_bias")


class ParticipationReportError(ValueError):
    """Raised when retained predictions cannot support the Sprint 9 report."""


@dataclass(frozen=True, slots=True)
class FrozenParticipationReport:
    schema_version: str
    experiment: dict[str, Any]
    probability_metrics: dict[str, dict[str, Any]]
    probability_gate: dict[str, Any]
    minutes_metrics: dict[str, dict[str, float]]
    xpts_metrics: dict[str, dict[str, dict[str, float]]]
    paired_xpts_comparisons: dict[str, dict[str, dict[str, Any]]]
    cohort_reports: dict[str, dict[str, Any]]
    coherence: dict[str, Any]
    blend_weights: list[dict[str, Any]]
    seed_noise: list[dict[str, Any]]
    promotion_gate: dict[str, Any]
    decision: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"frozen participation report already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination.resolve()


def regenerate_participation_report(
    artifact: RetainedPredictions,
    *,
    points_target: str = "y_points",
    minutes_target: str = "y_minutes",
    experiment_id: str = "sprint-9-participation-v1",
    data_manifest_hash: str | None = None,
    window_roles: dict[str, str] | None = None,
    bins: int = 10,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 42,
    runtime_seconds: float | None = None,
) -> FrozenParticipationReport:
    frame = artifact.frame
    required = {
        points_target,
        minutes_target,
        "y_play_any",
        "y_minutes_60",
        "expected_minutes_ensemble",
        "p_play_any_calibrated",
        "p_minutes_60_calibrated",
        "p_play_any",
        "p_minutes_60",
        "expected_minutes_unconstrained",
        "expected_minutes",
        "gw_max_minutes",
        "blend_direct_weight",
        *PROBABILITY_MODELS,
        *XPTS_MODELS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ParticipationReportError(f"participation artifact is missing columns {missing!r}")
    if set(int(seed) for seed in frame["seed"].unique()) != set(PARTICIPATION_SEEDS):
        raise ParticipationReportError(f"artifact must retain seeds {PARTICIPATION_SEEDS!r}")
    scored = frame.loc[frame["eligibility"]].copy()
    if scored.empty:
        raise ParticipationReportError("participation report has no eligible OOF rows")
    ensemble = _collapse_ensemble(scored)
    roles = window_roles or {
        str(fold): _infer_window_role(str(fold)) for fold in ensemble["fold"].unique()
    }

    probability_metrics: dict[str, dict[str, Any]] = {}
    minutes_metrics: dict[str, dict[str, float]] = {}
    xpts_metrics: dict[str, dict[str, dict[str, float]]] = {}
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    cohort_reports: dict[str, dict[str, Any]] = {}
    for fold, group in ensemble.groupby("fold", sort=True):
        fold_name = str(fold)
        probability_metrics[fold_name] = {
            model: _probability_payload(
                probability_metrics_by_gameweek(
                    group,
                    target_column=target,
                    probability_column=model,
                    bins=bins,
                )
            )
            for model, target in PROBABILITY_MODELS.items()
        }
        minute_report = ranking_metrics_by_gameweek(
            group, actual_column=minutes_target, prediction_column="expected_minutes_ensemble"
        )
        minutes_metrics[fold_name] = {
            metric: minute_report.summary[metric]
            for metric in ("mae", "rmse", "mean_bias", "gameweek_count", "row_count")
        }
        xpts_metrics[fold_name] = {}
        paired[fold_name] = {}
        for model in XPTS_MODELS:
            xpts_metrics[fold_name][model] = ranking_metrics_by_gameweek(
                group, actual_column=points_target, prediction_column=model
            ).summary
        for candidate in XPTS_MODELS[1:]:
            paired[fold_name][candidate] = {
                metric: _comparison_payload(
                    group,
                    candidate=candidate,
                    metric=metric,
                    actual=points_target,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                for metric in PRIMARY_METRICS
            }
        cohort_reports[fold_name] = _cohort_payload(
            group,
            points_target=points_target,
            minutes_target=minutes_target,
            bins=bins,
        )

    probability_gate = _probability_gate(
        probability_metrics,
        roles,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    coherence = _coherence_report(scored)
    weights = _blend_weight_report(scored)
    seed_noise = _seed_noise(scored, points_target)
    promotion_gate, decision, reason = _promotion_gate(xpts_metrics, seed_noise, roles)
    experiment = {
        "experiment_id": experiment_id,
        "hypothesis": "Participation decomposition improves calibrated risk and xPts ranking.",
        "data_manifest_hash": data_manifest_hash,
        "feature_contract": "baseline-46",
        "folds": sorted(str(fold) for fold in scored["fold"].unique()),
        "seeds": list(PARTICIPATION_SEEDS),
        "heads": [
            "play_any",
            "minutes_60",
            "conditional_minutes",
            "conditional_points",
            "direct_points",
        ],
        "classifier": "XGBClassifier(binary:logistic) plus chronological Platt calibration",
        "regressor": "XGBRegressor(reg:squarederror) with played-row conditional fitting",
        "hyperparameters": asdict(ParticipationConfig()),
        "blend_selection": "0.05 convex grid; calibration-window NDCG, MAE tie-break",
        "primary_xpts_metric": "ndcg_at_10",
        "primary_probability_metric": "brier",
        "bootstrap_unit": "gameweek",
        "bootstrap_samples": bootstrap_samples,
        "runtime_seconds": runtime_seconds,
    }
    return FrozenParticipationReport(
        schema_version="1.0.0",
        experiment=experiment,
        probability_metrics=probability_metrics,
        probability_gate=probability_gate,
        minutes_metrics=minutes_metrics,
        xpts_metrics=xpts_metrics,
        paired_xpts_comparisons=paired,
        cohort_reports=cohort_reports,
        coherence=coherence,
        blend_weights=weights,
        seed_noise=seed_noise,
        promotion_gate=promotion_gate,
        decision=decision,
        reason=reason,
    )


def _collapse_ensemble(frame: pd.DataFrame) -> pd.DataFrame:
    identity = ["season", "gameweek", "player_key", "fold"]
    observed = frame.groupby(identity, dropna=False)["seed"].agg(
        lambda values: frozenset(int(value) for value in values)
    )
    if not observed.map(lambda values: values == set(PARTICIPATION_SEEDS)).all():
        raise ParticipationReportError("each ensemble row must retain every declared seed")
    ensemble_columns = [column for column in frame if column.endswith("_ensemble")]
    for column in ensemble_columns:
        if frame.groupby(identity, dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ParticipationReportError(f"ensemble column {column!r} differs across seeds")
    return (
        frame.sort_values([*identity, "seed"])
        .drop_duplicates(identity, keep="first")
        .reset_index(drop=True)
    )


def _probability_payload(report: ProbabilityReport) -> dict[str, Any]:
    return {
        "summary": report.summary,
        "calibration_intercept": report.calibration_intercept,
        "calibration_slope": report.calibration_slope,
        "expected_calibration_error": report.expected_calibration_error,
        "reliability": report.calibration.to_dict(orient="records"),
        "by_gameweek": report.by_gameweek.to_dict(orient="records"),
    }


def _comparison_payload(
    frame: pd.DataFrame,
    *,
    candidate: str,
    metric: str,
    actual: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    try:
        comparison = compare_ranking_predictions(
            frame,
            frame,
            metric=metric,
            candidate_prediction=candidate,
            baseline_prediction="xpts_direct_ensemble",
            actual_column=actual,
            bootstrap_samples=samples,
            seed=seed,
        )
    except PairingError as error:
        candidate_report = ranking_metrics_by_gameweek(
            frame, actual_column=actual, prediction_column=candidate
        ).by_gameweek
        direct_report = ranking_metrics_by_gameweek(
            frame, actual_column=actual, prediction_column="xpts_direct_ensemble"
        ).by_gameweek
        paired = candidate_report[["season", "gameweek", metric]].merge(
            direct_report[["season", "gameweek", metric]],
            on=["season", "gameweek"],
            suffixes=("_candidate", "_direct"),
            validate="one_to_one",
        )
        paired["delta"] = paired[f"{metric}_candidate"] - paired[f"{metric}_direct"]
        return {
            "candidate_minus_direct": None,
            "bootstrap_std": None,
            "interval_low": None,
            "interval_high": None,
            "unavailable_reason": str(error),
            "by_gameweek": paired.to_dict(orient="records"),
        }
    return {
        "candidate_minus_direct": comparison.mean_delta,
        "bootstrap_std": comparison.bootstrap_std,
        "interval_low": comparison.interval_low,
        "interval_high": comparison.interval_high,
        "by_gameweek": comparison.by_gameweek.to_dict(orient="records"),
    }


def _cohort_payload(
    frame: pd.DataFrame, *, points_target: str, minutes_target: str, bins: int
) -> dict[str, Any]:
    required = {"fixture_count", "position", "status_risk_ordinal"}
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame))
        return {"unavailable_reason": f"missing retained cohort context: {missing}"}
    reports: dict[str, Any] = {}
    cohorts = diagnostic_cohorts(
        frame,
        minutes_column=minutes_target,
        position_column="position",
        status_column="status_risk_ordinal",
    )
    for cohort in cohorts:
        selected = cohort_frame(frame, cohort)
        if selected.empty:
            continue
        reports[cohort.name] = {
            "diagnostic_only": cohort.diagnostic_only,
            "row_count": len(selected),
            "play_any": _probability_payload(
                probability_metrics_by_gameweek(
                    selected,
                    target_column="y_play_any",
                    probability_column="p_play_any_ensemble",
                    bins=bins,
                    cohort=cohort.name,
                )
            ),
            "minutes_60": _probability_payload(
                probability_metrics_by_gameweek(
                    selected,
                    target_column="y_minutes_60",
                    probability_column="p_minutes_60_ensemble",
                    bins=bins,
                    cohort=cohort.name,
                )
            ),
            "expected_minutes": ranking_metrics_by_gameweek(
                selected,
                actual_column=minutes_target,
                prediction_column="expected_minutes_ensemble",
                cohort=cohort.name,
            ).summary,
            "xpts_blend": ranking_metrics_by_gameweek(
                selected,
                actual_column=points_target,
                prediction_column="xpts_blend_ensemble",
                cohort=cohort.name,
            ).summary,
        }
    return reports


def _coherence_report(frame: pd.DataFrame) -> dict[str, Any]:
    before_probability = frame["p_minutes_60_calibrated"].gt(frame["p_play_any_calibrated"] + 1e-12)
    after_probability = frame["p_minutes_60"].gt(frame["p_play_any"] + 1e-12)
    before_lower = frame["expected_minutes_unconstrained"].lt(60.0 * frame["p_minutes_60"] - 1e-9)
    before_upper = frame["expected_minutes_unconstrained"].gt(
        frame["gw_max_minutes"] * frame["p_play_any"] + 1e-9
    )
    after_lower = frame["expected_minutes"].lt(60.0 * frame["p_minutes_60"] - 1e-9)
    after_upper = frame["expected_minutes"].gt(frame["gw_max_minutes"] * frame["p_play_any"] + 1e-9)
    return {
        "row_count": len(frame),
        "probability_violations_before": int(before_probability.sum()),
        "probability_violation_rate_before": float(before_probability.mean()),
        "probability_violations_after": int(after_probability.sum()),
        "minutes_lower_violations_before": int(before_lower.sum()),
        "minutes_upper_violations_before": int(before_upper.sum()),
        "minutes_lower_violations_after": int(after_lower.sum()),
        "minutes_upper_violations_after": int(after_upper.sum()),
        "final_coherence_passed": bool(
            not after_probability.any() and not after_lower.any() and not after_upper.any()
        ),
    }


def _blend_weight_report(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (fold, seed), group in frame.groupby(["fold", "seed"], sort=True):
        values = group["blend_direct_weight"].unique()
        if len(values) != 1:
            raise ParticipationReportError("blend weight must be fixed within each fold/seed")
        rows.append({"fold": str(fold), "seed": int(seed), "direct_weight": float(values[0])})
    return rows


def _seed_noise(frame: pd.DataFrame, points_target: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = ("xpts_direct", "xpts_conditional", "xpts_blend")
    for fold, fold_frame in frame.groupby("fold", sort=True):
        for prediction in models:
            for metric in PRIMARY_METRICS:
                values = []
                for seed, seed_frame in fold_frame.groupby("seed", sort=True):
                    report = ranking_metrics_by_gameweek(
                        seed_frame,
                        actual_column=points_target,
                        prediction_column=prediction,
                    )
                    values.append((int(seed), float(report.summary[metric])))
                numeric = np.asarray([value for _, value in values], dtype=float)
                if not np.isfinite(numeric).all():
                    continue
                if metric == "mean_bias":
                    ordered = sorted(values, key=lambda item: abs(item[1]))
                else:
                    higher = metric in {"ndcg_at_10", "spearman"}
                    ordered = sorted(values, key=lambda item: item[1], reverse=higher)
                rows.append(
                    {
                        "fold": str(fold),
                        "model": prediction,
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
        probability_heads = (
            ("p_play_any", "y_play_any"),
            ("p_minutes_60", "y_minutes_60"),
        )
        for prediction, target in probability_heads:
            for metric in ("brier", "log_loss"):
                values = []
                for seed, seed_frame in fold_frame.groupby("seed", sort=True):
                    report = probability_metrics_by_gameweek(
                        seed_frame,
                        target_column=target,
                        probability_column=prediction,
                    )
                    values.append((int(seed), float(report.summary[metric])))
                numeric = np.asarray([value for _, value in values], dtype=float)
                ordered = sorted(values, key=lambda item: item[1])
                rows.append(
                    {
                        "fold": str(fold),
                        "model": prediction,
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


def _probability_gate(
    reports: dict[str, dict[str, Any]],
    roles: dict[str, str],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    heads = {
        "play_any": ("p_play_any_ensemble", "p_play_any_historical"),
        "minutes_60": ("p_minutes_60_ensemble", "p_minutes_60_historical"),
    }
    evidence: dict[str, Any] = {}
    decisions: dict[str, str] = {}
    for head, (model, baseline) in heads.items():
        evidence[head] = {}
        for fold, fold_reports in reports.items():
            model_brier = fold_reports[model]["summary"]["brier"]
            baseline_brier = fold_reports[baseline]["summary"]["brier"]
            model_by_gw = pd.DataFrame(fold_reports[model]["by_gameweek"])
            baseline_by_gw = pd.DataFrame(fold_reports[baseline]["by_gameweek"])
            paired = model_by_gw[["season", "gameweek", "brier"]].merge(
                baseline_by_gw[["season", "gameweek", "brier"]],
                on=["season", "gameweek"],
                suffixes=("_model", "_historical"),
                validate="one_to_one",
            )
            paired["delta"] = paired["brier_model"] - paired["brier_historical"]
            uncertainty = bootstrap_gameweek_deltas(
                paired,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            evidence[head][fold] = {
                "role": roles.get(fold, "unknown"),
                "model_brier": model_brier,
                "historical_brier": baseline_brier,
                "delta": model_brier - baseline_brier,
                "bootstrap_std": uncertainty["bootstrap_std"],
                "interval_low": uncertainty["interval_low"],
                "interval_high": uncertainty["interval_high"],
                "by_gameweek": paired.to_dict(orient="records"),
                "passed": model_brier < baseline_brier,
            }
        discovery = [row for row in evidence[head].values() if row["role"] == "discovery"]
        confirmation = [row for row in evidence[head].values() if row["role"] == "confirmation"]
        if not discovery or not confirmation:
            decisions[head] = "INCONCLUSIVE"
        elif all(row["passed"] for row in (*discovery, *confirmation)):
            decisions[head] = "KEEP"
        else:
            decisions[head] = "REJECT"
    return {
        "policy": (
            "Calibrated coherent probability must beat its historical-rate baseline on Brier "
            "in every discovery and confirmation fold."
        ),
        "heads": evidence,
        "decisions": decisions,
    }


def _promotion_gate(
    summaries: dict[str, dict[str, dict[str, float]]],
    seed_noise: list[dict[str, Any]],
    roles: dict[str, str],
) -> tuple[dict[str, Any], str, str]:
    noise = {
        (row["fold"], row["model"]): row["max_pairwise_change"]
        for row in seed_noise
        if row["metric"] == "ndcg_at_10"
    }
    candidates = XPTS_MODELS[1:]
    evidence: dict[str, Any] = {}
    for candidate in candidates:
        evidence[candidate] = {}
        for fold, reports in summaries.items():
            direct = reports[XPTS_MODELS[0]]
            model = reports[candidate]
            deltas = {
                "ndcg_at_10": model["ndcg_at_10"] - direct["ndcg_at_10"],
                "spearman": model["spearman"] - direct["spearman"],
                "mae": model["mae"] - direct["mae"],
                "absolute_bias": abs(model["mean_bias"]) - abs(direct["mean_bias"]),
            }
            noise_floor = noise.get((fold, candidate.removesuffix("_ensemble")), 0.0)
            checks = {
                "ndcg_improves": deltas["ndcg_at_10"] > 0,
                "ndcg_exceeds_seed_noise": deltas["ndcg_at_10"] > noise_floor,
                "spearman_guardrail": deltas["spearman"] >= 0,
                "mae_guardrail": deltas["mae"] <= 0,
                "absolute_bias_guardrail": deltas["absolute_bias"] <= 0,
            }
            evidence[candidate][fold] = {
                "role": roles.get(fold, "unknown"),
                "deltas": deltas,
                "seed_noise_floor": noise_floor,
                "checks": checks,
                "passed": all(checks.values()),
            }

    discovery_candidates = []
    for candidate in candidates:
        discovery = [row for row in evidence[candidate].values() if row["role"] == "discovery"]
        if discovery and all(row["passed"] for row in discovery):
            mean_delta = float(np.mean([row["deltas"]["ndcg_at_10"] for row in discovery]))
            discovery_candidates.append((mean_delta, candidate))
    selected = max(discovery_candidates)[1] if discovery_candidates else None
    gate = {
        "policy": (
            "Select on discovery only; require NDCG beyond seed noise and non-worse "
            "Spearman, MAE, and absolute bias; then repeat unchanged on confirmation."
        ),
        "candidates": evidence,
        "selected_on_discovery": selected,
        "confirmation_result": "not_applicable",
    }
    if selected is None:
        return gate, "KEEP_DIRECT", "No decomposed candidate cleared the discovery gate."
    confirmation = [row for row in evidence[selected].values() if row["role"] == "confirmation"]
    if not confirmation:
        gate["confirmation_result"] = "not_available"
        return (
            gate,
            "KEEP_DIRECT",
            "Discovery selected a candidate, but confirmation is unavailable.",
        )
    if not all(row["passed"] for row in confirmation):
        gate["confirmation_result"] = "failed"
        return gate, "KEEP_DIRECT", f"{selected} failed untouched confirmation; retain direct xPts."
    gate["confirmation_result"] = "passed"
    return (
        gate,
        f"PROMOTE_{selected.upper()}",
        f"{selected} repeated its discovery win on confirmation.",
    )


def _infer_window_role(fold: str) -> str:
    normalized = fold.casefold()
    for role in ("discovery", "confirmation", "prospective"):
        if role in normalized:
            return role
    return "unknown"


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
