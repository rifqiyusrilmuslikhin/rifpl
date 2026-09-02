"""Thin, deterministic serialization helpers for decision outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpl_model.decision.engine import DecisionResult
from fpl_model.decision.regret import DecisionRegretReport


def write_decision_report(result: DecisionResult, path: str | Path) -> Path:
    """Write one traceable Team ID decision without overwriting frozen evidence."""
    return _write_json(result.to_dict(), path, label="decision report")


def write_regret_report(report: DecisionRegretReport, path: str | Path) -> Path:
    """Write historical per-GW and paired regret evidence."""
    return _write_json(report.to_dict(), path, label="decision-regret report")


def decision_summary(result: DecisionResult) -> pd.DataFrame:
    """Return the display columns used by the thin notebook."""
    columns = [
        "player_name",
        "position",
        "squad_role",
        "lineup_rank",
        "bench_order",
        "is_captain",
        "is_vice_captain",
        "xpts",
        "p_play_any",
        "p_minutes_60",
        "expected_minutes",
        "data_quality_flag",
        "risk_flags",
        "decision_reason",
    ]
    return result.player_table.loc[:, columns].copy()


def _write_json(payload: dict[str, Any], path: str | Path, *, label: str) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"{label} already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination.resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
