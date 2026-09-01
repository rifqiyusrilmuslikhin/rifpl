"""Deterministic coverage and representative spot-check artifacts for feature frames."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_model.features.contract import BASELINE_FEATURE_CONTRACT, BASELINE_FEATURE_NAMES

REQUIRED_SPOT_CHECK_CASES = frozenset(
    {"single_gameweek", "double_gameweek", "debutant", "flagged_injury", "club_transfer"}
)


def build_coverage_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic missingness record per season and contracted feature."""
    _validate_frame(frame)
    records: list[dict[str, Any]] = []
    for season in sorted(frame["season"].astype(str).unique()):
        season_frame = frame.loc[frame["season"].astype(str) == season]
        total = len(season_frame)
        for definition in BASELINE_FEATURE_CONTRACT.features:
            non_missing = int(season_frame[definition.name].notna().sum())
            records.append(
                {
                    "season": season,
                    "feature": definition.name,
                    "dtype": definition.dtype,
                    "total_rows": total,
                    "non_missing_rows": non_missing,
                    "missing_rows": total - non_missing,
                    "coverage_rate": non_missing / total if total else 0.0,
                }
            )
    return pd.DataFrame.from_records(records)


def write_coverage_report(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write coverage as stable JSON or CSV, selected by destination suffix."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = build_coverage_report(frame)
    if destination.suffix.casefold() == ".csv":
        report.to_csv(destination, index=False, lineterminator="\n")
    elif destination.suffix.casefold() == ".json":
        destination.write_text(
            json.dumps(report.to_dict(orient="records"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        raise ValueError("coverage report path must end in .json or .csv")
    return destination.resolve()


def write_spot_check_report(
    frame: pd.DataFrame,
    cases: Mapping[str, Sequence[Any]],
    path: str | Path,
) -> Path:
    """Retain audited SGW, DGW, debutant, injury, and transfer feature rows.

    ``cases`` maps each required case name to ``(season, gameweek, player_key)``. Extra cases are
    permitted, but all five required archetypes must be represented exactly once.
    """
    _validate_frame(frame)
    missing_cases = REQUIRED_SPOT_CHECK_CASES.difference(cases)
    if missing_cases:
        raise ValueError(f"spot checks are missing required cases {sorted(missing_cases)!r}")
    output: list[dict[str, Any]] = []
    for case_name in sorted(cases):
        key = tuple(cases[case_name])
        if len(key) != 3:
            raise ValueError(f"spot-check case {case_name!r} must contain a 3-part key")
        selected = frame.loc[
            (frame["season"].astype(str) == str(key[0]))
            & (frame["gameweek"] == key[1])
            & (frame["player_key"].astype(str) == str(key[2]))
        ]
        if len(selected) != 1:
            raise ValueError(f"spot-check case {case_name!r} matched {len(selected)} rows")
        record = _json_record(selected.iloc[0].to_dict())
        output.append({"case": case_name, "key": list(key), "record": record})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_contract_version": BASELINE_FEATURE_CONTRACT.contract_version,
        "spot_checks": output,
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination.resolve()


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {"season", "gameweek", "player_key", *BASELINE_FEATURE_NAMES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"feature frame is missing columns {sorted(missing)!r}")
    if frame.duplicated(["season", "gameweek", "player_key"]).any():
        raise ValueError("feature frame contains duplicate player-gameweek keys")


def _json_record(record: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            output[key] = value.isoformat()
        elif isinstance(value, list):
            output[key] = [_json_scalar(item) for item in value]
        else:
            output[key] = _json_scalar(value)
    return output


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
