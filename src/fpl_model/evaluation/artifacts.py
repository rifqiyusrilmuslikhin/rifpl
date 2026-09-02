"""Immutable retained out-of-fold prediction artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_model.evaluation.comparison import ROW_KEY_COLUMNS


class PredictionArtifactError(ValueError):
    """Raised when an OOF artifact is incomplete, ambiguous, or would be overwritten."""


@dataclass(frozen=True, slots=True)
class RetainedPredictions:
    """Validated OOF rows sufficient to rebuild evaluation reports without retraining."""

    _frame: pd.DataFrame
    prediction_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    baseline_columns: tuple[str, ...]
    key_columns: tuple[str, ...] = ROW_KEY_COLUMNS

    def __post_init__(self) -> None:
        frame = self._frame.copy(deep=True)
        required = {
            *self.key_columns,
            *self.prediction_columns,
            *self.target_columns,
            *self.baseline_columns,
            "fold",
            "seed",
            "eligibility",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise PredictionArtifactError(
                f"retained predictions are missing required columns {missing!r}"
            )
        if frame.empty:
            raise PredictionArtifactError("retained predictions must not be empty")
        identity = [*self.key_columns, "fold", "seed"]
        if frame[identity].isna().any(axis=None) or frame[identity].duplicated().any():
            raise PredictionArtifactError(
                "retained prediction identity must be unique and complete"
            )
        if frame["eligibility"].isna().any() or not frame["eligibility"].isin((True, False)).all():
            raise PredictionArtifactError("eligibility must be retained as a non-missing boolean")
        if not self.prediction_columns or not self.target_columns:
            raise PredictionArtifactError("prediction and target column lists must not be empty")
        ordered = [
            *self.key_columns,
            "fold",
            "seed",
            "eligibility",
            *self.target_columns,
            *self.prediction_columns,
            *self.baseline_columns,
        ]
        extras = [column for column in frame.columns if column not in ordered]
        object.__setattr__(self, "_frame", frame.loc[:, [*ordered, *extras]].reset_index(drop=True))

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)

    def write(
        self, path: str | Path, *, metadata: dict[str, Any] | None = None
    ) -> tuple[Path, Path]:
        """Write Parquet plus deterministic audit metadata, refusing replacement."""
        destination = Path(path)
        if destination.suffix.casefold() != ".parquet":
            raise PredictionArtifactError("retained prediction path must end in .parquet")
        metadata_path = destination.with_suffix(".metadata.json")
        if destination.exists() or metadata_path.exists():
            raise FileExistsError(f"immutable prediction artifact already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        try:
            self._frame.to_parquet(temporary, index=False)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            payload = {
                "schema_version": "1.0.0",
                "sha256": digest,
                "rows": len(self._frame),
                "columns": list(self._frame.columns),
                "key_columns": list(self.key_columns),
                "prediction_columns": list(self.prediction_columns),
                "target_columns": list(self.target_columns),
                "baseline_columns": list(self.baseline_columns),
                "metadata": metadata or {},
            }
            temporary_metadata.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, destination)
            os.replace(temporary_metadata, metadata_path)
        finally:
            temporary.unlink(missing_ok=True)
            temporary_metadata.unlink(missing_ok=True)
        return destination, metadata_path

    @classmethod
    def read(cls, path: str | Path) -> RetainedPredictions:
        source = Path(path)
        metadata_path = source.with_suffix(".metadata.json")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != payload["sha256"]:
            raise PredictionArtifactError("retained prediction artifact hash mismatch")
        frame = pd.read_parquet(source)
        if len(frame) != payload["rows"] or list(frame.columns) != payload["columns"]:
            raise PredictionArtifactError("retained prediction artifact schema/row count mismatch")
        return cls(
            frame,
            prediction_columns=tuple(payload["prediction_columns"]),
            target_columns=tuple(payload["target_columns"]),
            baseline_columns=tuple(payload["baseline_columns"]),
            key_columns=tuple(payload["key_columns"]),
        )


def retained_predictions(
    frame: pd.DataFrame,
    *,
    prediction_columns: Sequence[str],
    target_columns: Sequence[str],
    baseline_columns: Sequence[str] = (),
    key_columns: Sequence[str] = ROW_KEY_COLUMNS,
) -> RetainedPredictions:
    return RetainedPredictions(
        frame,
        tuple(prediction_columns),
        tuple(target_columns),
        tuple(baseline_columns),
        tuple(key_columns),
    )
