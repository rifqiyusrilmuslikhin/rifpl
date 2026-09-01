"""Dataset manifests and immutable, content-addressed raw artifact storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_FORMAT_VERSION = "1.0.0"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact no longer matches its recorded checksum."""


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable provenance record for one acquired raw payload."""

    artifact_id: str
    source_name: str
    source_url: str
    source_commit: str | None
    content_sha256: str
    retrieved_at_utc: str
    seasons: tuple[str, ...]
    schema_version: str
    license_note: str
    media_type: str
    byte_count: int
    artifact_path: str
    manifest_format_version: str = MANIFEST_FORMAT_VERSION

    def __post_init__(self) -> None:
        if not self.source_url:
            raise ValueError("source_url is required")
        if not _SHA256_PATTERN.fullmatch(self.content_sha256):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if not self.seasons or any(not season for season in self.seasons):
            raise ValueError("at least one non-empty season is required")
        if not self.schema_version or not self.license_note:
            raise ValueError("schema_version and license_note are required")
        if self.byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        timestamp = datetime.fromisoformat(self.retrieved_at_utc.replace("Z", "+00:00"))
        if timestamp.utcoffset() != UTC.utcoffset(timestamp):
            raise ValueError("retrieved_at_utc must be timezone-aware UTC")

    def to_json(self) -> str:
        payload = asdict(self)
        payload["seasons"] = list(self.seasons)
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> DatasetManifest:
        values: dict[str, Any] = json.loads(payload)
        values["seasons"] = tuple(values["seasons"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    path: Path
    manifest_path: Path
    manifest: DatasetManifest


class RawArtifactStore:
    """Write-once storage keyed by content and stable source provenance."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def store(
        self,
        content: bytes,
        *,
        source_name: str,
        source_url: str,
        seasons: tuple[str, ...],
        schema_version: str,
        license_note: str,
        source_commit: str | None = None,
        media_type: str = "application/octet-stream",
        suffix: str = ".bin",
        retrieved_at: datetime | None = None,
    ) -> StoredArtifact:
        if not isinstance(content, bytes):
            raise TypeError("raw artifact content must be bytes")
        timestamp = retrieved_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
            raise ValueError("retrieved_at must be timezone-aware UTC")

        content_hash = hashlib.sha256(content).hexdigest()
        identity_payload = json.dumps(
            {
                "content_sha256": content_hash,
                "license_note": license_note,
                "schema_version": schema_version,
                "seasons": sorted(seasons),
                "source_commit": source_commit,
                "source_name": source_name,
                "source_url": source_url,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        artifact_id = hashlib.sha256(identity_payload).hexdigest()
        source_directory = self.root / _safe_component(source_name)
        normalized_suffix = _safe_suffix(suffix)
        artifact_path = source_directory / f"{artifact_id}{normalized_suffix}"
        manifest_path = source_directory / f"{artifact_id}.manifest.json"

        if artifact_path.exists() or manifest_path.exists():
            return self._load_existing(artifact_path, manifest_path, content, artifact_id)

        source_directory.mkdir(parents=True, exist_ok=True)
        manifest = DatasetManifest(
            artifact_id=artifact_id,
            source_name=source_name,
            source_url=source_url,
            source_commit=source_commit,
            content_sha256=content_hash,
            retrieved_at_utc=timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            seasons=tuple(sorted(seasons)),
            schema_version=schema_version,
            license_note=license_note,
            media_type=media_type,
            byte_count=len(content),
            artifact_path=artifact_path.relative_to(self.root).as_posix(),
        )
        _write_exclusive(artifact_path, content)
        try:
            _write_exclusive(manifest_path, manifest.to_json().encode("utf-8"))
        except Exception:
            artifact_path.unlink(missing_ok=True)
            raise
        return StoredArtifact(artifact_path, manifest_path, manifest)

    def verify(self, artifact: StoredArtifact | DatasetManifest) -> None:
        manifest = artifact.manifest if isinstance(artifact, StoredArtifact) else artifact
        artifact_path = self.root / manifest.artifact_path
        try:
            content = artifact_path.read_bytes()
        except FileNotFoundError as error:
            raise ArtifactIntegrityError(f"raw artifact is missing: {artifact_path}") from error
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != manifest.content_sha256 or len(content) != manifest.byte_count:
            raise ArtifactIntegrityError(
                f"checksum mismatch for {manifest.artifact_id}: "
                f"expected {manifest.content_sha256}, got {actual_hash}"
            )

    def _load_existing(
        self,
        artifact_path: Path,
        manifest_path: Path,
        requested_content: bytes,
        artifact_id: str,
    ) -> StoredArtifact:
        if not artifact_path.is_file() or not manifest_path.is_file():
            raise ArtifactIntegrityError(f"incomplete raw artifact pair for {artifact_id}")
        manifest = DatasetManifest.from_json(manifest_path.read_text(encoding="utf-8"))
        stored = StoredArtifact(artifact_path, manifest_path, manifest)
        self.verify(stored)
        if artifact_path.read_bytes() != requested_content:
            raise ArtifactIntegrityError(f"artifact identity collision for {artifact_id}")
        return stored


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
    if not component:
        raise ValueError("source_name must contain a safe path character")
    return component


def _safe_suffix(value: str) -> str:
    suffix = value if value.startswith(".") else f".{value}"
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix):
        raise ValueError(f"unsafe artifact suffix: {value!r}")
    return suffix.lower()


def _write_exclusive(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
