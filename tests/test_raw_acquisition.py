"""Offline tests for raw acquisition, manifests, and artifact integrity."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fpl_model.data.acquisition import (
    AcquisitionError,
    CurrentSeasonFPLLoader,
    HistoricalFPLLoader,
    HttpResponse,
    SourceSchemaError,
)
from fpl_model.data.manifest import ArtifactIntegrityError, RawArtifactStore

FIXTURES = Path(__file__).parent / "fixtures"
RETRIEVED_AT = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def test_raw_store_is_idempotent_and_manifest_has_provenance(tmp_path: Path) -> None:
    store = RawArtifactStore(tmp_path)
    arguments = {
        "source_name": "historical-fpl",
        "source_url": "https://example.test/2025-26/gw1.csv",
        "source_commit": "abc123",
        "seasons": ("2025-26",),
        "schema_version": "historical-gw-v1",
        "license_note": "Upstream repository license applies.",
        "media_type": "text/csv",
        "suffix": ".csv",
        "retrieved_at": RETRIEVED_AT,
    }

    first = store.store(b"player,points\n1,0\n", **arguments)
    second = store.store(b"player,points\n1,0\n", **arguments)

    assert first.path == second.path
    assert first.manifest == second.manifest
    assert first.manifest.source_commit == "abc123"
    assert first.manifest.retrieved_at_utc == "2026-09-01T12:00:00Z"
    assert first.manifest.byte_count == first.path.stat().st_size
    assert json.loads(first.manifest_path.read_text(encoding="utf-8"))["content_sha256"]
    store.verify(first)


def test_changed_content_creates_a_new_artifact_without_overwrite(tmp_path: Path) -> None:
    store = RawArtifactStore(tmp_path)
    arguments = {
        "source_name": "fpl-api",
        "source_url": "https://example.test/api/fixtures/",
        "seasons": ("2026-27",),
        "schema_version": "fixtures-v1",
        "license_note": "Public FPL API; terms may change.",
        "suffix": ".json",
        "retrieved_at": RETRIEVED_AT,
    }

    first = store.store(b"[]", **arguments)
    second = store.store(b'[{"id":1}]', **arguments)

    assert first.path != second.path
    assert first.path.read_bytes() == b"[]"
    assert second.path.read_bytes() == b'[{"id":1}]'


def test_checksum_verification_detects_tampering(tmp_path: Path) -> None:
    store = RawArtifactStore(tmp_path)
    artifact = store.store(
        b"original",
        source_name="test-source",
        source_url="https://example.test/data",
        seasons=("2026-27",),
        schema_version="v1",
        license_note="Test fixture.",
        retrieved_at=RETRIEVED_AT,
    )
    artifact.path.write_bytes(b"changed")

    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        store.verify(artifact)


def test_historical_loader_uses_offline_fixture_and_validates_columns(tmp_path: Path) -> None:
    content = (FIXTURES / "historical_gw.csv").read_bytes()
    requested_urls: list[str] = []

    def offline_fetcher(url: str) -> HttpResponse:
        requested_urls.append(url)
        return HttpResponse(content, "text/csv")

    loader = HistoricalFPLLoader(RawArtifactStore(tmp_path), offline_fetcher)
    artifact = loader.download(
        url="https://raw.example.test/revision/data/2025-26/gw1.csv",
        season="2025-26",
        source_commit="revision",
        license_note="Fixture derived from upstream schema.",
        schema_version="historical-gw-v1",
        required_csv_columns=("element", "fixture", "total_points", "minutes"),
    )

    assert requested_urls == ["https://raw.example.test/revision/data/2025-26/gw1.csv"]
    assert artifact.path.read_bytes() == content
    assert artifact.manifest.seasons == ("2025-26",)


def test_historical_schema_drift_fails_before_storage(tmp_path: Path) -> None:
    loader = HistoricalFPLLoader(
        RawArtifactStore(tmp_path),
        lambda _url: HttpResponse(b"element,fixture\n1,2\n", "text/csv"),
    )
    with pytest.raises(SourceSchemaError, match="missing required columns.*minutes"):
        loader.download(
            url="https://example.test/gw.csv",
            season="2025-26",
            source_commit="revision",
            license_note="Test fixture.",
            schema_version="v1",
            required_csv_columns=("element", "fixture", "minutes"),
        )
    assert not list(tmp_path.rglob("*.csv"))


def test_current_api_loader_validates_offline_bootstrap_fixture(tmp_path: Path) -> None:
    content = (FIXTURES / "bootstrap_static.json").read_bytes()
    loader = CurrentSeasonFPLLoader(
        RawArtifactStore(tmp_path),
        lambda _url: HttpResponse(content, "application/json"),
        base_url="https://example.test/api/",
    )

    artifact = loader.download_bootstrap(
        season="2026-27", license_note="Public API; terms may change."
    )

    assert artifact.manifest.source_url == "https://example.test/api/bootstrap-static/"
    assert artifact.manifest.schema_version == "fpl-bootstrap-v1"
    assert artifact.path.read_bytes() == content


def test_current_api_schema_drift_and_failure_are_not_stored(tmp_path: Path) -> None:
    drift_loader = CurrentSeasonFPLLoader(
        RawArtifactStore(tmp_path),
        lambda _url: HttpResponse(b'{"events": []}', "application/json"),
    )
    with pytest.raises(SourceSchemaError, match="missing required fields"):
        drift_loader.download_bootstrap(season="2026-27", license_note="Test fixture.")

    def failed_fetcher(url: str) -> HttpResponse:
        raise OSError("offline")

    failure_loader = CurrentSeasonFPLLoader(RawArtifactStore(tmp_path), failed_fetcher)
    with pytest.raises(AcquisitionError, match="offline") as failure:
        failure_loader.download_fixtures(season="2026-27", license_note="Test fixture.")

    assert failure.value.source_url.endswith("fixtures/")
    assert not list(tmp_path.rglob("*.json"))
