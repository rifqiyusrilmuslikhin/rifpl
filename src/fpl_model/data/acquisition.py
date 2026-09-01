"""Network-isolated loaders for historical data and the current FPL API."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from fpl_model.data.manifest import RawArtifactStore, StoredArtifact


class AcquisitionError(RuntimeError):
    """External source was expected but could not be acquired or decoded."""

    def __init__(self, source_url: str, reason: str) -> None:
        self.source_url = source_url
        self.reason = reason
        super().__init__(f"acquisition failed for {source_url}: {reason}")


class SourceSchemaError(ValueError):
    """A raw source payload drifted from its minimum expected contract."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    content: bytes
    media_type: str = "application/octet-stream"


class Fetcher(Protocol):
    def __call__(self, url: str) -> HttpResponse: ...


def fetch_url(url: str, *, timeout_seconds: float = 30.0) -> HttpResponse:
    """Fetch bytes over HTTPS. This is the only default network boundary."""
    if urlparse(url).scheme != "https":
        raise AcquisitionError(url, "only HTTPS sources are accepted")
    request = Request(url, headers={"User-Agent": "fpl-model/0.1 raw-data-acquisition"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            media_type = response.headers.get_content_type()
            return HttpResponse(response.read(), media_type)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise AcquisitionError(url, str(error)) from error


class HistoricalFPLLoader:
    """Acquire a pinned historical file and retain it without transformation."""

    def __init__(self, store: RawArtifactStore, fetcher: Fetcher = fetch_url) -> None:
        self.store = store
        self.fetcher = fetcher

    def download(
        self,
        *,
        url: str,
        season: str,
        source_commit: str,
        license_note: str,
        schema_version: str,
        required_csv_columns: Sequence[str] = (),
    ) -> StoredArtifact:
        if not source_commit:
            raise ValueError("historical sources must pin a commit or immutable source revision")
        response = self._fetch(url)
        suffix = Path(PurePosixPath(urlparse(url).path).name).suffix or ".bin"
        if required_csv_columns:
            _validate_csv_columns(response.content, required_csv_columns, url)
            suffix = ".csv"
        return self.store.store(
            response.content,
            source_name="historical-fpl",
            source_url=url,
            source_commit=source_commit,
            seasons=(season,),
            schema_version=schema_version,
            license_note=license_note,
            media_type=response.media_type,
            suffix=suffix,
        )

    def _fetch(self, url: str) -> HttpResponse:
        try:
            return self.fetcher(url)
        except AcquisitionError:
            raise
        except Exception as error:
            raise AcquisitionError(url, str(error)) from error


JSONContract = Callable[[Any], None]


class CurrentSeasonFPLLoader:
    """Acquire current-season JSON from the public FPL API."""

    DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api/"

    def __init__(
        self,
        store: RawArtifactStore,
        fetcher: Fetcher = fetch_url,
        *,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.store = store
        self.fetcher = fetcher
        self.base_url = base_url.rstrip("/") + "/"

    def download_bootstrap(self, *, season: str, license_note: str) -> StoredArtifact:
        return self._download_json(
            endpoint="bootstrap-static/",
            season=season,
            schema_version="fpl-bootstrap-v1",
            license_note=license_note,
            validator=_validate_bootstrap,
        )

    def download_fixtures(self, *, season: str, license_note: str) -> StoredArtifact:
        return self._download_json(
            endpoint="fixtures/",
            season=season,
            schema_version="fpl-fixtures-v1",
            license_note=license_note,
            validator=_validate_fixtures,
        )

    def download_element_summary(
        self, *, element_id: int, season: str, license_note: str
    ) -> StoredArtifact:
        if element_id <= 0:
            raise ValueError("element_id must be positive")
        return self._download_json(
            endpoint=f"element-summary/{element_id}/",
            season=season,
            schema_version="fpl-element-summary-v1",
            license_note=license_note,
            validator=_validate_element_summary,
        )

    def _download_json(
        self,
        *,
        endpoint: str,
        season: str,
        schema_version: str,
        license_note: str,
        validator: JSONContract,
    ) -> StoredArtifact:
        url = urljoin(self.base_url, endpoint)
        try:
            response = self.fetcher(url)
        except AcquisitionError:
            raise
        except Exception as error:
            raise AcquisitionError(url, str(error)) from error
        try:
            payload = json.loads(response.content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SourceSchemaError(f"{url}: response is not valid JSON: {error}") from error
        validator(payload)
        return self.store.store(
            response.content,
            source_name="fpl-api",
            source_url=url,
            seasons=(season,),
            schema_version=schema_version,
            license_note=license_note,
            media_type="application/json",
            suffix=".json",
        )


def _validate_csv_columns(content: bytes, required_columns: Sequence[str], source_url: str) -> None:
    try:
        text = content.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)))
    except (UnicodeDecodeError, StopIteration, csv.Error) as error:
        raise SourceSchemaError(f"{source_url}: invalid or empty CSV: {error}") from error
    missing = sorted(set(required_columns).difference(header))
    if missing:
        raise SourceSchemaError(f"{source_url}: CSV is missing required columns {missing}")


def _require_mapping_with_lists(
    payload: Any, *, source: str, required_lists: Sequence[str]
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise SourceSchemaError(f"{source}: expected a JSON object")
    missing = [key for key in required_lists if key not in payload]
    if missing:
        raise SourceSchemaError(f"{source}: missing required fields {missing}")
    wrong = [key for key in required_lists if not isinstance(payload[key], list)]
    if wrong:
        raise SourceSchemaError(f"{source}: fields must be lists: {wrong}")
    return payload


def _validate_bootstrap(payload: Any) -> None:
    _require_mapping_with_lists(
        payload,
        source="bootstrap-static",
        required_lists=("events", "elements", "teams", "element_types"),
    )


def _validate_fixtures(payload: Any) -> None:
    if not isinstance(payload, list):
        raise SourceSchemaError("fixtures: expected a JSON list")
    required = {"id", "event", "kickoff_time", "team_h", "team_a"}
    for index, fixture in enumerate(payload):
        if not isinstance(fixture, Mapping):
            raise SourceSchemaError(f"fixtures: row {index} must be an object")
        missing = required.difference(fixture)
        if missing:
            raise SourceSchemaError(f"fixtures: row {index} missing fields {sorted(missing)}")


def _validate_element_summary(payload: Any) -> None:
    _require_mapping_with_lists(
        payload,
        source="element-summary",
        required_lists=("fixtures", "history", "history_past"),
    )
