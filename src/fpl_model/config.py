"""Typed project configuration loaded from TOML."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_CONFIG_PATH = Path("config/project.toml")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Stable project settings shared by local and Colab runs."""

    name: str
    target_season: str
    timezone: ZoneInfo
    paths: dict[str, Path]


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load project settings, resolving relative data paths from the repository root."""
    config_path = Path(path or os.environ.get("FPL_CONFIG", DEFAULT_CONFIG_PATH)).resolve()
    with config_path.open("rb") as config_file:
        payload = tomllib.load(config_file)

    project = payload["project"]
    timezone_name = project["timezone"]
    if timezone_name != "UTC":
        raise ValueError("Project timestamps must use UTC")

    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown timezone: {timezone_name}") from error

    repository_root = config_path.parent.parent
    paths = {
        key: _resolve_path(repository_root, configured_path)
        for key, configured_path in payload["paths"].items()
    }
    return ProjectConfig(
        name=project["name"],
        target_season=project["target_season"],
        timezone=timezone,
        paths=paths,
    )


def _resolve_path(repository_root: Path, configured_path: str) -> Path:
    candidate = Path(configured_path)
    return candidate if candidate.is_absolute() else repository_root / candidate
