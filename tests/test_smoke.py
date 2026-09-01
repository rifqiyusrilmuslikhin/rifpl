"""Sprint 0 package and configuration smoke tests."""

from datetime import UTC

import fpl_model
from fpl_model.config import load_config


def test_package_imports() -> None:
    assert fpl_model.__version__


def test_default_configuration_is_utc() -> None:
    config = load_config()

    assert config.target_season == "2026-27"
    assert config.timezone.key == "UTC"
    assert config.timezone.utcoffset(None) == UTC.utcoffset(None)
    assert config.paths["raw"].is_absolute()
