"""FPL model research package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fpl-model")
except PackageNotFoundError:  # pragma: no cover - useful when imported without installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
