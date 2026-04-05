"""Config loader with directory resolution and per-file merging.

Resolution order (first found wins):
    1. QRACER_CONFIG_DIR env var
    2. ./.qracer/   (project-local)
    3. ~/.qracer/   (user default)

Merge strategy:
    - Project-local values override user-default values per file.
    - credentials.env is always loaded from ~/.qracer/ only.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
from typing import Any

from dotenv import dotenv_values

from qracer.config.models import (
    AppConfig,
    PortfolioConfig,
    ProvidersConfig,
    QracerConfig,
)

logger = logging.getLogger(__name__)

_CONFIG_DIR_NAME = ".qracer"
_CREDENTIALS_FILE = "credentials.env"

# Lazy singleton
_cached_config: QracerConfig | None = None


def _user_dir() -> Path:
    """Return ~/.qracer/."""
    return Path.home() / _CONFIG_DIR_NAME


def _project_dir() -> Path:
    """Return ./.qracer/ relative to cwd."""
    return Path.cwd() / _CONFIG_DIR_NAME


def resolve_config_dirs() -> list[Path]:
    """Return config directories in priority order (highest first).

    Only directories that actually exist on disk are returned.
    """
    candidates: list[Path] = []

    env_dir = os.environ.get("QRACER_CONFIG_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.append(_project_dir())
    candidates.append(_user_dir())

    return [d for d in candidates if d.is_dir()]


def _load_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file, returning an empty dict on failure."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning("Failed to parse %s", path, exc_info=True)
        return {}


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* into *base* (override wins on conflicts)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_merged_toml(filename: str, dirs: list[Path]) -> dict[str, Any]:
    """Load *filename* from each dir and merge (later dirs are base, earlier override)."""
    # dirs are in priority order (highest first).  We merge from lowest to highest
    # so that higher-priority values override lower ones.
    result: dict[str, Any] = {}
    for d in reversed(dirs):
        data = _load_toml(d / filename)
        result = _merge_dicts(result, data)
    return result


def _load_credentials() -> dict[str, str]:
    """Load credentials.env from the user-level directory only."""
    creds_path = _user_dir() / _CREDENTIALS_FILE
    if not creds_path.is_file():
        return {}
    values = dotenv_values(creds_path)
    return {k: v for k, v in values.items() if v is not None}


def load_config(*, force_reload: bool = False) -> QracerConfig:
    """Load and return the merged QracerConfig (lazy-cached singleton).

    Pass *force_reload=True* to bypass the cache (useful in tests).
    """
    global _cached_config  # noqa: PLW0603

    if _cached_config is not None and not force_reload:
        return _cached_config

    dirs = resolve_config_dirs()

    app_data = _load_merged_toml("config.toml", dirs)
    providers_data = _load_merged_toml("providers.toml", dirs)
    portfolio_data = _load_merged_toml("portfolio.toml", dirs)
    credentials = _load_credentials()

    config = QracerConfig(
        app=AppConfig(**app_data),
        providers=ProvidersConfig(**providers_data) if providers_data else ProvidersConfig(),
        portfolio=PortfolioConfig(**portfolio_data),
        credentials=credentials,
    )

    _cached_config = config
    return config
