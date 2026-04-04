"""Tests for config loader — resolution order, merge, credentials isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from tracer.config.loader import (
    _load_credentials,
    _load_merged_toml,
    _merge_dicts,
    load_config,
    resolve_config_dirs,
)
from tracer.config.models import QracerConfig


@pytest.fixture()
def user_qracer(tmp_path: Path) -> Path:
    """Create a fake ~/.qracer/ directory."""
    d = tmp_path / "home" / ".qracer"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def project_qracer(tmp_path: Path) -> Path:
    """Create a fake ./.qracer/ directory."""
    d = tmp_path / "project" / ".qracer"
    d.mkdir(parents=True)
    return d


# --- Resolution order ---


class TestResolveConfigDirs:
    def test_env_var_takes_priority(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_dir = tmp_path / "env_config"
        env_dir.mkdir()
        monkeypatch.setenv("QRACER_CONFIG_DIR", str(env_dir))
        # cwd and home may or may not have .qracer, but env_dir should be first
        dirs = resolve_config_dirs()
        assert dirs[0] == env_dir

    def test_project_local_before_user(
        self,
        user_qracer: Path,
        project_qracer: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("QRACER_CONFIG_DIR", raising=False)
        monkeypatch.setattr("tracer.config.loader._project_dir", lambda: project_qracer)
        monkeypatch.setattr("tracer.config.loader._user_dir", lambda: user_qracer)
        dirs = resolve_config_dirs()
        assert dirs == [project_qracer, user_qracer]

    def test_missing_dirs_excluded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("QRACER_CONFIG_DIR", raising=False)
        monkeypatch.setattr("tracer.config.loader._project_dir", lambda: tmp_path / "nope")
        monkeypatch.setattr("tracer.config.loader._user_dir", lambda: tmp_path / "also_nope")
        assert resolve_config_dirs() == []


# --- Merge strategy ---


class TestMergeDicts:
    def test_shallow_override(self) -> None:
        assert _merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}

    def test_deep_merge(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        assert _merge_dicts(base, override) == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_non_overlapping_keys(self) -> None:
        assert _merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


class TestMergedTomlLoading:
    def test_project_overrides_user(
        self,
        user_qracer: Path,
        project_qracer: Path,
    ) -> None:
        (user_qracer / "config.toml").write_text('default_mode = "deep"\nllm_provider = "claude"\n')
        (project_qracer / "config.toml").write_text('default_mode = "quick"\n')

        data = _load_merged_toml("config.toml", [project_qracer, user_qracer])
        assert data["default_mode"] == "quick"
        # User value preserved when project doesn't override
        assert data["llm_provider"] == "claude"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        assert _load_merged_toml("config.toml", [d]) == {}


# --- Credentials isolation ---


class TestCredentials:
    def test_credentials_loaded_from_user_dir(
        self,
        user_qracer: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (user_qracer / "credentials.env").write_text("ALPHA_VANTAGE_KEY=abc123\n")
        monkeypatch.setattr("tracer.config.loader._user_dir", lambda: user_qracer)

        creds = _load_credentials()
        assert creds == {"ALPHA_VANTAGE_KEY": "abc123"}

    def test_missing_credentials_returns_empty(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("tracer.config.loader._user_dir", lambda: tmp_path / "nope")
        assert _load_credentials() == {}


# --- Full load_config ---


class TestLoadConfig:
    def test_defaults_when_no_dirs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tracer.config.loader.resolve_config_dirs", lambda: [])
        monkeypatch.setattr("tracer.config.loader._load_credentials", lambda: {})
        cfg = load_config(force_reload=True)
        assert isinstance(cfg, QracerConfig)
        assert cfg.app.default_mode == "quick"
        assert cfg.portfolio.limits.max_single_position_pct == 15.0

    def test_lazy_caching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tracer.config.loader.resolve_config_dirs", lambda: [])
        monkeypatch.setattr("tracer.config.loader._load_credentials", lambda: {})
        cfg1 = load_config(force_reload=True)
        cfg2 = load_config()
        assert cfg1 is cfg2

    def test_force_reload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tracer.config.loader.resolve_config_dirs", lambda: [])
        monkeypatch.setattr("tracer.config.loader._load_credentials", lambda: {})
        cfg1 = load_config(force_reload=True)
        cfg2 = load_config(force_reload=True)
        assert cfg1 is not cfg2

    def test_full_integration(
        self,
        user_qracer: Path,
        project_qracer: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # User-level config
        (user_qracer / "config.toml").write_text('default_mode = "deep"\nllm_provider = "claude"\n')
        (user_qracer / "providers.toml").write_text(
            '[providers.yfinance]\nenabled = true\npriority = 100\ntier = "hot"\n'
        )
        (user_qracer / "portfolio.toml").write_text(
            'currency = "KRW"\n\n[[holdings]]\nticker = "AAPL"\nshares = 10.0\navg_cost = 150.50\n'
        )
        (user_qracer / "credentials.env").write_text("MY_KEY=secret\n")

        # Project-level override
        (project_qracer / "config.toml").write_text('default_mode = "quick"\n')

        monkeypatch.setattr("tracer.config.loader._project_dir", lambda: project_qracer)
        monkeypatch.setattr("tracer.config.loader._user_dir", lambda: user_qracer)
        monkeypatch.delenv("QRACER_CONFIG_DIR", raising=False)

        cfg = load_config(force_reload=True)

        # Project overrides user for default_mode
        assert cfg.app.default_mode == "quick"
        # User value preserved
        assert cfg.app.llm_provider == "claude"
        # Providers loaded
        assert "yfinance" in cfg.providers.providers
        # Portfolio from user
        assert cfg.portfolio.currency == "KRW"
        assert len(cfg.portfolio.holdings) == 1
        assert cfg.portfolio.holdings[0].ticker == "AAPL"
        assert cfg.portfolio.holdings[0].shares == 10.0
        assert cfg.portfolio.holdings[0].avg_cost == 150.50
        # Credentials from user dir
        assert cfg.credentials == {"MY_KEY": "secret"}
