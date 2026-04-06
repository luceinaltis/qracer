"""CLI entrypoint — Click-based commands for qracer."""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path

import click

from tracer.config.loader import (
    _load_toml,
    _user_dir,
    load_config,
    resolve_config_dirs,
)

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).parent / "config" / "schema"

BANNER = """\
╔══════════════════════════════════════════╗
║  Tracer — conversational alpha engine    ║
╚══════════════════════════════════════════╝
Type your query, or 'quit' to exit.
"""


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """qracer — conversational investment agent CLI."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# qracer install
# ---------------------------------------------------------------------------

_CREDENTIAL_KEYS = [
    ("FINNHUB_API_KEY", "Finnhub API key"),
    ("FRED_API_KEY", "FRED API key"),
    ("TELEGRAM_BOT_TOKEN", "Telegram bot token"),
    ("TELEGRAM_CHAT_ID", "Telegram chat ID"),
]


@main.command()
def install() -> None:
    """Interactive first-time setup wizard."""
    home_dir = _user_dir()
    click.echo(f"Setting up qracer config in {home_dir}\n")

    # 1. Create directory
    home_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy default config files
    for name in ("config.toml", "providers.toml", "portfolio.toml"):
        src = SCHEMA_DIR / name
        dest = home_dir / name
        if dest.exists():
            click.echo(f"  {name} already exists, skipping.")
        else:
            shutil.copy2(src, dest)
            click.echo(f"  Created {name}")

    # 3. Prompt for credentials
    click.echo()
    creds: dict[str, str] = {}
    for env_key, label in _CREDENTIAL_KEYS:
        value = click.prompt(f"  {label}", default="", show_default=False).strip()
        if value:
            creds[env_key] = value

    # 4. Prompt for portfolio currency
    currency = click.prompt("\n  Portfolio currency", default="USD").strip()

    # 5. Write credentials.env
    creds_path = home_dir / "credentials.env"
    lines = [f"{k}={v}" for k, v in creds.items()]
    creds_path.write_text("\n".join(lines) + "\n" if lines else "")
    click.echo(f"\n  Written {creds_path}")

    # 6. Update portfolio currency if non-default
    if currency != "USD":
        portfolio_path = home_dir / "portfolio.toml"
        if portfolio_path.exists():
            text = portfolio_path.read_text(encoding="utf-8")
            text = text.replace('currency = "USD"', f'currency = "{currency}"')
            portfolio_path.write_text(text, encoding="utf-8")

    click.echo("\n✓ Setup complete!")
    click.echo("\nNext steps:")
    click.echo("  qracer status   — check configuration")
    click.echo("  qracer config   — view merged config")
    click.echo("  qracer repl     — start interactive session")


# ---------------------------------------------------------------------------
# qracer status
# ---------------------------------------------------------------------------


@main.command()
def status() -> None:
    """Show system status."""
    dirs = resolve_config_dirs()

    # Config directories
    if dirs:
        click.echo("Config directories:")
        for d in dirs:
            click.echo(f"  ✓ {d}")
    else:
        click.echo("No config directory found. Run 'qracer install' first.")
        return

    cfg = load_config(force_reload=True)

    # Providers
    click.echo("\nProviders:")
    if cfg.providers.providers:
        for name, prov in cfg.providers.providers.items():
            state = "enabled" if prov.enabled else "disabled"
            click.echo(f"  {name}: {state} (tier={prov.tier}, priority={prov.priority})")
    else:
        click.echo("  (none configured)")

    # Portfolio
    click.echo(
        f"\nPortfolio: {len(cfg.portfolio.holdings)} holdings, currency={cfg.portfolio.currency}"
    )

    # Credentials
    click.echo("\nCredentials:")
    if cfg.credentials:
        for key, val in cfg.credentials.items():
            masked = val[:2] + "***" + val[-2:] if len(val) > 4 else "***"
            click.echo(f"  {key}: {masked}")
    else:
        click.echo("  (none set)")


# ---------------------------------------------------------------------------
# qracer config
# ---------------------------------------------------------------------------


@main.command("config")
@click.option("--set", "set_value", default=None, help="Set a value: key=value")
def config_cmd(set_value: str | None) -> None:
    """Show or update current merged config."""
    if set_value is not None:
        _config_set(set_value)
        return

    cfg = load_config(force_reload=True)
    click.echo(cfg.model_dump_json(indent=2))


def _config_set(pair: str) -> None:
    """Write key=value into ~/.qracer/config.toml."""
    if "=" not in pair:
        raise click.BadParameter("Expected key=value format", param_hint="--set")

    key, _, value = pair.partition("=")
    key = key.strip()
    value = value.strip()

    home_dir = _user_dir()
    config_path = home_dir / "config.toml"

    if not config_path.exists():
        click.echo("No config.toml found. Run 'qracer install' first.")
        raise SystemExit(1)

    # Load existing, update, and write back
    data = _load_toml(config_path)
    data[key] = value
    _write_toml(config_path, data)
    click.echo(f"Set {key}={value} in {config_path}")


def _write_toml(path: Path, data: dict) -> None:  # type: ignore[type-arg]
    """Write a flat dict back to TOML."""
    lines: list[str] = []
    for k, v in data.items():
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        else:
            lines.append(f'{k} = "{v}"')
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# qracer repl
# ---------------------------------------------------------------------------


def _build_registries() -> tuple:  # type: ignore[type-arg]
    """Build default LLM and data registries."""
    from tracer.data.registry import DataRegistry
    from tracer.llm.providers import Role
    from tracer.llm.registry import LLMRegistry

    llm_registry = LLMRegistry()
    data_registry = DataRegistry()

    try:
        from tracer.llm.claude_adapter import ClaudeAdapter

        adapter = ClaudeAdapter()
        llm_registry.register(
            "claude",
            adapter,
            [Role.RESEARCHER, Role.ANALYST, Role.STRATEGIST, Role.REPORTER],
        )
    except Exception:
        logger.warning("Claude adapter unavailable — LLM calls will fail", exc_info=True)

    try:
        from tracer.data.providers import (
            FundamentalProvider,
            NewsProvider,
            PriceProvider,
        )
        from tracer.data.yfinance_adapter import YfinanceAdapter

        yf = YfinanceAdapter()
        caps: list[type] = [PriceProvider, FundamentalProvider, NewsProvider]
        data_registry.register("yfinance", yf, caps)  # type: ignore[arg-type]
    except Exception:
        logger.warning("yfinance adapter unavailable — data calls will fail", exc_info=True)

    return llm_registry, data_registry


async def _repl_loop(engine: object) -> None:
    """Run the interactive read-eval-print loop."""
    click.echo(BANNER)

    while True:
        try:
            user_input = input("tracer> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            click.echo("Goodbye.")
            break

        if user_input.lower() in ("save", "save analysis", "/save"):
            path = engine.save_last_report()  # type: ignore[attr-defined]
            if path:
                click.echo(f"Saved to {path}\n")
            else:
                click.echo("No analysis to save.\n")
            continue

        if user_input.lower() in ("save json", "/save json"):
            path = engine.save_last_report(fmt="json")  # type: ignore[attr-defined]
            if path:
                click.echo(f"Saved to {path}\n")
            else:
                click.echo("No analysis to save.\n")
            continue

        try:
            response = await engine.query(user_input)  # type: ignore[attr-defined]
            click.echo()
            click.echo(response.text)
            click.echo()
        except Exception:
            logger.exception("Error processing query")
            click.echo("An error occurred. Check logs for details.\n")


@main.command()
def repl() -> None:
    """Start interactive conversational session."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    import uuid

    from tracer.conversation.engine import ConversationEngine
    from tracer.memory.session_logger import SessionLogger

    llm_registry, data_registry = _build_registries()

    # Create session logger in ~/.qracer/sessions/<uuid>.jsonl
    sessions_dir = _user_dir() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    session_logger = SessionLogger(sessions_dir / f"{session_id}.jsonl")

    reports_dir = _user_dir() / "reports"

    engine = ConversationEngine(
        llm_registry,
        data_registry,
        session_logger=session_logger,
        report_dir=reports_dir,
    )
    asyncio.run(_repl_loop(engine))


if __name__ == "__main__":
    main()
