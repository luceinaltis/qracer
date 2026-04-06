"""CLI entrypoint — Click-based commands for qracer."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import click

from qracer.config.loader import (
    _load_toml,
    _user_dir,
    load_config,
    resolve_config_dirs,
)

logger = logging.getLogger(__name__)

SCHEMA_DIR = Path(__file__).parent / "config" / "schema"

BANNER = """\
╔══════════════════════════════════════════╗
║  qracer — conversational alpha engine   ║
╚══════════════════════════════════════════╝
Type your query, or 'quit' to exit.
Commands: save, save json, help
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


_LLM_DISPLAY_NAMES: dict[str, str] = {
    "claude": "Claude (Anthropic)",
    "openai": "OpenAI (GPT-4o)",
    "gemini": "Gemini (Google)",
}


def _collect_llm_choices() -> list[tuple[str, str, str | None]]:
    """Return ``[(name, display_label, api_key_env), ...]`` for LLM providers."""
    schema_data = _load_toml(SCHEMA_DIR / "providers.toml")
    result: list[tuple[str, str, str | None]] = []
    for name, cfg in schema_data.get("providers", {}).items():
        if cfg.get("kind") == "llm":
            label = _LLM_DISPLAY_NAMES.get(name) or name.capitalize()
            env: str | None = cfg.get("api_key_env")
            result.append((name, label, env))
    return result


def _update_provider_selection(
    providers_path: Path,
    chosen: str,
    all_llm: list[tuple[str, str, str | None]],
) -> None:
    """Enable the chosen LLM provider and disable others in providers.toml."""
    if not providers_path.exists():
        return
    text = providers_path.read_text(encoding="utf-8")
    for name, _, _ in all_llm:
        enable = "true" if name == chosen else "false"
        pattern = rf"(\[providers\.{re.escape(name)}\][^\[]*?)enabled\s*=\s*(true|false)"
        text = re.sub(pattern, rf"\1enabled = {enable}", text)
    providers_path.write_text(text, encoding="utf-8")


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

    # 3. LLM provider selection
    llm_choices = _collect_llm_choices()
    click.echo("\n  Select LLM provider:")
    for i, (_, label, _) in enumerate(llm_choices, 1):
        click.echo(f"    {i}. {label}")

    choice_idx = click.prompt("  Choice", type=click.IntRange(1, len(llm_choices)), default=1)
    chosen_name, chosen_label, chosen_env = llm_choices[choice_idx - 1]

    # 4. Prompt for the chosen provider's API key only
    creds: dict[str, str] = {}
    if chosen_env:
        value = click.prompt(f"\n  {chosen_env}", default="", show_default=False).strip()
        if value:
            creds[chosen_env] = value

    # 5. Prompt for portfolio currency
    currency = click.prompt("\n  Portfolio currency", default="USD").strip()

    # 6. Write credentials.env
    creds_path = home_dir / "credentials.env"
    lines = [f"{k}={v}" for k, v in creds.items()]
    creds_path.write_text("\n".join(lines) + "\n" if lines else "")

    # 7. Enable chosen LLM provider in providers.toml
    _update_provider_selection(home_dir / "providers.toml", chosen_name, llm_choices)

    # 8. Update portfolio currency if non-default
    if currency != "USD":
        portfolio_path = home_dir / "portfolio.toml"
        if portfolio_path.exists():
            text = portfolio_path.read_text(encoding="utf-8")
            text = text.replace('currency = "USD"', f'currency = "{currency}"')
            portfolio_path.write_text(text, encoding="utf-8")

    click.echo(f"\n✓ Setup complete! {chosen_label} enabled as LLM provider.")

    # Warn if API key was not provided
    if chosen_env and not creds.get(chosen_env) and not os.environ.get(chosen_env):
        click.echo(f"\n⚠ {chosen_env} not set — set it before running 'qracer repl'.")

    click.echo("\nNext steps:")
    click.echo("  qracer status   — check configuration")
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

    # Data providers
    click.echo("\nData providers:")
    data_providers = {n: p for n, p in cfg.providers.providers.items() if p.kind == "data"}
    if data_providers:
        for name, prov in data_providers.items():
            state = "enabled" if prov.enabled else "disabled"
            click.echo(f"  {name}: {state} (tier={prov.tier}, priority={prov.priority})")
    else:
        click.echo("  (none configured)")

    # LLM providers
    click.echo("\nLLM providers:")
    llm_providers = {n: p for n, p in cfg.providers.providers.items() if p.kind == "llm"}
    if llm_providers:
        for name, prov in llm_providers.items():
            env_key = prov.api_key_env
            has_key = bool(env_key and (cfg.credentials.get(env_key) or os.environ.get(env_key)))
            if not prov.enabled:
                click.echo(f"  {name}: disabled")
            elif has_key:
                click.echo(f"  ✓ {name}: ready")
            else:
                click.echo(f"  ✗ {name}: {env_key} not set")
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
    """Build LLM and data registries from providers.toml + provider catalog."""
    import importlib

    from qracer.data.registry import DataRegistry
    from qracer.llm.providers import Role
    from qracer.llm.registry import LLMRegistry
    from qracer.provider_catalog import BUILTIN_DATA_PROVIDERS, BUILTIN_LLM_PROVIDERS

    config = load_config()
    llm_registry = LLMRegistry()
    data_registry = DataRegistry()

    sorted_providers = sorted(
        config.providers.providers.items(),
        key=lambda item: item[1].priority,
    )

    for name, prov_cfg in sorted_providers:
        if not prov_cfg.enabled:
            continue

        if prov_cfg.kind == "data" and name in BUILTIN_DATA_PROVIDERS:
            adapter_path, cap_paths = BUILTIN_DATA_PROVIDERS[name]
            try:
                mod_path, cls_name = adapter_path.rsplit(".", 1)
                adapter_cls = getattr(importlib.import_module(mod_path), cls_name)
                # Inject API key from credentials if declared
                api_key = None
                if prov_cfg.api_key_env:
                    api_key = config.credentials.get(prov_cfg.api_key_env) or os.environ.get(
                        prov_cfg.api_key_env
                    )
                adapter = adapter_cls(api_key=api_key) if api_key else adapter_cls()
                caps = []
                for cp in cap_paths:
                    cp_mod, cp_name = cp.rsplit(".", 1)
                    caps.append(getattr(importlib.import_module(cp_mod), cp_name))
                data_registry.register(name, adapter, caps)
            except Exception:
                logger.warning("Data provider '%s' unavailable", name, exc_info=True)

        elif prov_cfg.kind == "llm" and name in BUILTIN_LLM_PROVIDERS:
            adapter_path, role_values = BUILTIN_LLM_PROVIDERS[name]
            try:
                mod_path, cls_name = adapter_path.rsplit(".", 1)
                adapter_cls = getattr(importlib.import_module(mod_path), cls_name)
                # Inject API key from credentials if declared
                api_key = None
                if prov_cfg.api_key_env:
                    api_key = config.credentials.get(prov_cfg.api_key_env) or os.environ.get(
                        prov_cfg.api_key_env
                    )
                adapter = adapter_cls(api_key=api_key)
                roles = [Role(v) for v in role_values]
                llm_registry.register(name, adapter, roles)
            except Exception:
                logger.warning("LLM provider '%s' unavailable", name, exc_info=True)

    return llm_registry, data_registry


_HELP_TEXT = """\
Available commands:
  save              Save last analysis as Markdown
  save json         Save last analysis as JSON
  watchlist         Show watchlist with current prices
  watch TICKER      Add ticker to watchlist
  unwatch TICKER    Remove ticker from watchlist
  help              Show this help
  quit              Exit

Tips:
  - Ask about any ticker: "Analyze AAPL", "Why did TSLA spike?"
  - Compare tickers: "Compare AAPL and MSFT"
  - Follow up naturally: "What about Samsung?", "More details?"

Note: qracer provides research analysis only, not investment advice.
      It cannot execute trades or predict future prices.
"""


async def _repl_loop(engine: object, watchlist: object) -> None:
    """Run the interactive read-eval-print loop."""
    from qracer.config.loader import has_config_changed

    click.echo(BANNER)

    while True:
        try:
            user_input = input("qracer> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nGoodbye.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("quit", "exit", "q"):
            click.echo("Goodbye.")
            break

        if cmd in ("help", "/help"):
            click.echo(_HELP_TEXT)
            continue

        # Hot-plug: reload config and rebuild registries if files changed
        if has_config_changed():
            try:
                llm_reg, data_reg = _build_registries()
                engine.update_registries(llm_reg, data_reg)  # type: ignore[attr-defined]
                click.echo("⟳ Configuration reloaded.\n")
            except Exception:
                logger.warning("Config reload failed", exc_info=True)

        if cmd in ("save", "save analysis", "/save"):
            path = engine.save_last_report()  # type: ignore[attr-defined]
            if path:
                click.echo(f"Saved to {path}\n")
            else:
                click.echo("No analysis to save. Run a query first.\n")
            continue

        if cmd in ("save json", "/save json"):
            path = engine.save_last_report(fmt="json")  # type: ignore[attr-defined]
            if path:
                click.echo(f"Saved to {path}\n")
            else:
                click.echo("No analysis to save. Run a query first.\n")
            continue

        # Watchlist commands
        if cmd in ("watchlist", "wl", "/watchlist"):
            _show_watchlist(watchlist)  # type: ignore[arg-type]
            continue

        if cmd.startswith(("watch ", "/watch ")):
            ticker = user_input.split(maxsplit=1)[1].strip().upper()
            if watchlist.add(ticker):  # type: ignore[attr-defined]
                click.echo(f"Added {ticker} to watchlist.\n")
            else:
                click.echo(f"{ticker} is already on your watchlist.\n")
            continue

        if cmd.startswith(("unwatch ", "/unwatch ")):
            ticker = user_input.split(maxsplit=1)[1].strip().upper()
            if watchlist.remove(ticker):  # type: ignore[attr-defined]
                click.echo(f"Removed {ticker} from watchlist.\n")
            else:
                click.echo(f"{ticker} is not on your watchlist.\n")
            continue

        # Show progress while query is processing.
        click.echo("Analyzing...", nl=False)
        try:
            response = await engine.query(user_input)  # type: ignore[attr-defined]
            click.echo("\r" + " " * 20 + "\r", nl=False)  # clear "Analyzing..."
            click.echo(response.text)
            click.echo()
        except KeyError as exc:
            click.echo("\r" + " " * 20 + "\r", nl=False)
            click.echo(f"Missing component: {exc}")
            click.echo("Hint: run 'qracer status' to check provider configuration.\n")
        except Exception as exc:
            click.echo("\r" + " " * 20 + "\r", nl=False)
            logger.exception("Error processing query")
            click.echo(f"Something went wrong: {type(exc).__name__}")
            click.echo("Hint: try rephrasing your query or check 'qracer status'.\n")


def _show_watchlist(watchlist: object) -> None:
    """Display the current watchlist."""
    from qracer.watchlist import Watchlist

    wl: Watchlist = watchlist  # type: ignore[assignment]
    if not wl.tickers:
        click.echo("Watchlist is empty. Use 'watch TICKER' to add.\n")
        return

    click.echo(f"Watchlist ({len(wl)} stocks)")
    for ticker in wl.tickers:
        click.echo(f"  {ticker}")
    click.echo()


@main.command()
def repl() -> None:
    """Start interactive conversational session."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    import uuid

    from qracer.conversation.engine import ConversationEngine
    from qracer.memory.session_logger import SessionLogger
    from qracer.watchlist import Watchlist

    llm_registry, data_registry = _build_registries()

    # Create session logger in ~/.qracer/sessions/<uuid>.jsonl
    sessions_dir = _user_dir() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    session_logger = SessionLogger(sessions_dir / f"{session_id}.jsonl")

    reports_dir = _user_dir() / "reports"
    watchlist = Watchlist(_user_dir() / "watchlist.json")

    engine = ConversationEngine(
        llm_registry,
        data_registry,
        session_logger=session_logger,
        report_dir=reports_dir,
    )
    asyncio.run(_repl_loop(engine, watchlist))


if __name__ == "__main__":
    main()
