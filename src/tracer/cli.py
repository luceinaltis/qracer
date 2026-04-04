"""CLI entrypoint — interactive REPL for the Tracer conversational engine."""

from __future__ import annotations

import asyncio
import logging
import sys

from tracer.conversation.engine import ConversationEngine
from tracer.data.registry import DataRegistry
from tracer.llm.providers import Role
from tracer.llm.registry import LLMRegistry

logger = logging.getLogger(__name__)

BANNER = """\
╔══════════════════════════════════════════╗
║  Tracer — conversational alpha engine    ║
╚══════════════════════════════════════════╝
Type your query, or 'quit' to exit.
"""


def _build_registries() -> tuple[LLMRegistry, DataRegistry]:
    """Build default LLM and data registries.

    Imports adapters lazily so the CLI can still start (with errors at
    query time) even if optional dependencies are missing.
    """
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


async def _repl(engine: ConversationEngine) -> None:
    """Run the interactive read-eval-print loop."""
    print(BANNER)

    while True:
        try:
            user_input = input("tracer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        try:
            response = await engine.query(user_input)
            print()
            print(response.text)
            print()
        except Exception:
            logger.exception("Error processing query")
            print("An error occurred. Check logs for details.\n")


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    llm_registry, data_registry = _build_registries()
    engine = ConversationEngine(llm_registry, data_registry)
    asyncio.run(_repl(engine))


if __name__ == "__main__":
    main()
