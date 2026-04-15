"""MEMORY.md / BOOTSTRAP.md — user-curated cross-session long-term memory.

Bridges the auto-generated Tier 2 summaries with a human-editable document
that keeps active theses, upcoming catalysts, and free-form user notes
visible at the start of every session.

Format
------

MEMORY.md is Markdown with a single machine-managed region delimited by
HTML comment markers::

    # qracer MEMORY.md

    *Last updated: 2026-04-15T12:34:56+00:00*

    <!-- BEGIN:auto -->
    ## Active Theses

    - **AAPL** (conviction 8/10): Long AAPL on AI tailwinds. ...

    ## Upcoming Catalysts

    - AAPL: AI revenue growth — Q2 2026

    <!-- END:auto -->

    ## Watchpoints

    (User-editable free text. Preserved across refreshes.)

Everything outside the ``BEGIN:auto`` / ``END:auto`` markers is user
content and is never overwritten by :func:`refresh_memory`. The auto
region is regenerated wholesale from the :class:`FactStore`.

BOOTSTRAP.md is an even simpler file — its raw text is loaded at session
start and prepended to the session history as a system message, letting
users seed the conversation with long-term preferences ("I'm a long-term
value investor") without editing code.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from qracer.memory.fact_models import PersistedThesis
from qracer.memory.fact_store import FactStore

logger = logging.getLogger(__name__)


AUTO_BEGIN = "<!-- BEGIN:auto -->"
AUTO_END = "<!-- END:auto -->"

_HEADER = "# qracer MEMORY.md"
_DEFAULT_USER_CONTENT = """\
## Watchpoints

_(User-editable. Anything outside the auto block is preserved across refreshes.)_

## User Preferences

- Risk tolerance:
- Preferred sectors:
- Position sizing:

## Notes

_Free-text notes live here._
"""


@dataclass
class MemoryDocument:
    """In-memory representation of MEMORY.md.

    ``auto_theses`` and ``auto_catalysts`` are lists of fully-rendered
    bullet lines (without the leading ``- ``). ``user_content`` is the
    raw Markdown that falls outside the machine-managed region — it is
    preserved verbatim across refreshes.
    """

    auto_theses: list[str] = field(default_factory=list)
    auto_catalysts: list[str] = field(default_factory=list)
    user_content: str = _DEFAULT_USER_CONTENT
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary_line(self) -> str:
        """One-line summary for session briefings / status output."""
        return (
            f"MEMORY.md: {len(self.auto_theses)} active theses, "
            f"{len(self.auto_catalysts)} upcoming catalysts"
        )


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _render_theses_line(thesis: PersistedThesis) -> str:
    """Render a single thesis as a MEMORY.md bullet line."""
    catalyst = thesis.catalyst
    if thesis.catalyst_date:
        catalyst = f"{catalyst} ({thesis.catalyst_date})"
    return (
        f"**{thesis.ticker}** (conviction {thesis.conviction}/10): "
        f"{thesis.summary.rstrip('.')}. "
        f"Entry ${thesis.entry_zone_low:.2f}-${thesis.entry_zone_high:.2f}, "
        f"target ${thesis.target_price:.2f}, stop ${thesis.stop_loss:.2f}. "
        f"Catalyst: {catalyst}."
    )


def _render_catalyst_line(thesis: PersistedThesis) -> str:
    """Render a single upcoming catalyst as a MEMORY.md bullet line."""
    date = thesis.catalyst_date or "TBD"
    return f"{thesis.ticker}: {thesis.catalyst} — {date}"


def render_memory(doc: MemoryDocument) -> str:
    """Render a :class:`MemoryDocument` as canonical MEMORY.md text."""
    parts: list[str] = [
        _HEADER,
        "",
        f"*Last updated: {doc.last_updated.isoformat()}*",
        "",
        AUTO_BEGIN,
        "",
        "## Active Theses",
        "",
    ]
    if doc.auto_theses:
        parts.extend(f"- {line}" for line in doc.auto_theses)
    else:
        parts.append("_No open theses yet._")
    parts.extend(["", "## Upcoming Catalysts", ""])
    if doc.auto_catalysts:
        parts.extend(f"- {line}" for line in doc.auto_catalysts)
    else:
        parts.append("_No upcoming catalysts within the horizon._")
    parts.extend(["", AUTO_END, "", doc.user_content.rstrip(), ""])
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

_AUTO_REGION_RE = re.compile(
    rf"{re.escape(AUTO_BEGIN)}(.*?){re.escape(AUTO_END)}",
    re.DOTALL,
)

_LAST_UPDATED_RE = re.compile(r"\*Last updated:\s*([^*]+?)\s*\*")

_THESES_HEADING = re.compile(r"^##\s+Active Theses\s*$", re.MULTILINE)
_CATALYSTS_HEADING = re.compile(r"^##\s+Upcoming Catalysts\s*$", re.MULTILINE)


def _extract_bullets(block: str) -> list[str]:
    """Extract bullet-line content (without the leading ``- ``) from *block*."""
    bullets: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if line.startswith("- "):
            bullets.append(line[2:].strip())
    return bullets


def _parse_auto_region(region: str) -> tuple[list[str], list[str]]:
    """Split the auto region into ``(theses, catalysts)`` bullet lists."""
    theses_match = _THESES_HEADING.search(region)
    catalysts_match = _CATALYSTS_HEADING.search(region)

    theses_block = ""
    catalysts_block = ""
    if theses_match:
        start = theses_match.end()
        end = catalysts_match.start() if catalysts_match else len(region)
        theses_block = region[start:end]
    if catalysts_match:
        catalysts_block = region[catalysts_match.end() :]

    return _extract_bullets(theses_block), _extract_bullets(catalysts_block)


def parse_memory(text: str) -> MemoryDocument:
    """Parse MEMORY.md source text into a :class:`MemoryDocument`.

    Malformed files are tolerated: missing auto region, missing
    timestamp, or entirely empty input all yield a sensible default.
    """
    if not text.strip():
        return MemoryDocument()

    last_updated = datetime.now(timezone.utc)
    m = _LAST_UPDATED_RE.search(text)
    if m:
        try:
            last_updated = datetime.fromisoformat(m.group(1).strip())
        except ValueError:
            pass

    auto_match = _AUTO_REGION_RE.search(text)
    if auto_match is None:
        # No auto region — treat entire body (minus header/timestamp) as
        # user content so we don't silently drop anything on a rewrite.
        body = text
        for pattern in (re.compile(rf"^{re.escape(_HEADER)}\s*$", re.MULTILINE), _LAST_UPDATED_RE):
            body = pattern.sub("", body, count=1)
        return MemoryDocument(
            user_content=body.strip() or _DEFAULT_USER_CONTENT,
            last_updated=last_updated,
        )

    auto_theses, auto_catalysts = _parse_auto_region(auto_match.group(1))
    user_content = (text[: auto_match.start()] + text[auto_match.end() :]).strip()
    # Strip header + last-updated line from the user content so we don't
    # duplicate them when rendering.
    user_content = re.sub(
        rf"^{re.escape(_HEADER)}\s*$", "", user_content, count=1, flags=re.MULTILINE
    )
    user_content = _LAST_UPDATED_RE.sub("", user_content, count=1)
    user_content = user_content.strip() or _DEFAULT_USER_CONTENT

    return MemoryDocument(
        auto_theses=auto_theses,
        auto_catalysts=auto_catalysts,
        user_content=user_content,
        last_updated=last_updated,
    )


# ----------------------------------------------------------------------
# Persistence + refresh
# ----------------------------------------------------------------------


def load_memory(path: Path) -> MemoryDocument:
    """Load a MEMORY.md file, returning a fresh :class:`MemoryDocument`
    if the file is missing or unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return MemoryDocument()
    except OSError:
        logger.warning("MEMORY.md unreadable at %s", path, exc_info=True)
        return MemoryDocument()
    return parse_memory(text)


def save_memory(doc: MemoryDocument, path: Path) -> None:
    """Atomically write *doc* to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_memory(doc), encoding="utf-8")
    tmp.replace(path)


def refresh_memory(
    doc: MemoryDocument,
    fact_store: FactStore,
    *,
    catalyst_horizon_days: int = 30,
) -> MemoryDocument:
    """Return a new :class:`MemoryDocument` with auto sections regenerated
    from *fact_store*. User content and the provided doc are not mutated."""
    open_theses = fact_store.get_open_theses()
    upcoming = fact_store.get_upcoming_catalysts(days_ahead=catalyst_horizon_days)

    # ``get_upcoming_catalysts`` only returns theses whose catalyst_date is
    # parseable *and* within the horizon. Overlap with open_theses is
    # possible but they render differently, so dedup is unnecessary.

    return MemoryDocument(
        auto_theses=[_render_theses_line(t) for t in open_theses],
        auto_catalysts=[_render_catalyst_line(t) for t in upcoming],
        user_content=doc.user_content,
        last_updated=datetime.now(timezone.utc),
    )


def refresh_memory_file(
    path: Path,
    fact_store: FactStore,
    *,
    catalyst_horizon_days: int = 30,
) -> MemoryDocument:
    """Convenience: load MEMORY.md, refresh auto sections, write back."""
    current = load_memory(path)
    refreshed = refresh_memory(current, fact_store, catalyst_horizon_days=catalyst_horizon_days)
    save_memory(refreshed, path)
    return refreshed


# ----------------------------------------------------------------------
# BOOTSTRAP.md
# ----------------------------------------------------------------------


def load_bootstrap(path: Path) -> str | None:
    """Load BOOTSTRAP.md contents as a system-prompt extension.

    Returns ``None`` when the file is missing, empty, or unreadable so
    callers can cheaply skip injection without branching on exceptions.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning("BOOTSTRAP.md unreadable at %s", path, exc_info=True)
        return None
    text = text.strip()
    return text or None
