"""PID file management — prevents duplicate qracer serve instances."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def acquire(pid_path: Path) -> bool:
    """Create a PID file. Returns False if another instance is running."""
    if is_running(pid_path):
        return False
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release(pid_path: Path) -> None:
    """Remove the PID file."""
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Failed to remove PID file %s", pid_path)


def is_running(pid_path: Path) -> bool:
    """Check if a process with the recorded PID is still alive."""
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    try:
        os.kill(pid, 0)  # Signal 0 = check existence, don't actually kill
        return True
    except OSError:
        # Process doesn't exist — stale PID file
        release(pid_path)
        return False
