"""qracer.web — FastAPI dashboard API for remote access.

This package exposes the same alerts/tasks/watchlist/portfolio state used by the
REPL and the ``qracer serve`` background service over a small REST API. It is
designed to run as a separate process that shares ``~/.qracer/`` files with the
other qracer processes; the file-backed stores hot-reload on mtime change so
mutations made elsewhere become visible automatically.
"""

from qracer.web.app import create_app

__all__ = ["create_app"]
