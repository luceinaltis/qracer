"""REST API routes for the qracer web dashboard.

The routes operate on the file-backed stores configured in
:func:`qracer.web.app.create_app`. They expose read access to alerts, tasks,
watchlist, and portfolio state, plus minimal write endpoints for managing
alerts and the watchlist from a remote client.

A FastAPI ``Request`` is used to access ``app.state`` so the same router can
be mounted on test apps with isolated state directories.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from qracer.alerts import Alert, AlertCondition, AlertStore
from qracer.config.loader import load_config
from qracer.risk.calculator import RiskCalculator
from qracer.risk.models import HoldingSnapshot
from qracer.tasks import Task, TaskStore
from qracer.watchlist import Watchlist

if TYPE_CHECKING:
    from fastapi import APIRouter, Request

try:
    from fastapi import APIRouter, HTTPException, Request

    router: "APIRouter" = APIRouter()
except ImportError:  # pragma: no cover - import guard for optional dep
    router = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AlertCreateRequest(BaseModel):
    """Body for ``POST /api/alerts``."""

    ticker: str = Field(..., min_length=1, description="Ticker symbol, e.g. AAPL")
    condition: AlertCondition
    threshold: float
    reference_price: float | None = None


class WatchlistAddRequest(BaseModel):
    """Body for ``POST /api/watchlist``."""

    ticker: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_alert(alert: Alert) -> dict[str, Any]:
    data = asdict(alert)
    data["condition"] = alert.condition.value
    return data


def _serialize_task(task: Task) -> dict[str, Any]:
    data = asdict(task)
    data["action_type"] = task.action_type.value
    data["schedule_type"] = task.schedule_type.value
    data["status"] = task.status.value
    return data


def _serialize_holding(h: HoldingSnapshot) -> dict[str, Any]:
    return {
        "ticker": h.ticker,
        "shares": h.shares,
        "avg_cost": h.avg_cost,
        "current_price": h.current_price,
        "market_value": h.market_value,
        "weight_pct": h.weight_pct,
        "unrealized_pnl": h.unrealized_pnl,
        "unrealized_pnl_pct": h.unrealized_pnl_pct,
    }


# ---------------------------------------------------------------------------
# Store accessors
# ---------------------------------------------------------------------------


def _alert_store(request: "Request") -> AlertStore:
    return request.app.state.alert_store  # type: ignore[no-any-return]


def _task_store(request: "Request") -> TaskStore:
    return request.app.state.task_store  # type: ignore[no-any-return]


def _watchlist(request: "Request") -> Watchlist:
    return request.app.state.watchlist  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


if router is not None:

    @router.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    # ---- Alerts ----------------------------------------------------------

    @router.get("/alerts")
    async def list_alerts(request: Request) -> dict[str, Any]:
        store = _alert_store(request)
        return {"alerts": [_serialize_alert(a) for a in store.alerts]}

    @router.post("/alerts", status_code=201)
    async def create_alert(request: Request, body: AlertCreateRequest) -> dict[str, Any]:
        store = _alert_store(request)
        alert = store.create(
            body.ticker,
            body.condition,
            body.threshold,
            reference_price=body.reference_price,
        )
        return {"alert": _serialize_alert(alert)}

    @router.delete("/alerts/{alert_id}")
    async def delete_alert(request: Request, alert_id: str) -> dict[str, Any]:
        store = _alert_store(request)
        # Touch the public accessor first so the mtime-based hot-reload picks
        # up writes from other processes (qracer repl, qracer serve).
        _ = store.alerts
        if not store.remove(alert_id):
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        return {"removed": alert_id}

    # ---- Tasks -----------------------------------------------------------

    @router.get("/tasks")
    async def list_tasks(request: Request) -> dict[str, Any]:
        store = _task_store(request)
        return {"tasks": [_serialize_task(t) for t in store.get_all()]}

    @router.delete("/tasks/{task_id}")
    async def delete_task(request: Request, task_id: str) -> dict[str, Any]:
        store = _task_store(request)
        # Trigger mtime-based hot-reload so writes from other processes
        # (qracer repl, qracer serve) are visible to the lookup below.
        _ = store.get_all()
        if not store.remove(task_id):
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return {"removed": task_id}

    # ---- Watchlist -------------------------------------------------------

    @router.get("/watchlist")
    async def get_watchlist(request: Request) -> dict[str, Any]:
        wl = _watchlist(request)
        return {"tickers": wl.tickers}

    @router.post("/watchlist", status_code=201)
    async def add_to_watchlist(request: Request, body: WatchlistAddRequest) -> dict[str, Any]:
        wl = _watchlist(request)
        added = wl.add(body.ticker)
        return {"ticker": body.ticker.upper(), "added": added, "tickers": wl.tickers}

    @router.delete("/watchlist/{ticker}")
    async def remove_from_watchlist(request: Request, ticker: str) -> dict[str, Any]:
        wl = _watchlist(request)
        if not wl.remove(ticker):
            raise HTTPException(status_code=404, detail=f"{ticker} not in watchlist")
        return {"removed": ticker.upper(), "tickers": wl.tickers}

    # ---- Portfolio -------------------------------------------------------

    @router.get("/portfolio")
    async def get_portfolio() -> dict[str, Any]:
        """Return a portfolio snapshot using the most recent avg_cost as the price.

        The web API runs as a separate process and may not have a configured
        data registry; rather than failing the route we fall back to avg_cost,
        which is a useful baseline view of holdings, weights, and total value.
        Clients with a price feed can override per-holding prices client-side.
        """
        config = load_config()
        prices = {h.ticker: h.avg_cost for h in config.portfolio.holdings}
        calculator = RiskCalculator(config.portfolio)
        snapshot = calculator.build_snapshot(prices)
        return {
            "currency": snapshot.currency,
            "total_value": snapshot.total_value,
            "as_of": snapshot.as_of.isoformat(),
            "holdings": [_serialize_holding(h) for h in snapshot.holdings],
        }


__all__ = ["router", "AlertCreateRequest", "WatchlistAddRequest"]
