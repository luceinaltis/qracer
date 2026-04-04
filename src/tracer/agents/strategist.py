"""Strategist agent — contrarian detection and conviction scoring.

Pipeline steps: Contrarian Detection (step 5), Conviction Scoring (step 6).
"""

from __future__ import annotations

from tracer.agents.base import BaseAgent
from tracer.llm.providers import Role
from tracer.models import Signal, SignalDirection


class Strategist(BaseAgent):
    """Investment decision-making and signal generation.

    Responsibilities:
    - Contrarian detection: find where consensus is wrong or late.
    - Conviction scoring: rank signals by strength, horizon, and risk.
    """

    role = Role.STRATEGIST

    async def detect_contrarian(
        self,
        cross_market: dict,
        consensus: dict,
    ) -> list[dict]:
        """Compare cross-market findings against consensus to find contrarian signals.

        Args:
            cross_market: Output from Analyst.discover_cross_market.
            consensus: Output from Researcher.map_consensus.

        Returns:
            List of contrarian signal dicts with ticker, thesis,
            contrarian_angle, and direction.
        """
        import json

        response = await self._complete(
            system=(
                "You are a contrarian investment strategist. Compare cross-market "
                "analysis against consensus views. Identify where consensus is wrong, "
                "late, or ignoring signals. Look for: oversold with improving "
                "fundamentals, overhyped with deteriorating data, ignored catalysts.\n\n"
                "Return a JSON array of objects with keys: ticker, thesis, "
                "contrarian_angle, direction (long/short/neutral), evidence (list)."
            ),
            user=(
                f"Cross-market findings:\n{json.dumps(cross_market, indent=2)}\n\n"
                f"Consensus view:\n{json.dumps(consensus, indent=2)}"
            ),
        )

        try:
            result = json.loads(response.content)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, ValueError):
            return []

    async def score_signals(self, contrarian_signals: list[dict]) -> list[Signal]:
        """Score contrarian signals by conviction, time horizon, and risk.

        Args:
            contrarian_signals: Output from detect_contrarian.

        Returns:
            Ranked list of Signal objects with conviction scores.
        """
        if not contrarian_signals:
            return []

        import json

        response = await self._complete(
            system=(
                "You are a conviction scorer. For each contrarian signal, assign:\n"
                "- conviction: 0.0-10.0 (data quality, signal convergence, "
                "historical patterns)\n"
                "- time_horizon_days: estimated days for thesis to play out\n"
                "- risk_factors: list of what could go wrong\n\n"
                "Return a JSON array of objects with keys: ticker, direction, "
                "conviction, thesis, evidence, contrarian_angle, risk_factors, "
                "time_horizon_days."
            ),
            user=f"Score these signals:\n{json.dumps(contrarian_signals, indent=2)}",
        )

        try:
            scored = json.loads(response.content)
            if not isinstance(scored, list):
                return []
        except (json.JSONDecodeError, ValueError):
            return []

        signals: list[Signal] = []
        for s in scored:
            try:
                direction_str = s.get("direction", "neutral").lower()
                direction_map = {
                    "long": SignalDirection.LONG,
                    "short": SignalDirection.SHORT,
                    "neutral": SignalDirection.NEUTRAL,
                }
                signals.append(
                    Signal(
                        ticker=s["ticker"],
                        direction=direction_map.get(direction_str, SignalDirection.NEUTRAL),
                        conviction=float(s.get("conviction", 5.0)),
                        thesis=s.get("thesis", ""),
                        evidence=s.get("evidence", []),
                        contrarian_angle=s.get("contrarian_angle"),
                        risk_factors=s.get("risk_factors", []),
                        time_horizon_days=s.get("time_horizon_days"),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue

        # Sort by conviction descending
        signals.sort(key=lambda sig: sig.conviction, reverse=True)
        return signals

    async def run(
        self, cross_market: dict, consensus: dict, **kwargs
    ) -> list[Signal]:
        """Default entry point — detect contrarian signals and score them."""
        contrarian = await self.detect_contrarian(cross_market, consensus)
        return await self.score_signals(contrarian)
