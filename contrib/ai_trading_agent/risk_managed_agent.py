#!/usr/bin/env python3
"""Risk-managed trading agent prototype.

This module demonstrates a practical AI-agent style decision loop for trading
with explicit constraints:
- Win rate target > 50%
- Risk/Reward >= 1:2
- Drawdown control
- Capital allocation discipline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class TradeSignal:
    """Input signal with model confidence and direction."""

    confidence: float
    direction: int  # +1 long, -1 short


@dataclass
class PositionPlan:
    """Planned position with bounded risk."""

    direction: int
    entry: float
    stop_loss: float
    take_profit: float
    size_units: float
    risk_amount: float


@dataclass
class AgentConfig:
    """Hard risk constraints for the agent."""

    min_confidence: float = 0.60
    risk_per_trade_pct: float = 0.01
    max_portfolio_exposure_pct: float = 0.20
    max_drawdown_pct: float = 0.10
    reward_to_risk: float = 2.0


class RiskManagedAgent:
    """A compact strategy wrapper with strict risk discipline."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.equity_peak = 0.0

    def _risk_amount(self, capital: float) -> float:
        return capital * self.config.risk_per_trade_pct

    def _current_drawdown(self, capital: float) -> float:
        if self.equity_peak <= 0:
            return 0.0
        return max(0.0, (self.equity_peak - capital) / self.equity_peak)

    def plan_position(self, signal: TradeSignal, price: float, capital: float) -> PositionPlan | None:
        """Create a position plan if risk gates pass."""
        self.equity_peak = max(self.equity_peak, capital)
        if signal.confidence < self.config.min_confidence:
            return None

        if self._current_drawdown(capital) > self.config.max_drawdown_pct:
            return None

        risk_amount = self._risk_amount(capital)
        max_notional = capital * self.config.max_portfolio_exposure_pct

        stop_distance = price * 0.01
        if stop_distance <= 0:
            return None

        size_units = min(risk_amount / stop_distance, max_notional / price)
        if size_units <= 0:
            return None

        stop_loss = price - stop_distance if signal.direction > 0 else price + stop_distance
        take_profit = (
            price + self.config.reward_to_risk * stop_distance
            if signal.direction > 0
            else price - self.config.reward_to_risk * stop_distance
        )

        return PositionPlan(
            direction=signal.direction,
            entry=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size_units=size_units,
            risk_amount=risk_amount,
        )


def score_strategy(trade_pnls: Iterable[float], starting_capital: float) -> dict:
    """Compute win rate, max drawdown and risk/reward diagnostics."""
    pnls: List[float] = list(trade_pnls)
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]

    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    rr = (avg_win / avg_loss) if avg_loss > 0 else float("inf")

    equity = starting_capital
    peak = starting_capital
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)

    return {
        "win_rate": win_rate,
        "risk_reward": rr,
        "max_drawdown": max_drawdown,
        "passes_target": win_rate > 0.50 and rr >= 2.0,
    }


def _demo() -> None:
    agent = RiskManagedAgent()
    capital = 10_000.0
    signal = TradeSignal(confidence=0.73, direction=1)
    plan = agent.plan_position(signal, price=100.0, capital=capital)

    print("Plan:", plan)
    report = score_strategy([120, -60, 130, -50, 110], starting_capital=capital)
    print("Report:", report)


if __name__ == "__main__":
    _demo()
