"""
Performance metrics calculator.

Computes all standard quantitative trading metrics from a trade log
and equity curve. Designed as a stateless utility — all methods are
pure functions that take data in and return results, making them
trivially wrappable as MCP tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from engine.config import Trade


class MetricsCalculator:
    """Stateless performance analytics engine.

    All methods are static or class methods — no internal state.
    This makes the calculator safe for concurrent use and simple
    to expose via MCP tools.
    """

    @staticmethod
    def compute_all(
        trades: list["Trade"],
        equity_curve: list[dict],
        starting_capital: float,
    ) -> dict:
        """Compute the full metrics suite from trade log and equity curve.

        Returns a flat dict of metrics (JSON-serializable).
        """
        metrics: dict = {}

        # --- Basic Trade Stats ---
        total_trades = len(trades)
        metrics["total_trades"] = total_trades

        if total_trades == 0:
            metrics.update({
                "net_profit": 0.0,
                "ending_balance": starting_capital,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "avg_trade_duration_bars": 0.0,
                "max_consecutive_losses": 0,
                "total_fees": 0.0,
                "avg_pnl_per_trade": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "long_trades": 0,
                "short_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
            })
            return metrics

        # Decompose trades
        pnls = [t.pnl for t in trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]
        fees = [t.fee_paid for t in trades]
        durations = [t.duration_bars for t in trades]
        sides = [t.side for t in trades]

        gross_profit = sum(winning)
        gross_loss = abs(sum(losing))

        metrics["net_profit"] = sum(pnls)
        metrics["ending_balance"] = starting_capital + sum(pnls)
        metrics["winning_trades"] = len(winning)
        metrics["losing_trades"] = len(losing)
        metrics["win_rate_pct"] = (len(winning) / total_trades) * 100
        metrics["profit_factor"] = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )
        metrics["gross_profit"] = gross_profit
        metrics["gross_loss"] = gross_loss
        metrics["total_fees"] = sum(fees)
        metrics["avg_pnl_per_trade"] = np.mean(pnls)
        metrics["avg_trade_duration_bars"] = np.mean(durations)
        metrics["long_trades"] = sum(1 for s in sides if s == "long")
        metrics["short_trades"] = sum(1 for s in sides if s == "short")

        # --- Max Consecutive Losses ---
        metrics["max_consecutive_losses"] = MetricsCalculator._max_consecutive_losses(
            pnls
        )

        # --- Drawdown from Equity Curve ---
        metrics["max_drawdown_pct"] = MetricsCalculator._max_drawdown(equity_curve)

        # --- Sharpe Ratio (annualized) ---
        metrics["sharpe_ratio"] = MetricsCalculator._sharpe_ratio(
            equity_curve, periods_per_year=365 * 6  # 4h candles → 6 per day
        )

        return metrics

    @staticmethod
    def _max_consecutive_losses(pnls: list[float]) -> int:
        """Longest streak of consecutive losing trades."""
        max_streak = 0
        current_streak = 0
        for pnl in pnls:
            if pnl <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    @staticmethod
    def _max_drawdown(equity_curve: list[dict]) -> float:
        """Maximum peak-to-trough drawdown as a negative percentage.

        Returns a negative number (e.g., -12.5 means 12.5% drawdown).
        """
        if not equity_curve:
            return 0.0

        equities = pd.Series([e["equity"] for e in equity_curve], dtype=float)
        running_max = equities.cummax()

        # Avoid division by zero
        drawdown_pct = ((equities - running_max) / running_max) * 100
        drawdown_pct = drawdown_pct.replace([np.inf, -np.inf], 0).fillna(0)

        return float(drawdown_pct.min())

    @staticmethod
    def _sharpe_ratio(
        equity_curve: list[dict], periods_per_year: float = 2190
    ) -> float:
        """Annualized Sharpe Ratio (risk-free rate assumed 0).

        Computed from per-period returns derived from the equity curve.
        Default periods_per_year assumes 4h candles (6 per day × 365).
        """
        if len(equity_curve) < 2:
            return 0.0

        equities = pd.Series([e["equity"] for e in equity_curve], dtype=float)
        returns = equities.pct_change().dropna()

        if returns.std() == 0 or len(returns) < 2:
            return 0.0

        sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        return float(sharpe)

    @staticmethod
    def format_metrics_table(metrics: dict) -> str:
        """Format metrics as a human-readable table for console output."""
        lines = [
            "+-----------------------------+---------------+",
            "|        Metric               |     Value     |",
            "+-----------------------------+---------------+",
        ]

        rows = [
            ("Ending Balance", f"${metrics.get('ending_balance', 0):>11,.2f}"),
            ("Net Profit", f"${metrics.get('net_profit', 0):>11,.2f}"),
            ("Total Trades", f"{metrics.get('total_trades', 0):>13}"),
            ("Win Rate", f"{metrics.get('win_rate_pct', 0):>12.1f}%"),
            ("Profit Factor", f"{metrics.get('profit_factor', 0):>13.2f}"),
            ("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):>12.2f}%"),
            ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):>13.3f}"),
            ("Avg Duration (bars)", f"{metrics.get('avg_trade_duration_bars', 0):>13.1f}"),
            ("Max Consec. Losses", f"{metrics.get('max_consecutive_losses', 0):>13}"),
            ("Total Fees", f"${metrics.get('total_fees', 0):>11,.2f}"),
        ]

        for label, value in rows:
            lines.append(f"| {label:<27} | {value} |")

        lines.append("+-----------------------------+---------------+")
        return "\n".join(lines)
