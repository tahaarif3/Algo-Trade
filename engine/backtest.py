"""
Event-driven backtesting engine.

Processes candles one-by-one, simulating order execution, SL/TP fills,
fee deductions, and equity tracking. Designed to be replaced in the
future — satisfies BacktestEngineProtocol.

Key design choices (documented for future engine rebuild):
    - Fills at candle close for entries (conservative)
    - SL/TP checked against intra-bar high/low (realistic)
    - SL-priority on ambiguous bars (adversarial assumption)
    - No partial fills or slippage model (simplification)
    - Single position at a time (no pyramiding)
    - Fees deducted on both entry and exit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import pandas as pd

from engine.config import StrategyConfig, Trade, BacktestResult
from engine.metrics import MetricsCalculator

if TYPE_CHECKING:
    from engine.protocols import StrategyProtocol


@dataclass
class _OpenPosition:
    """Internal state for a currently open position."""

    entry_time: str
    entry_bar_idx: int
    side: str            # 'long' or 'short'
    entry_price: float
    qty: float
    sl_price: float
    tp_price: float
    entry_fee: float


class BacktestEngine:
    """Bar-by-bar event-driven backtest engine.

    Satisfies BacktestEngineProtocol. Stateless between runs —
    all state is local to the run() method.

    Usage:
        engine = BacktestEngine()
        result = engine.run(data, strategy, config)
    """

    def run(
        self,
        data: pd.DataFrame,
        strategy: "StrategyProtocol",
        config: StrategyConfig,
    ) -> BacktestResult:
        """Execute a full backtest.

        Args:
            data: Raw OHLCV DataFrame (will be enriched with indicators).
            strategy: Object satisfying StrategyProtocol.
            config: Full backtest configuration.

        Returns:
            BacktestResult with trades, equity curve, and metrics.
        """
        # Step 1: Compute indicators on the full dataset
        df = strategy.compute_indicators(data)

        # Step 2: Determine backtest window (skip warmup)
        warmup = config.warmup_candles
        if warmup >= len(df):
            raise ValueError(
                f"Warmup period ({warmup}) >= available candles ({len(df)}). "
                "Need more historical data."
            )

        # Step 3: Initialize state
        capital = config.starting_capital
        position: Optional[_OpenPosition] = None
        pending_order: Optional[tuple[str, float, int, pd.Series]] = None
        trades: list[Trade] = []
        equity_curve: list[dict] = []

        # Step 4: Main simulation loop
        for bar_idx in range(warmup, len(df)):
            row = df.iloc[bar_idx]
            timestamp = str(df.index[bar_idx])

            # --- A0: Check pending limit order fills ---
            if pending_order is not None:
                side, limit_price, order_bar, order_row = pending_order
                if side == "long" and row["low"] <= limit_price:
                    position = self._open_position(
                        "long", row, bar_idx, timestamp, capital, strategy, config, entry_price=limit_price
                    )
                    capital -= position.entry_fee
                    pending_order = None
                elif side == "short" and row["high"] >= limit_price:
                    position = self._open_position(
                        "short", row, bar_idx, timestamp, capital, strategy, config, entry_price=limit_price
                    )
                    capital -= position.entry_fee
                    pending_order = None
                else:
                    # Cancel order if not filled on the next bar (1-bar fill window)
                    pending_order = None

            # --- A: Check exits on current bar if in position ---
            if position is not None:
                exit_result = self._check_exit(position, row, bar_idx, config)

                if exit_result is not None:
                    exit_price, exit_reason = exit_result
                    trade, realized_pnl = self._close_position(
                        position, exit_price, exit_reason, timestamp,
                        bar_idx, config
                    )
                    capital += realized_pnl
                    trades.append(trade)
                    position = None

            # --- B: Check entries if flat ---
            if position is None:
                get_entry_price_fn = getattr(strategy, "get_entry_price", None)

                if strategy.should_long(row):
                    limit_price = get_entry_price_fn("long", row) if get_entry_price_fn is not None else None
                    if limit_price is None:
                        position = self._open_position(
                            "long", row, bar_idx, timestamp, capital, strategy, config
                        )
                        capital -= position.entry_fee  # Deduct entry fee
                    else:
                        pending_order = ("long", limit_price, bar_idx, row)

                elif strategy.should_short(row):
                    limit_price = get_entry_price_fn("short", row) if get_entry_price_fn is not None else None
                    if limit_price is None:
                        position = self._open_position(
                            "short", row, bar_idx, timestamp, capital, strategy, config
                        )
                        capital -= position.entry_fee
                    else:
                        pending_order = ("short", limit_price, bar_idx, row)

            # --- C: Record equity ---
            unrealized = self._unrealized_pnl(position, row["close"]) if position else 0
            equity_curve.append({
                "timestamp": timestamp,
                "equity": capital + unrealized,
            })

        # Step 5: Force-close any open position at last bar
        if position is not None:
            last_row = df.iloc[-1]
            last_ts = str(df.index[-1])
            trade, realized_pnl = self._close_position(
                position, last_row["close"], "end_of_data",
                last_ts, len(df) - 1, config
            )
            capital += realized_pnl
            trades.append(trade)

        # Step 6: Compute metrics
        metrics = MetricsCalculator.compute_all(
            trades, equity_curve, config.starting_capital
        )

        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            config=config,
            metrics=metrics,
        )
        print(result.summary)
        return result

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _check_exit(
        self,
        pos: _OpenPosition,
        row: pd.Series,
        bar_idx: int,
        config: StrategyConfig,
    ) -> Optional[tuple[float, str]]:
        """Check if SL or TP is hit on the current bar.

        Uses intra-bar high/low. SL-priority on ambiguous bars
        (conservative assumption — if both could trigger, assume SL hit first).

        Returns (exit_price, reason) or None.
        """
        high = row["high"]
        low = row["low"]

        if pos.side == "long":
            sl_hit = low <= pos.sl_price
            tp_hit = high >= pos.tp_price

            if sl_hit and tp_hit:
                # Ambiguous bar: assume SL hit first (adversarial)
                return (pos.sl_price, "sl")
            elif sl_hit:
                return (pos.sl_price, "sl")
            elif tp_hit:
                return (pos.tp_price, "tp")

        else:  # short
            sl_hit = high >= pos.sl_price
            tp_hit = low <= pos.tp_price

            if sl_hit and tp_hit:
                return (pos.sl_price, "sl")
            elif sl_hit:
                return (pos.sl_price, "sl")
            elif tp_hit:
                return (pos.tp_price, "tp")

        return None

    def _open_position(
        self,
        side: str,
        row: pd.Series,
        bar_idx: int,
        timestamp: str,
        capital: float,
        strategy: "StrategyProtocol",
        config: StrategyConfig,
        entry_price: Optional[float] = None,
    ) -> _OpenPosition:
        """Create a new position. If entry_price is not provided, uses row['close']."""
        if entry_price is None:
            entry_price = row["close"]
        qty = strategy.get_position_qty(capital, entry_price)
        sl = strategy.get_stop_loss(entry_price, side, row)
        tp = strategy.get_take_profit(entry_price, side, row)
        entry_fee = qty * entry_price * config.fee_rate

        return _OpenPosition(
            entry_time=timestamp,
            entry_bar_idx=bar_idx,
            side=side,
            entry_price=entry_price,
            qty=qty,
            sl_price=sl,
            tp_price=tp,
            entry_fee=entry_fee,
        )

    def _close_position(
        self,
        pos: _OpenPosition,
        exit_price: float,
        exit_reason: str,
        exit_time: str,
        bar_idx: int,
        config: StrategyConfig,
    ) -> tuple[Trade, float]:
        """Close a position and compute realized P&L.

        Returns (Trade record, net realized P&L including fees).
        """
        exit_fee = pos.qty * exit_price * config.fee_rate

        if pos.side == "long":
            gross_pnl = (exit_price - pos.entry_price) * pos.qty
        else:  # short
            gross_pnl = (pos.entry_price - exit_price) * pos.qty

        # Net P&L after both entry and exit fees
        total_fees = pos.entry_fee + exit_fee
        net_pnl = gross_pnl - total_fees

        trade_value = pos.qty * pos.entry_price
        pnl_pct = (net_pnl / trade_value) * 100 if trade_value > 0 else 0

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=exit_time,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.qty,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fee_paid=total_fees,
            exit_reason=exit_reason,
            duration_bars=bar_idx - pos.entry_bar_idx,
        )

        # The realized P&L returned to the capital pool includes restoring
        # the trade's notional value plus the net P&L minus exit fee
        # Capital was NOT reduced by the notional on entry (we only deducted
        # the entry fee). So the realized amount to add back is just net_pnl.
        # Wait — let's reconsider the capital model:
        #
        # Capital model:
        #   On entry:  capital -= entry_fee (fee only, not notional)
        #   On exit:   capital += gross_pnl - exit_fee
        #              = capital += net_pnl + entry_fee - exit_fee... no.
        #
        # Simpler: capital tracks cash. On entry, we deduct entry_fee.
        # On exit, we get gross_pnl minus exit_fee.
        # Total P&L from this trade = gross_pnl - entry_fee - exit_fee = net_pnl
        # But we already deducted entry_fee from capital on entry.
        # So on exit, we add back: gross_pnl - exit_fee
        realized_return = gross_pnl - exit_fee

        return trade, realized_return

    def _unrealized_pnl(self, pos: _OpenPosition, current_price: float) -> float:
        """Mark-to-market unrealized P&L for equity curve tracking."""
        if pos.side == "long":
            return (current_price - pos.entry_price) * pos.qty
        else:
            return (pos.entry_price - current_price) * pos.qty
