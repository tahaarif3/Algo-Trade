from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from engine.events import MarketTick, SignalEvent, FillEvent, OrderAckEvent, L2Delta
from engine.protocols import HFTStrategyProtocol, HFTOrderBookProtocol

if TYPE_CHECKING:
    import pandas as pd
    from strategies.trend_oscillator import TrendOscillatorStrategy
    from engine.des_engine import DiscreteEventSimulator


class BarToHFTWrapper(HFTStrategyProtocol):
    """Adapts a bar-based strategy (e.g. TrendOscillatorStrategy) to operate on
    tick-level events in the nanosecond DiscreteEventSimulator (HFT engine).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy: TrendOscillatorStrategy,
        warmup_candles: int,
    ) -> None:
        self.df = df
        self.strategy = strategy
        self.warmup = warmup_candles

        # Position and order state
        self.position_side: Optional[str] = None
        self.entry_price: float = 0.0
        self.sl_price: float = 0.0
        self.tp_price: float = 0.0
        self.qty: float = 0.0

        # Track pending entry limit orders
        self.pending_entry_bar_idx: Optional[int] = None
        self.signal_bar_idx: Optional[int] = None

        # Time-syncing state: we start evaluating at index `warmup_candles`
        self.last_evaluated_bar_idx: int = warmup_candles - 1

        # Calculate bar duration dynamically
        if len(df) > 1:
            self.bar_duration_ns = df.index[1].value - df.index[0].value
        else:
            self.bar_duration_ns = 4 * 3600 * 1_000_000_000  # fallback to 4H

        self.des: Optional[DiscreteEventSimulator] = None

    def on_market_tick(
        self,
        tick: MarketTick,
        book: HFTOrderBookProtocol,
    ) -> list[SignalEvent]:
        # Initialize last_evaluated_bar_idx on first tick to avoid executing historic signals
        if self.last_evaluated_bar_idx == self.warmup - 1:
            for i in range(self.warmup, len(self.df)):
                bar_start_ns = self.df.index[i].value
                bar_close_ns = bar_start_ns + self.bar_duration_ns
                if bar_start_ns <= tick.timestamp_ns < bar_close_ns:
                    self.last_evaluated_bar_idx = i - 1
                    break
            else:
                if tick.timestamp_ns < self.df.index[self.warmup].value:
                    self.last_evaluated_bar_idx = self.warmup - 1
                else:
                    self.last_evaluated_bar_idx = len(self.df) - 1

        signals: list[SignalEvent] = []

        # 1. Intra-bar exits: Check SL/TP on every tick if in a position
        if self.position_side is not None:
            triggered = False
            if self.position_side == "long":
                if tick.price <= self.sl_price:
                    signals.append(
                        SignalEvent(
                            timestamp_ns=tick.timestamp_ns,
                            side="short",
                            price=0.0,
                            qty=self.qty,
                            order_type="market",
                        )
                    )
                    triggered = True
                elif tick.price >= self.tp_price:
                    signals.append(
                        SignalEvent(
                            timestamp_ns=tick.timestamp_ns,
                            side="short",
                            price=0.0,
                            qty=self.qty,
                            order_type="market",
                        )
                    )
                    triggered = True
            elif self.position_side == "short":
                if tick.price >= self.sl_price:
                    signals.append(
                        SignalEvent(
                            timestamp_ns=tick.timestamp_ns,
                            side="long",
                            price=0.0,
                            qty=self.qty,
                            order_type="market",
                        )
                    )
                    triggered = True
                elif tick.price <= self.tp_price:
                    signals.append(
                        SignalEvent(
                            timestamp_ns=tick.timestamp_ns,
                            side="long",
                            price=0.0,
                            qty=self.qty,
                            order_type="market",
                        )
                    )
                    triggered = True

            if triggered:
                from hft_engine import PositionStatus
                if hasattr(self, "state_broker") and self.state_broker is not None:
                    self.state_broker.update_status(PositionStatus.PendingClose)
                if hasattr(self, "trace_logger") and self.trace_logger is not None:
                    self.trace_logger.log_event(
                        tick.timestamp_ns,
                        2,
                        "StopTriggered",
                        f"Exit triggered: stop price hit at {tick.price}"
                    )
                return signals

        # 2. Time-syncing: Process bar transitions & check entries
        while True:
            next_eval_idx = self.last_evaluated_bar_idx + 1
            if next_eval_idx >= len(self.df):
                break

            bar_close_ns = self.df.index[next_eval_idx].value + self.bar_duration_ns
            if tick.timestamp_ns < bar_close_ns:
                break

            # 2a. Reconcile state at the boundary
            from hft_engine import PositionStatus
            if hasattr(self, "trace_logger") and self.trace_logger is not None:
                self.trace_logger.log_event(
                    tick.timestamp_ns,
                    2,
                    "BoundarySyncStart",
                    f"Boundary bar_idx: {next_eval_idx}"
                )

            if hasattr(self, "gatekeeper") and self.gatekeeper is not None:
                self.gatekeeper.reconcile_boundary_state(bar_close_ns, 50) # 50ms timeout
                
                # Pull execution reports to reconcile strategy state
                reports = self.gatekeeper.pull_reports()
                for r in reports:
                    if r.status == PositionStatus.Flat:
                        self.position_side = None
                        self.entry_price = 0.0
                        self.sl_price = 0.0
                        self.tp_price = 0.0
                        self.qty = 0.0
                        self.pending_entry_bar_idx = None
                        self.signal_bar_idx = None
                    else:
                        self.position_side = "long" if r.status == PositionStatus.Long else "short"
                        self.entry_price = r.fill_price
                        self.qty = r.fill_qty

            if hasattr(self, "trace_logger") and self.trace_logger is not None:
                current_status = self.state_broker.get_status() if hasattr(self, "state_broker") else "Unknown"
                self.trace_logger.log_event(
                    tick.timestamp_ns,
                    2,
                    "BoundarySyncEnd",
                    f"Final state: {current_status}"
                )

            signals.extend(
                self._evaluate_bar(next_eval_idx, tick.timestamp_ns, tick.price)
            )
            self.last_evaluated_bar_idx = next_eval_idx

        return signals

    def _evaluate_bar(
        self, bar_idx: int, timestamp_ns: int, current_price: float
    ) -> list[SignalEvent]:
        signals: list[SignalEvent] = []

        # A. Cancel pending entry limit order if not filled within the 1-bar window
        if self.pending_entry_bar_idx is not None:
            if bar_idx > self.pending_entry_bar_idx:
                if self.des and self.des._open_order_id is not None:
                    self.des._lob.cancel_order(self.des._open_order_id)
                    self.des._pending_orders.pop(self.des._open_order_id, None)
                    self.des._open_order_id = None
                self.pending_entry_bar_idx = None

        # B. Check entries if currently flat
        if self.position_side is None:
            row = self.df.iloc[bar_idx]

            if self.strategy.should_long(row):
                get_entry_price_fn = getattr(self.strategy, "get_entry_price", None)
                limit_price = (
                    get_entry_price_fn("long", row)
                    if get_entry_price_fn is not None
                    else None
                )

                capital = self.des._capital if self.des else 10000.0
                if limit_price is None:
                    # Market entry
                    qty = self.strategy.get_position_qty(capital, current_price)
                    signals.append(
                        SignalEvent(
                            timestamp_ns=timestamp_ns,
                            side="long",
                            price=0.0,
                            qty=qty,
                            order_type="market",
                        )
                    )
                    self.signal_bar_idx = bar_idx
                else:
                    # Limit entry
                    qty = self.strategy.get_position_qty(capital, limit_price)
                    signals.append(
                        SignalEvent(
                            timestamp_ns=timestamp_ns,
                            side="long",
                            price=limit_price,
                            qty=qty,
                            order_type="limit",
                        )
                    )
                    self.pending_entry_bar_idx = bar_idx
                    self.signal_bar_idx = bar_idx

            elif self.strategy.should_short(row):
                get_entry_price_fn = getattr(self.strategy, "get_entry_price", None)
                limit_price = (
                    get_entry_price_fn("short", row)
                    if get_entry_price_fn is not None
                    else None
                )

                capital = self.des._capital if self.des else 10000.0
                if limit_price is None:
                    # Market entry
                    qty = self.strategy.get_position_qty(capital, current_price)
                    signals.append(
                        SignalEvent(
                            timestamp_ns=timestamp_ns,
                            side="short",
                            price=0.0,
                            qty=qty,
                            order_type="market",
                        )
                    )
                    self.signal_bar_idx = bar_idx
                else:
                    # Limit entry
                    qty = self.strategy.get_position_qty(capital, limit_price)
                    signals.append(
                        SignalEvent(
                            timestamp_ns=timestamp_ns,
                            side="short",
                            price=limit_price,
                            qty=qty,
                            order_type="limit",
                        )
                    )
                    self.pending_entry_bar_idx = bar_idx
                    self.signal_bar_idx = bar_idx

        return signals

    def on_fill(
        self, fill: FillEvent, book: HFTOrderBookProtocol
    ) -> list[SignalEvent]:
        # Determine if fill is entry or exit
        if self.position_side is None:
            # Entry fill
            self.position_side = "long" if fill.side == "bid" else "short"
            self.entry_price = fill.fill_price
            self.qty = fill.fill_qty

            # Retrieve the row when the signal was generated
            signal_idx = (
                self.signal_bar_idx
                if self.signal_bar_idx is not None
                else self.last_evaluated_bar_idx
            )
            if signal_idx < 0:
                signal_idx = 0
            row = self.df.iloc[signal_idx]

            # Compute SL/TP prices
            self.sl_price = self.strategy.get_stop_loss(
                self.entry_price, self.position_side, row
            )
            self.tp_price = self.strategy.get_take_profit(
                self.entry_price, self.position_side, row
            )

            # Reset pending entry limit order tracker
            self.pending_entry_bar_idx = None
        else:
            # Exit fill
            self.position_side = None
            self.entry_price = 0.0
            self.sl_price = 0.0
            self.tp_price = 0.0
            self.qty = 0.0
            self.pending_entry_bar_idx = None
            self.signal_bar_idx = None

        return []

    def on_l2_delta(
        self, delta: L2Delta, book: HFTOrderBookProtocol
    ) -> list[SignalEvent]:
        return []

    def on_order_ack(self, ack: OrderAckEvent) -> None:
        pass
