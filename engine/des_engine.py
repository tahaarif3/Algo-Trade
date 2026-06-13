"""
Discrete Event Simulator (DES) — Python heapq implementation.

This is the orchestrator layer for the HFT backtest. It:
  1. Accepts an iterator of BaseEvents (from HFTDataLoader or a synthetic source).
  2. Loads them into a min-heap priority queue sorted by timestamp_ns.
  3. Pops events in chronological order and dispatches each to the correct handler.
  4. Routes strategy signals through the InProcessLatencyBridge, which re-inserts
     them as LatencyDelayEvents at timestamp_ns + latency_ns.
  5. Produces a BacktestResult compatible with the existing MetricsCalculator.

Slippage model:
  The latency bridge reschedules SignalEvents into the future. Any market events
  (MarketTick, L2Delta) processed between the signal time and the delayed order
  time update the order book state before the order is submitted. The strategy's
  desired price may no longer be the best price — this is slippage. No artificial
  slip constant is added; it emerges from real tick data.

Designed as a drop-in Python reference before the Rust DES loop (hft_engine.run_des)
is available. The handlers (_on_market_tick, etc.) will be mirrored 1:1 in Rust.
"""

from __future__ import annotations

import heapq
import math
import time
import uuid
from typing import Iterator, Optional
import numpy as np

from engine.config import HFTConfig, Trade, BacktestResult
from engine.events import (
    BaseEvent, MarketTick, L2Delta,
    L3OrderAdd, L3OrderCancel, L3OrderFill,
    SignalEvent, LatencyDelayEvent,
    OrderSubmitEvent, OrderAckEvent, FillEvent, SimEndEvent,
)
from engine.lob_python import LimitOrderBook
from engine.metrics import MetricsCalculator
from engine.latency_bridge import InProcessLatencyBridge


class DiscreteEventSimulator:
    """Nanosecond-resolution discrete event simulator.

    Usage:
        config = HFTConfig(latency_ns=50_000)
        des = DiscreteEventSimulator(config)
        result = des.run(event_stream, strategy)
    """

    def __init__(self, config: HFTConfig) -> None:
        self.config = config
        self._queue: list[BaseEvent] = []   # heapq min-heap
        if config.use_rust_lob:
            import hft_engine
            self._lob = hft_engine.OrderBook()
        else:
            self._lob = LimitOrderBook()
        
        self.rng = np.random.default_rng(config.mc_random_seed)
        self._latency_bridge = InProcessLatencyBridge(
            self,
            latency_ns=config.latency_ns,
            jitter_std=config.latency_jitter_std,
            rng=self.rng
        )

        # Position state
        self._capital = config.starting_capital
        self._position_side: Optional[str] = None   # 'long' | 'short' | None
        self._position_qty: float = 0.0
        self._position_entry_price: float = 0.0
        self._position_entry_fee: float = 0.0
        self._position_entry_time: int = 0
        self._open_order_id: Optional[int] = None
        self._pending_orders: dict[int, SignalEvent] = {}

        # Output
        self._trades: list[Trade] = []
        self._equity_curve: list[dict] = []
        self._event_count: int = 0
        self._order_id_counter: int = 0

        # Timing
        self._sim_start_ns: int = 0
        self._last_market_ts_ns: int = 0
        self._last_market_price: float = 0.0

        # Concurrency & State Sync Primitives (Rust HFT core bindings)
        from hft_engine import RealTimeStateBroker, BarBoundaryGatekeeper, TraceLogger
        self.state_broker = RealTimeStateBroker()
        self.gatekeeper = BarBoundaryGatekeeper(self.state_broker)
        self.trace_logger = TraceLogger()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def schedule(self, event: BaseEvent) -> None:
        """Push an event into the priority queue."""
        heapq.heappush(self._queue, event)

    def run(
        self,
        events: Iterator[BaseEvent],
        strategy: object,      # HFTStrategyProtocol — avoid circular import
    ) -> BacktestResult:
        """Execute the full simulation.

        Args:
            events:   Ascending-timestamp event iterator (Parquet data source).
            strategy: Object satisfying HFTStrategyProtocol.

        Returns:
            BacktestResult with trades, equity curve, and metrics.
        """
        self._strategy = strategy
        if hasattr(strategy, "des") and strategy.des is None:
            strategy.des = self
        strategy.state_broker = self.state_broker
        strategy.gatekeeper = self.gatekeeper
        strategy.trace_logger = self.trace_logger
        self._sim_start_ns = time.perf_counter_ns()

        # Seed the queue with all incoming events
        for event in events:
            # Stochastic packet drops for Level 2 data
            if isinstance(event, L2Delta) and self.config.packet_drop_rate > 0.0:
                if self.rng.random() < self.config.packet_drop_rate:
                    continue
            self.schedule(event)

        # Schedule a sentinel to terminate the loop cleanly
        if self._queue:
            last_ts = max(e.timestamp_ns for e in self._queue)
            self.schedule(SimEndEvent(timestamp_ns=last_ts + 1))

        # Main DES loop
        warmup_remaining = self.config.warmup_events
        while self._queue:
            event = heapq.heappop(self._queue)
            self._event_count += 1

            if self._event_count > self.config.max_events:
                print(
                    f"  [DES] Circuit breaker: {self._event_count} events processed, stopping."
                )
                break

            # Burn warmup events before recording stats
            if warmup_remaining > 0:
                warmup_remaining -= 1
                # Still dispatch market events during warmup to build book state
                if isinstance(event, (MarketTick, L2Delta, L3OrderAdd, L3OrderCancel)):
                    self._dispatch_market(event)
                continue

            self._dispatch(event)

        # Force-close any open position at sim end
        if self._position_side is not None:
            self._force_close(self._last_market_ts_ns)

        metrics = MetricsCalculator.compute_all(
            self._trades,
            self._equity_curve,
            self.config.starting_capital,
        )
        result = BacktestResult(
            trades=self._trades,
            equity_curve=self._equity_curve,
            config=self.config,       # type: ignore[arg-type]
            metrics=metrics,
        )
        print(result.summary)
        return result

    # ------------------------------------------------------------------ #
    #  Event dispatcher                                                   #
    # ------------------------------------------------------------------ #

    def _dispatch(self, event: BaseEvent) -> None:
        """Route event to the correct handler by type."""
        if isinstance(event, SimEndEvent):
            return

        if isinstance(event, MarketTick):
            self._on_market_tick(event)
        elif isinstance(event, L2Delta):
            self._on_l2_delta(event)
        elif isinstance(event, L3OrderAdd):
            self._on_l3_order_add(event)
        elif isinstance(event, L3OrderCancel):
            self._on_l3_order_cancel(event)
        elif isinstance(event, L3OrderFill):
            self._on_l3_order_fill(event)
        elif isinstance(event, LatencyDelayEvent):
            self._on_latency_delay(event)
        elif isinstance(event, FillEvent):
            self._on_fill(event)
        elif isinstance(event, OrderAckEvent):
            self._on_order_ack(event)
        else:
            pass  # Unknown event type — extensible

    def _dispatch_market(self, event: BaseEvent) -> None:
        """Dispatch only market data events (used during warmup)."""
        if isinstance(event, MarketTick):
            self._on_market_tick(event)
        elif isinstance(event, L2Delta):
            self._on_l2_delta(event)

    # ------------------------------------------------------------------ #
    #  Market data handlers                                               #
    # ------------------------------------------------------------------ #

    def _on_market_tick(self, tick: MarketTick) -> None:
        """Handle an aggressor trade tick.

        Updates the last-known price and calls the strategy for signals.
        The strategy receives the current LOB state and may return SignalEvents.
        Each signal is immediately routed through the latency bridge.
        """
        self.gatekeeper.update_processed_tick(tick.timestamp_ns)
        self._last_market_ts_ns = tick.timestamp_ns
        self._last_market_price = tick.price

        # 1. Match our resting orders against the market trade
        best_bid = self._lob.best_bid()
        if best_bid > 0.0 and tick.price <= best_bid:
            if self._open_order_id in self._pending_orders:
                signal = self._pending_orders[self._open_order_id]
                fill_qty = signal.qty
                fill_price = signal.price
                self._lob.cancel_order(self._open_order_id)
                fill_event = FillEvent(
                    timestamp_ns=tick.timestamp_ns,
                    order_id=self._open_order_id,
                    fill_price=fill_price,
                    fill_qty=fill_qty,
                    side="bid",
                    is_taker=False,
                )
                self._dispatch(fill_event)

        best_ask = self._lob.best_ask()
        if best_ask < math.inf and tick.price >= best_ask:
            if self._open_order_id in self._pending_orders:
                signal = self._pending_orders[self._open_order_id]
                fill_qty = signal.qty
                fill_price = signal.price
                self._lob.cancel_order(self._open_order_id)
                fill_event = FillEvent(
                    timestamp_ns=tick.timestamp_ns,
                    order_id=self._open_order_id,
                    fill_price=fill_price,
                    fill_qty=fill_qty,
                    side="ask",
                    is_taker=False,
                )
                self._dispatch(fill_event)

        # 2. Ask strategy for signals
        signals = self._strategy.on_market_tick(tick, self._lob)
        for signal in signals:
            self._latency_bridge.send_order(signal)

        # Record equity snapshot at each tick (may be downsampled for perf)
        self._record_equity(tick.timestamp_ns, tick.price)

    def _on_l2_delta(self, delta: L2Delta) -> None:
        """Handle an L2 order book update.

        Feeds the delta into the LOB and calls the strategy.
        Note: L2 snapshots give price+qty but not individual order IDs.
        We model each L2 level as a synthetic single order.
        """
        self._last_market_ts_ns = delta.timestamp_ns

        # We don't add L2 data to our own LOB here because L2 snapshots
        # represent the aggregate state of ALL market participants, not just
        # our orders. The LOB we maintain tracks our own resting orders only.
        # The strategy reads `delta` directly to infer market structure.

        signals = self._strategy.on_l2_delta(delta, self._lob)
        for signal in signals:
            self._latency_bridge.send_order(signal)

    def _on_l3_order_add(self, event: L3OrderAdd) -> None:
        """Handle an L3 new order event (market-by-order data)."""
        fills = self._lob.add_order(
            order_id=event.order_id,
            price=event.price,
            qty=event.qty,
            side=event.side,
            timestamp_ns=event.timestamp_ns,
        )
        for fill in fills:
            if fill.order_id == self._open_order_id:
                # Our resting order got filled!
                self._dispatch(fill)

    def _on_l3_order_cancel(self, event: L3OrderCancel) -> None:
        """Handle an L3 cancel event."""
        self._lob.cancel_order(event.order_id)

    def _on_l3_order_fill(self, event: L3OrderFill) -> None:
        """Handle an L3 partial/full fill from external market data."""
        pass  # External fills just update book state; no position impact

    # ------------------------------------------------------------------ #
    #  Latency bridge                                                     #
    # ------------------------------------------------------------------ #

    def _on_latency_delay(self, delay_event: LatencyDelayEvent) -> None:
        """Fire the inner event after the latency window has elapsed.

        At this point, all market events between the signal time and now
        have been processed. We submit the order to the LOB.
        """
        signal: SignalEvent = delay_event.inner_event  # type: ignore[assignment]
        if signal is None:
            return

        # Skip if already in a position on the SAME side (no pyramiding)
        if self._position_side is not None and signal.side == self._position_side:
            return

        # Cancel any previous resting order
        if self._open_order_id is not None:
            self._lob.cancel_order(self._open_order_id)
            self._pending_orders.pop(self._open_order_id, None)
            self._open_order_id = None

        self._order_id_counter += 1
        order_id = self._order_id_counter

        # Convert signal side to LOB side
        lob_side = "bid" if signal.side == "long" else "ask"
        qty = signal.qty

        # Determine fill price: use best available if market order
        is_market = (signal.order_type == "market" or signal.price == 0.0)
        is_closing = (self._position_side is not None)
        if is_market:
            fill_price = self._lob.best_ask() if lob_side == "bid" else self._lob.best_bid()
            if fill_price == 0.0 or fill_price == math.inf:
                # Fallback to last trade price if our own book doesn't have matching liquidity
                fill_price = getattr(self, "_last_market_price", 0.0)
                if fill_price == 0.0:
                    return  # No price information yet
                
                # Add to pending orders so _on_fill can retrieve the signal
                self._pending_orders[order_id] = signal
                
                fill = FillEvent(
                    timestamp_ns=delay_event.timestamp_ns,
                    order_id=order_id,
                    fill_price=fill_price,
                    fill_qty=qty,
                    side=lob_side,
                    is_taker=True,
                )
                self._dispatch(fill)
                return
        else:
            fill_price = signal.price

        # Submit to LOB — may get immediate fills
        fills = self._lob.add_order(
            order_id=order_id,
            price=fill_price,
            qty=qty,
            side=lob_side,
            timestamp_ns=delay_event.timestamp_ns,
        )

        is_closing = (self._position_side is not None)

        if fills:
            # Immediate fill (market order crossed the spread)
            for fill in fills:
                fill_with_correct_ts = FillEvent(
                    timestamp_ns=delay_event.timestamp_ns,
                    order_id=fill.order_id,
                    fill_price=fill.fill_price,
                    fill_qty=fill.fill_qty,
                    side=lob_side,
                    is_taker=True,
                )
                if is_closing:
                    self._close_position(fill_with_correct_ts)
                else:
                    self._open_position(signal, fill_with_correct_ts)
        else:
            # Resting limit order — track it
            self._open_order_id = order_id
            self._pending_orders[order_id] = signal

    # ------------------------------------------------------------------ #
    #  Fill and ack handlers                                              #
    # ------------------------------------------------------------------ #

    def _on_fill(self, fill: FillEvent) -> None:
        """Handle a fill for one of our orders."""
        if self._position_side is None:
            # Opening fill
            signal = self._pending_orders.pop(fill.order_id, None)
            if signal:
                self._open_position(signal, fill)
        else:
            # Closing fill
            self._close_position(fill)
            self._pending_orders.pop(fill.order_id, None)

        signals = self._strategy.on_fill(fill, self._lob)
        for signal in signals:
            self._latency_bridge.send_order(signal)

    def _on_order_ack(self, ack: OrderAckEvent) -> None:
        """Handle order acknowledgement."""
        self._strategy.on_order_ack(ack)

    # ------------------------------------------------------------------ #
    #  Position management                                                #
    # ------------------------------------------------------------------ #

    def _open_position(self, signal: SignalEvent, fill: FillEvent) -> None:
        """Record a new position opening."""
        if self._position_side is not None:
            return  # Already in position

        fee = fill.fill_qty * fill.fill_price * self.config.fee_rate_taker
        self._position_side = signal.side
        self._position_qty = fill.fill_qty
        self._position_entry_price = fill.fill_price
        self._position_entry_fee = fee
        self._position_entry_time = fill.timestamp_ns
        self._capital -= fee

        from hft_engine import PositionStatus, ExecutionReport
        status = PositionStatus.Long if signal.side == "long" else PositionStatus.Short
        self.state_broker.update_status(status)
        self.trace_logger.log_event(
            fill.timestamp_ns,
            1,
            "StateBrokerUpdate",
            f"Open {signal.side} position"
        )
        self.gatekeeper.push_report(ExecutionReport(
            fill.timestamp_ns,
            fill.order_id,
            fill.fill_price,
            fill.fill_qty,
            -fee,
            status
        ))

    def _close_position(self, fill: FillEvent) -> None:
        """Close an open position and record the trade."""
        if self._position_side is None:
            return

        exit_fee = fill.fill_qty * fill.fill_price * self.config.fee_rate_taker

        if self._position_side == "long":
            gross_pnl = (fill.fill_price - self._position_entry_price) * fill.fill_qty
        else:
            gross_pnl = (self._position_entry_price - fill.fill_price) * fill.fill_qty

        net_pnl = gross_pnl - self._position_entry_fee - exit_fee
        self._capital += gross_pnl - exit_fee

        trade_value = self._position_qty * self._position_entry_price
        pnl_pct = (net_pnl / trade_value * 100) if trade_value > 0 else 0.0

        entry_ts_str = str(self._position_entry_time)
        exit_ts_str = str(fill.timestamp_ns)

        trade = Trade(
            entry_time=entry_ts_str,
            exit_time=exit_ts_str,
            side=self._position_side,
            entry_price=self._position_entry_price,
            exit_price=fill.fill_price,
            qty=self._position_qty,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fee_paid=self._position_entry_fee + exit_fee,
            exit_reason="fill",
            duration_bars=0,   # Not applicable for tick-level trading
        )
        self._trades.append(trade)

        # Reset position state
        self._position_side = None
        self._position_qty = 0.0
        self._position_entry_price = 0.0
        self._position_entry_fee = 0.0
        self._open_order_id = None

        from hft_engine import PositionStatus, ExecutionReport
        self.state_broker.update_status(PositionStatus.Flat)
        self.trace_logger.log_event(
            fill.timestamp_ns,
            1,
            "StateBrokerUpdate",
            "Position closed (Flat)"
        )
        self.gatekeeper.push_report(ExecutionReport(
            fill.timestamp_ns,
            fill.order_id,
            fill.fill_price,
            fill.fill_qty,
            net_pnl,
            PositionStatus.Flat
        ))

    def _force_close(self, timestamp_ns: int) -> None:
        """Force-close open position at simulation end using mid-price."""
        mid = self._lob.mid_price()
        if mid == 0.0:
            mid = getattr(self, "_last_market_price", 0.0)
            if mid == 0.0:
                return

        synthetic_fill = FillEvent(
            timestamp_ns=timestamp_ns,
            order_id=-1,
            fill_price=mid,
            fill_qty=self._position_qty,
            side="ask" if self._position_side == "long" else "bid",
            is_taker=True,
        )
        self._close_position(synthetic_fill)

    # ------------------------------------------------------------------ #
    #  Equity curve                                                       #
    # ------------------------------------------------------------------ #

    def _record_equity(self, timestamp_ns: int, current_price: float) -> None:
        """Append an equity snapshot to the curve.

        To keep memory bounded, only records one snapshot per unique
        millisecond (1_000_000 ns). For 1 day of BTC tick data this
        produces ~86_400 data points at 1-second resolution.
        """
        # Downsample: record every 1_000_000 ns (1ms)
        ms_bucket = timestamp_ns // 1_000_000
        if self._equity_curve and self._equity_curve[-1]["timestamp"] == ms_bucket:
            return

        if self._position_side == "long":
            unrealized = (current_price - self._position_entry_price) * self._position_qty
        elif self._position_side == "short":
            unrealized = (self._position_entry_price - current_price) * self._position_qty
        else:
            unrealized = 0.0

        self._equity_curve.append({
            "timestamp": ms_bucket,
            "equity": self._capital + unrealized,
        })
