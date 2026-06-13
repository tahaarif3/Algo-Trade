"""
Protocol definitions for the backtesting framework.

These define the contracts between components. Any future engine replacement,
MCP-based data source, or alternative strategy implementation must satisfy
these protocols. Using runtime_checkable allows isinstance() validation
when wiring up components dynamically (e.g., from MCP tool invocations).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Iterator, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from engine.config import StrategyConfig, BacktestResult, HFTConfig
    from engine.events import (
        BaseEvent, MarketTick, L2Delta, SignalEvent,
        FillEvent, OrderAckEvent,
    )


@runtime_checkable
class DataSourceProtocol(Protocol):
    """Contract for candle data providers.

    Current implementation: BinanceDataLoader (public REST API).
    Future: MCP-based data source, database-backed loader, CSV import, etc.
    """

    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT').
            interval: Candle timeframe (e.g., '4h', '1h').
            start_date: ISO date string 'YYYY-MM-DD'.
            end_date: ISO date string 'YYYY-MM-DD'.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
            Index should be a DatetimeIndex or integer index.
        """
        ...


@runtime_checkable
class StrategyProtocol(Protocol):
    """Contract for trading strategies.

    Strategies are pure logic — they receive indicator-enriched data and
    return entry signals. They do NOT manage orders or positions (that's
    the engine's job).
    """

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to the candle DataFrame.

        Must not mutate the input DataFrame. Returns a new or augmented copy.
        """
        ...

    def should_long(self, row: pd.Series) -> bool:
        """Return True if a long entry signal is present at this bar."""
        ...

    def should_short(self, row: pd.Series) -> bool:
        """Return True if a short entry signal is present at this bar."""
        ...

    def get_stop_loss(self, entry_price: float, side: str, row: pd.Series) -> float:
        """Compute stop-loss price given entry and current bar data."""
        ...

    def get_take_profit(self, entry_price: float, side: str, row: pd.Series) -> float:
        """Compute take-profit price given entry and current bar data."""
        ...

    def get_position_qty(self, capital: float, entry_price: float) -> float:
        """Compute position size (quantity) given available capital."""
        ...

    def get_entry_price(self, side: str, row: pd.Series) -> float | None:
        """Return a limit entry price, or None to use market (close).

        When limit pricing is active, the engine fills at this price
        only if the next bar's low (long) or high (short) reaches it.
        Otherwise the order is cancelled (no fill).
        """
        ...


@runtime_checkable
class BacktestEngineProtocol(Protocol):
    """Contract for backtest engines.

    Current implementation: event-driven bar-by-bar engine.
    Future: vectorized engine, tick-level engine, or MCP-delegated engine.
    """

    def run(
        self,
        data: pd.DataFrame,
        strategy: StrategyProtocol,
        config: "StrategyConfig",
    ) -> "BacktestResult":
        """Execute a full backtest and return structured results.

        Args:
            data: Raw OHLCV DataFrame (indicators will be computed by strategy).
            strategy: Object satisfying StrategyProtocol.
            config: Backtest configuration (capital, fees, etc.).

        Returns:
            BacktestResult with trades, equity curve, and computed metrics.
        """
        ...


# ---------------------------------------------------------------------------
# HFT Protocols  (additive — do not modify the above)
# ---------------------------------------------------------------------------

@runtime_checkable
class HFTEventSourceProtocol(Protocol):
    """Contract for sources that stream nanosecond-timestamped events.

    Current implementation: HFTDataLoader reading local Parquet files.
    Future: live WebSocket feed, PCAP replayer, synthetic generator.
    """

    def stream_events(
        self,
        symbol: str,
        date: str,
    ) -> Iterator["BaseEvent"]:
        """Yield BaseEvent subclasses in ascending timestamp_ns order.

        Caller can mix MarketTick, L2Delta, etc. The DES dispatches
        by isinstance check, so type ordering in the stream doesn't matter
        as long as timestamps are monotonically non-decreasing.
        """
        ...


@runtime_checkable
class HFTOrderBookProtocol(Protocol):
    """Contract for the Limit Order Book matching engine.

    Current implementation: lob_python.LimitOrderBook (Python, O(log n)).
    Future: hft_engine.PyOrderBook (Rust PyO3, O(1) per cancel/add).

    Price representation: float (Python LOB) or fixed-point int64 (Rust LOB).
    The Rust binding converts at the PyO3 boundary.
    """

    def add_order(
        self,
        order_id: int,
        price: float,
        qty: float,
        side: str,          # 'bid' | 'ask'
    ) -> list["FillEvent"]:
        """Add a limit order. Returns list of immediate fills if it crosses.

        O(1) append to tail of price level's doubly-linked list (Rust).
        O(log n) level lookup + O(1) deque append (Python LOB).
        """
        ...

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a resting order by ID. Returns True if found and removed.

        O(1) hash-map lookup + O(1) doubly-linked-list splice (Rust).
        O(1) dict lookup + O(n) deque removal (Python LOB).
        """
        ...

    def best_bid(self) -> float:
        """Return the highest bid price, or 0.0 if no bids."""
        ...

    def best_ask(self) -> float:
        """Return the lowest ask price, or inf if no asks."""
        ...

    def mid_price(self) -> float:
        """Return (best_bid + best_ask) / 2."""
        ...

    def spread(self) -> float:
        """Return best_ask - best_bid in price units."""
        ...


@runtime_checkable
class HFTStrategyProtocol(Protocol):
    """Contract for HFT strategies operating on tick-level event streams.

    Unlike StrategyProtocol (which receives a bar row), HFT strategies
    receive atomic events and return optional SignalEvents. They are
    stateful: they track open orders and position state internally.
    """

    def on_market_tick(
        self,
        tick: "MarketTick",
        book: HFTOrderBookProtocol,
    ) -> list["SignalEvent"]:
        """Called on every MarketTick. Return zero or more signals."""
        ...

    def on_l2_delta(
        self,
        delta: "L2Delta",
        book: HFTOrderBookProtocol,
    ) -> list["SignalEvent"]:
        """Called on every L2Delta. Return zero or more signals."""
        ...

    def on_fill(
        self,
        fill: "FillEvent",
        book: HFTOrderBookProtocol,
    ) -> list["SignalEvent"]:
        """Called when one of our orders is filled. May generate follow-up signals."""
        ...

    def on_order_ack(
        self,
        ack: "OrderAckEvent",
    ) -> None:
        """Called when an order submission is acknowledged."""
        ...


@runtime_checkable
class HFTBacktestEngineProtocol(Protocol):
    """Contract for HFT backtest engines operating on event streams.

    Replaces BacktestEngineProtocol's DataFrame input with an iterator
    of nanosecond-timestamped BaseEvents. This is the DES entry point.

    Current implementation: des_engine.DiscreteEventSimulator (Python heapq).
    Future: hft_engine.run_des() (Rust BinaryHeap, PyO3 binding).
    """

    def run(
        self,
        events: Iterator["BaseEvent"],
        strategy: HFTStrategyProtocol,
        config: "HFTConfig",
    ) -> "BacktestResult":
        """Execute a full HFT backtest from an event stream.

        Args:
            events: Ascending-timestamp BaseEvent iterator (from HFTDataLoader).
            strategy: Object satisfying HFTStrategyProtocol.
            config: HFT backtest configuration.

        Returns:
            BacktestResult compatible with existing MetricsCalculator.
        """
        ...
