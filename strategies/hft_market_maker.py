from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from engine.events import MarketTick, L2Delta, FillEvent, OrderAckEvent
    from engine.protocols import HFTOrderBookProtocol

from engine.events import SignalEvent


class HFTMarketMaker:
    """HFT Market Maker Strategy.

    Satisfies HFTStrategyProtocol.
    On each MarketTick:
      - If flat: alternates posting a resting bid 1 tick below best bid,
        or a resting ask 1 tick above best ask.
      - If in a position: monitors price for profit target or stop loss
        and triggers a market exit signal.
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        profit_target: float = 100.0,
        stop_loss: float = 50.0,
        qty: float = 0.1,
    ) -> None:
        self.tick_size = tick_size
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.qty = qty

        # Position tracking
        self.position_side: str | None = None
        self.entry_price: float = 0.0

        # Alternating state for quoting when flat
        self.quote_side = "bid"

    def on_market_tick(
        self,
        tick: MarketTick,
        book: HFTOrderBookProtocol,
    ) -> list[SignalEvent]:
        """Called on every MarketTick."""
        # 1. Position management: check for TP/SL exit
        if self.position_side == "long":
            if tick.price >= self.entry_price + self.profit_target:
                # Take profit
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="short",
                    price=0.0,
                    qty=self.qty,
                    order_type="market",
                )]
            elif tick.price <= self.entry_price - self.stop_loss:
                # Stop loss
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="short",
                    price=0.0,
                    qty=self.qty,
                    order_type="market",
                )]
        elif self.position_side == "short":
            if tick.price <= self.entry_price - self.profit_target:
                # Take profit
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="long",
                    price=0.0,
                    qty=self.qty,
                    order_type="market",
                )]
            elif tick.price >= self.entry_price + self.stop_loss:
                # Stop loss
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="long",
                    price=0.0,
                    qty=self.qty,
                    order_type="market",
                )]

        # 2. Quoting when flat: post resting limit order
        else:
            # Determine bid/ask prices based on current market price
            # Post 1 tick below/above market price as proxy for best bid/ask
            if self.quote_side == "bid":
                bid_price = tick.price - self.tick_size
                self.quote_side = "ask"  # alternate for next time
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="long",
                    price=bid_price,
                    qty=self.qty,
                    order_type="limit",
                )]
            else:
                ask_price = tick.price + self.tick_size
                self.quote_side = "bid"
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="short",
                    price=ask_price,
                    qty=self.qty,
                    order_type="limit",
                )]

        return []

    def on_l2_delta(
        self,
        delta: L2Delta,
        book: HFTOrderBookProtocol,
    ) -> list[SignalEvent]:
        """Called on every L2Delta."""
        return []

    def on_fill(
        self,
        fill: FillEvent,
        book: HFTOrderBookProtocol,
    ) -> list[SignalEvent]:
        """Called when one of our orders is filled."""
        if self.position_side is None:
            # Entry fill
            self.position_side = "long" if fill.side == "bid" else "short"
            self.entry_price = fill.fill_price
        else:
            # Exit fill
            self.position_side = None
            self.entry_price = 0.0
        return []

    def on_order_ack(self, ack: OrderAckEvent) -> None:
        """Called when an order submission is acknowledged."""
        pass
