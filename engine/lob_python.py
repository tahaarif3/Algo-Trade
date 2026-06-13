"""
Pure-Python Limit Order Book — correctness reference implementation.

This is the ground-truth reference against which the Rust LOB (hft_engine.PyOrderBook)
will be validated. It uses SortedContainers for O(log n) price-level access and
Python deques for O(1) head/tail append.

Performance profile vs the Rust LOB:
    add_order:     O(log n) price level lookup + O(1) deque append
    cancel_order:  O(1) dict lookup + O(n) deque scan (Rust: O(1) linked-list splice)
    best_bid/ask:  O(log n) SortedDict min/max key (Rust: O(log n) BTreeMap)

The Python cancel_order is the only operation that degrades at scale. For the
validation workload (< 1M events), this is acceptable. The Rust LOB fixes this
with a hash-map of order_id -> arena slot that enables O(1) splice.

Usage:
    lob = LimitOrderBook()
    fills = lob.add_order(order_id=1, price=50000.0, qty=0.1, side='ask')
    fills = lob.add_order(order_id=2, price=50000.0, qty=0.05, side='bid')
    # -> [FillEvent(...)] because the bid crossed the ask
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from sortedcontainers import SortedDict

from engine.events import FillEvent


# ---------------------------------------------------------------------------
# Internal order node
# ---------------------------------------------------------------------------

@dataclass
class _Order:
    """A resting limit order in the book."""
    order_id: int
    price: float
    qty: float
    remaining_qty: float
    side: str           # 'bid' | 'ask'


# ---------------------------------------------------------------------------
# Price level
# ---------------------------------------------------------------------------

class _PriceLevel:
    """A FIFO queue of orders at a single price point."""

    __slots__ = ("price", "side", "_orders", "total_qty")

    def __init__(self, price: float, side: str) -> None:
        self.price = price
        self.side = side
        self._orders: deque[_Order] = deque()
        self.total_qty: float = 0.0

    def append(self, order: _Order) -> None:
        self._orders.append(order)
        self.total_qty += order.remaining_qty

    def peek_head(self) -> Optional[_Order]:
        return self._orders[0] if self._orders else None

    def pop_head(self) -> _Order:
        order = self._orders.popleft()
        self.total_qty -= order.remaining_qty
        return order

    def remove(self, order_id: int) -> Optional[_Order]:
        """O(n) scan — acceptable for Python reference, fixed in Rust."""
        for i, o in enumerate(self._orders):
            if o.order_id == order_id:
                del self._orders[i]
                self.total_qty -= o.remaining_qty
                return o
        return None

    def is_empty(self) -> bool:
        return len(self._orders) == 0

    def __len__(self) -> int:
        return len(self._orders)


# ---------------------------------------------------------------------------
# Limit Order Book
# ---------------------------------------------------------------------------

class LimitOrderBook:
    """Price-time priority limit order book.

    Maintains separate SortedDicts for bids (descending) and asks (ascending).
    Each price maps to a _PriceLevel containing a FIFO deque of orders.

    The order_map provides O(1) order_id lookup for cancels.

    Thread safety: single-threaded. The DES ensures sequential event dispatch.
    """

    def __init__(self) -> None:
        # SortedDict key = price (float). Bids: highest first (negate key).
        # We use _bids with negated keys so SortedDict[0] = best bid.
        self._bids: SortedDict = SortedDict()   # {-price: _PriceLevel}
        self._asks: SortedDict = SortedDict()   # {+price: _PriceLevel}

        # O(1) lookup: order_id -> _Order (for cancel)
        self._order_map: dict[int, _Order] = {}

        self._next_fill_ts: int = 0   # Monotonic fill timestamp placeholder

    # ------------------------------------------------------------------ #
    #  Public interface (satisfies HFTOrderBookProtocol)                  #
    # ------------------------------------------------------------------ #

    def add_order(
        self,
        order_id: int,
        price: float,
        qty: float,
        side: str,
        timestamp_ns: int = 0,
    ) -> list[FillEvent]:
        """Add a limit order. Returns immediate fills if it crosses spread.

        Args:
            order_id: Unique order identifier.
            price:    Limit price.
            qty:      Order quantity in base asset.
            side:     'bid' (buy) or 'ask' (sell).
            timestamp_ns: Event timestamp for FillEvent records.

        Returns:
            List of FillEvent objects for any immediate matches.
        """
        if order_id in self._order_map:
            raise ValueError(f"Duplicate order_id {order_id}")

        price = round(price, 8)
        qty = round(qty, 8)

        order = _Order(
            order_id=order_id,
            price=price,
            qty=qty,
            remaining_qty=qty,
            side=side,
        )
        self._order_map[order_id] = order

        fills: list[FillEvent] = []

        if side == "bid":
            fills = self._match_bid(order, timestamp_ns)
            if order.remaining_qty > 0:
                # Not fully filled — rest on book
                key = -order.price
                if key not in self._bids:
                    self._bids[key] = _PriceLevel(order.price, "bid")
                self._bids[key].append(order)
        else:  # ask
            fills = self._match_ask(order, timestamp_ns)
            if order.remaining_qty > 0:
                key = order.price
                if key not in self._asks:
                    self._asks[key] = _PriceLevel(order.price, "ask")
                self._asks[key].append(order)

        if order.remaining_qty <= 0:
            self._order_map.pop(order_id, None)

        return fills

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a resting order. Returns True if found and removed."""
        order = self._order_map.pop(order_id, None)
        if order is None:
            return False

        if order.side == "bid":
            key = -order.price
            level = self._bids.get(key)
            if level:
                level.remove(order_id)
                if level.is_empty():
                    del self._bids[key]
        else:
            key = order.price
            level = self._asks.get(key)
            if level:
                level.remove(order_id)
                if level.is_empty():
                    del self._asks[key]

        return True

    def best_bid(self) -> float:
        """Highest resting bid price, or 0.0 if no bids."""
        if not self._bids:
            return 0.0
        neg_price = self._bids.keys()[0]
        return -neg_price

    def best_ask(self) -> float:
        """Lowest resting ask price, or math.inf if no asks."""
        if not self._asks:
            return math.inf
        return self._asks.keys()[0]

    def mid_price(self) -> float:
        """(best_bid + best_ask) / 2."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid == 0.0 or ask == math.inf:
            return 0.0
        return (bid + ask) / 2.0

    def spread(self) -> float:
        """best_ask - best_bid in price units."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid == 0.0 or ask == math.inf:
            return math.inf
        return ask - bid

    def depth(self, side: str, levels: int = 5) -> list[tuple[float, float]]:
        """Return top N price levels as [(price, total_qty), ...].

        Bids: descending price order. Asks: ascending price order.
        """
        if side == "bid":
            return [
                (level.price, level.total_qty)
                for _, level in list(self._bids.items())[:levels]
            ]
        else:
            return [
                (level.price, level.total_qty)
                for _, level in list(self._asks.items())[:levels]
            ]

    def __repr__(self) -> str:
        bid = self.best_bid()
        ask = self.best_ask()
        spread = self.spread()
        return (
            f"LimitOrderBook(best_bid={bid:.2f}, best_ask={ask:.2f}, "
            f"spread={spread:.2f}, bids={len(self._bids)}, asks={len(self._asks)})"
        )

    # ------------------------------------------------------------------ #
    #  Internal matching                                                  #
    # ------------------------------------------------------------------ #

    def _match_bid(self, order: _Order, timestamp_ns: int) -> list[FillEvent]:
        """Try to match an incoming bid against resting asks."""
        fills = []
        while order.remaining_qty > 0 and self._asks:
            best_ask_price = self._asks.keys()[0]
            if order.price < best_ask_price:
                break   # No cross: limit price is below best ask

            level = self._asks[best_ask_price]
            while order.remaining_qty > 0 and not level.is_empty():
                resting = level.peek_head()
                fill_qty = min(order.remaining_qty, resting.remaining_qty)
                fill_price = resting.price  # Price-time priority: resting price

                # Update quantities
                order.remaining_qty -= fill_qty
                resting.remaining_qty -= fill_qty
                level.total_qty -= fill_qty

                fills.append(FillEvent(
                    timestamp_ns=timestamp_ns,
                    order_id=order.order_id,
                    fill_price=fill_price,
                    fill_qty=fill_qty,
                    side="bid",
                    is_taker=True,
                ))

                if resting.remaining_qty <= 0:
                    level.pop_head()
                    self._order_map.pop(resting.order_id, None)

            if level.is_empty():
                del self._asks[best_ask_price]

        return fills

    def _match_ask(self, order: _Order, timestamp_ns: int) -> list[FillEvent]:
        """Try to match an incoming ask against resting bids."""
        fills = []
        while order.remaining_qty > 0 and self._bids:
            best_bid_neg = self._bids.keys()[0]
            best_bid_price = -best_bid_neg
            if order.price > best_bid_price:
                break   # No cross: limit price is above best bid

            level = self._bids[best_bid_neg]
            while order.remaining_qty > 0 and not level.is_empty():
                resting = level.peek_head()
                fill_qty = min(order.remaining_qty, resting.remaining_qty)
                fill_price = resting.price

                order.remaining_qty -= fill_qty
                resting.remaining_qty -= fill_qty
                level.total_qty -= fill_qty

                fills.append(FillEvent(
                    timestamp_ns=timestamp_ns,
                    order_id=order.order_id,
                    fill_price=fill_price,
                    fill_qty=fill_qty,
                    side="ask",
                    is_taker=True,
                ))

                if resting.remaining_qty <= 0:
                    level.pop_head()
                    self._order_map.pop(resting.order_id, None)

            if level.is_empty():
                del self._bids[best_bid_neg]

        return fills
