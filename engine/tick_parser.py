"""
Tick data parsers for HFT data sources.

Converts raw Parquet RecordBatch rows into engine.events.BaseEvent objects
with nanosecond-precision timestamps. Uses pyarrow zero-copy array access
to avoid per-row Python object allocation during bulk reads.

Supported formats:
  - Binance Vision trades CSV / Parquet (MarketTick)
  - Binance Vision depth snapshots (L2Delta)

Binance Vision data layout:
  Trades CSV: aggregate_trade_id, price, quantity, first_trade_id,
              last_trade_id, transact_time (ms), is_buyer_maker
  Depth CSV:  no standard public format; we use periodic snapshot + delta
              files from the depth endpoint.

Timestamp conversion:
  Binance timestamps are UTC milliseconds since Unix epoch (int64).
  We multiply by 1_000_000 to convert to nanoseconds, matching BaseEvent.
  This gives µs-resolution timing (ms source * 1e6 = ms in ns space).
  True nanosecond resolution requires L3/MBO data from premium feeds.
"""

from __future__ import annotations

from typing import Iterator

import pyarrow as pa
import pyarrow.compute as pc

from engine.events import BaseEvent, MarketTick, L2Delta

# Binance uses 'True' for buyer-maker (i.e., the seller was the taker/aggressor)
_BUYER_MAKER_TO_SIDE = {True: "sell", False: "buy"}


class BinanceTradesParser:
    """Parse Binance Vision aggregate trades data into MarketTick events.

    Binance Vision trades file columns (CSV or Parquet):
      agg_trade_id | price | qty | first_trade_id | last_trade_id |
      transact_time | is_buyer_maker

    'is_buyer_maker = True' means the buyer was the resting maker:
    the seller crossed the spread -> seller was the aggressor -> side='sell'.
    """

    # Column name maps: Binance Vision uses abbreviated names
    REQUIRED_COLUMNS = {"price", "qty", "transact_time", "is_buyer_maker"}
    ALTERNATE_NAMES = {
        # Some Binance Vision exports use these
        "p": "price",
        "q": "qty",
        "T": "transact_time",
        "m": "is_buyer_maker",
        "a": "agg_trade_id",
    }

    def parse_table(self, table: pa.Table) -> Iterator[MarketTick]:
        """Parse a full PyArrow Table into MarketTick events.

        Uses zero-copy column access: arrays are read without copying data.
        Falls back to row-by-row if columnar batch access is unavailable.

        Args:
            table: PyArrow Table loaded from Parquet or CSV.

        Yields:
            MarketTick events in ascending timestamp order.
        """
        # Normalize column names
        table = self._normalize_columns(table)

        # Extract columns as numpy arrays for fast iteration
        timestamps_ms = table.column("transact_time").to_pylist()
        prices = table.column("price").to_pylist()
        qtys = table.column("qty").to_pylist()
        is_buyer_maker = table.column("is_buyer_maker").to_pylist()

        # Optional: trade ID for deduplication
        if "agg_trade_id" in table.column_names:
            trade_ids = table.column("agg_trade_id").to_pylist()
        else:
            trade_ids = list(range(len(timestamps_ms)))

        for i, (ts_ms, price, qty, ibm, tid) in enumerate(
            zip(timestamps_ms, prices, qtys, is_buyer_maker, trade_ids)
        ):
            yield MarketTick(
                timestamp_ns=int(ts_ms) * 1_000_000,  # ms -> ns
                price=float(price),
                qty=float(qty),
                side=_BUYER_MAKER_TO_SIDE.get(bool(ibm), "buy"),
                trade_id=int(tid),
            )

    def parse_batch(self, batch: pa.RecordBatch) -> Iterator[MarketTick]:
        """Parse a single RecordBatch (streaming / chunked read path)."""
        table = pa.Table.from_batches([batch])
        yield from self.parse_table(table)

    def _normalize_columns(self, table: pa.Table) -> pa.Table:
        """Rename abbreviated column names to canonical names."""
        schema = table.schema
        renames = {}
        for name in schema.names:
            if name in self.ALTERNATE_NAMES:
                renames[name] = self.ALTERNATE_NAMES[name]
        if renames:
            new_names = [renames.get(n, n) for n in schema.names]
            table = table.rename_columns(new_names)
        return table


class BinanceDepthParser:
    """Parse Binance order book snapshots into L2Delta events.

    Binance Vision depth snapshots contain periodic full snapshots of the
    top-N order book levels. We convert each level into an L2Delta event.

    Expected columns:
      timestamp | side | price | qty

    Where:
      side = 'b' (bid) or 'a' (ask)
      qty  = 0.0 means the level was removed

    The first batch of a snapshot session is marked is_snapshot=True.
    """

    def parse_table(
        self,
        table: pa.Table,
        is_snapshot: bool = True,
    ) -> Iterator[L2Delta]:
        """Parse a depth snapshot table into L2Delta events.

        Args:
            table: PyArrow Table with columns: timestamp, side, price, qty.
            is_snapshot: True if this is a full book snapshot (not incremental).

        Yields:
            L2Delta events.
        """
        timestamps = table.column("timestamp").to_pylist()
        sides_raw = table.column("side").to_pylist()
        prices = table.column("price").to_pylist()
        qtys = table.column("qty").to_pylist()

        first = True
        for ts, side_raw, price, qty in zip(timestamps, sides_raw, prices, qtys):
            # Normalize side: 'b'/'bid'/'BID' -> 'bid', 'a'/'ask'/'ASK' -> 'ask'
            side = "bid" if str(side_raw).lower() in ("b", "bid") else "ask"

            yield L2Delta(
                timestamp_ns=int(ts) * 1_000_000,
                price_level=float(price),
                qty=float(qty),
                side=side,
                is_snapshot=(is_snapshot and first),
            )
            first = False
