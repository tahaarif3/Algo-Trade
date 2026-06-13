"""
HFT Data Loader — Parquet-backed tick data source.

Extends the existing BinanceDataLoader (OHLCV, unchanged) with a new
HFTDataLoader class that fetches, converts, and caches tick-level data
as local Parquet files.

Pipeline (one-time per symbol/date):
  1. Download Binance Vision ZIP (trades + depth snapshots)
  2. Parse CSV with pyarrow
  3. Convert timestamps ms -> ns (int64)
  4. Write to Parquet with Snappy compression
  5. Cache locally: data/hft/<SYMBOL>/<date>_trades.parquet

Subsequent runs read directly from Parquet -- no network call.

Directory layout:
  data/
    hft/
      BTCUSDT/
        2024-01-01_trades.parquet
        2024-01-01_depth.parquet
        2024-01-02_trades.parquet
        ...
    BTCUSDT_4h_2023-07-01_2025-01-01.csv   <- existing OHLCV cache
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.csv
import pyarrow.parquet as pq
import requests

from engine.events import BaseEvent, MarketTick, L2Delta
from engine.tick_parser import BinanceTradesParser, BinanceDepthParser


# Binance Vision public data base URL (no auth required)
_BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/daily"
_REQUEST_DELAY_SEC = 0.25     # courtesy pause between downloads
_DOWNLOAD_TIMEOUT_SEC = 60

_TRADES_PARSER = BinanceTradesParser()
_DEPTH_PARSER = BinanceDepthParser()


class HFTDataLoader:
    """Fetches and streams nanosecond-timestamped tick data from local Parquet files.

    On first call for a given symbol/date, downloads from Binance Vision and
    converts to Parquet. All subsequent calls are pure local reads.

    Usage:
        loader = HFTDataLoader(cache_dir='data/hft')
        for tick in loader.stream_trades('BTCUSDT', '2024-01-15'):
            ...  # MarketTick objects with timestamp_ns in nanoseconds
    """

    def __init__(self, cache_dir: str = "data/hft") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public streaming API                                               #
    # ------------------------------------------------------------------ #

    def stream_trades(
        self,
        symbol: str,
        date: str,  # 'YYYY-MM-DD'
    ) -> Iterator[MarketTick]:
        """Stream MarketTick events from local Parquet, downloading if needed.

        Events are yielded in ascending timestamp_ns order (Binance Vision
        data is already sorted chronologically).

        Args:
            symbol: e.g. 'BTCUSDT'
            date:   'YYYY-MM-DD'

        Yields:
            MarketTick events with nanosecond timestamps.
        """
        parquet_path = self._trades_parquet_path(symbol, date)
        if not parquet_path.exists():
            print(f"  [HFTDataLoader] Downloading trades: {symbol} {date}")
            self._download_and_convert_trades(symbol, date)

        print(f"  [HFTDataLoader] Streaming trades from: {parquet_path.name}")
        table = pq.read_table(str(parquet_path))
        yield from _TRADES_PARSER.parse_table(table)

    def stream_l2_snapshots(
        self,
        symbol: str,
        date: str,
        depth: int = 20,
    ) -> Iterator[L2Delta]:
        """Stream L2Delta events from local Parquet depth snapshot file.

        Args:
            symbol: e.g. 'BTCUSDT'
            date:   'YYYY-MM-DD'
            depth:  Number of price levels per side to retain.

        Yields:
            L2Delta events with nanosecond timestamps.
        """
        parquet_path = self._depth_parquet_path(symbol, date)
        if not parquet_path.exists():
            print(f"  [HFTDataLoader] Downloading depth snapshots: {symbol} {date}")
            self._download_and_convert_depth(symbol, date)

        if not parquet_path.exists():
            print(f"  [HFTDataLoader] No depth data available for {symbol} {date}, skipping.")
            return

        print(f"  [HFTDataLoader] Streaming depth from: {parquet_path.name}")
        table = pq.read_table(str(parquet_path))
        yield from _DEPTH_PARSER.parse_table(table, is_snapshot=True)

    def stream_all_events(
        self,
        symbol: str,
        date: str,
    ) -> Iterator[BaseEvent]:
        """Merge trades and depth events into a single sorted event stream.

        Merges by timestamp_ns using a two-pointer merge. Since each source
        is already sorted, this is O(n) in total events — no full sort needed.

        Args:
            symbol: e.g. 'BTCUSDT'
            date:   'YYYY-MM-DD'

        Yields:
            MarketTick and L2Delta events interleaved in timestamp order.
        """
        import heapq

        trades_iter = self.stream_trades(symbol, date)
        depth_iter = self.stream_l2_snapshots(symbol, date)

        # Use heapq.merge for O(n log k) k-way merge (k=2 here)
        yield from heapq.merge(trades_iter, depth_iter)

    # ------------------------------------------------------------------ #
    #  Download + conversion                                              #
    # ------------------------------------------------------------------ #

    def _download_and_convert_trades(self, symbol: str, date: str) -> Path:
        """Download Binance Vision trades ZIP, convert to Parquet, cache locally."""
        url = f"{_BINANCE_VISION_BASE}/trades/{symbol}/{symbol}-trades-{date}.zip"
        out_path = self._trades_parquet_path(symbol, date)

        csv_bytes = self._download_zip(url)
        if csv_bytes is None:
            print(f"  [HFTDataLoader] WARNING: No trades data for {symbol} {date}")
            return out_path

        # Parse CSV with pyarrow (standard spot trades layout)
        parse_options = pa.csv.ParseOptions(delimiter=",")
        convert_options = pa.csv.ConvertOptions(
            column_types={
                "trade_id": pa.int64(),
                "price": pa.float64(),
                "qty": pa.float64(),
                "quote_qty": pa.float64(),
                "transact_time": pa.int64(),
                "is_buyer_maker": pa.bool_(),
                "is_best_match": pa.bool_(),
            }
        )
        read_options = pa.csv.ReadOptions(
            column_names=[
                "trade_id", "price", "qty", "quote_qty",
                "transact_time", "is_buyer_maker", "is_best_match",
            ],
            skip_rows=0,
        )

        try:
            table = pa.csv.read_csv(
                io.BytesIO(csv_bytes),
                read_options=read_options,
                parse_options=parse_options,
                convert_options=convert_options,
            )
        except Exception:
            # Fallback: let pyarrow infer schema
            table = pa.csv.read_csv(io.BytesIO(csv_bytes))

        # Write to Parquet with Snappy compression
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(out_path), compression="snappy")

        print(
            f"  [HFTDataLoader] Cached {len(table)} trades -> {out_path.name} "
            f"({out_path.stat().st_size / 1024:.0f} KB)"
        )
        return out_path

    def _download_and_convert_depth(self, symbol: str, date: str) -> Path:
        """Download Binance Vision depth snapshot, convert to Parquet."""
        # Binance Vision also has bookDepth files
        url = f"{_BINANCE_VISION_BASE}/bookDepth/{symbol}/{symbol}-bookDepth-{date}.zip"
        out_path = self._depth_parquet_path(symbol, date)

        csv_bytes = self._download_zip(url)
        if csv_bytes is None:
            # bookDepth not available for all dates — silently skip
            return out_path

        try:
            table = pa.csv.read_csv(io.BytesIO(csv_bytes))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, str(out_path), compression="snappy")
            print(
                f"  [HFTDataLoader] Cached depth snapshot -> {out_path.name} "
                f"({out_path.stat().st_size / 1024:.0f} KB)"
            )
        except Exception as e:
            print(f"  [HFTDataLoader] WARNING: Could not parse depth data: {e}")

        return out_path

    def _download_zip(self, url: str) -> bytes | None:
        """Download a Binance Vision ZIP and return the inner CSV bytes."""
        try:
            resp = requests.get(url, timeout=_DOWNLOAD_TIMEOUT_SEC)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 404:
                return None  # Data not available for this date
            raise
        except Exception as e:
            print(f"  [HFTDataLoader] Download failed: {e}")
            return None

        time.sleep(_REQUEST_DELAY_SEC)  # Rate-limit courtesy pause

        # Extract first CSV from the ZIP
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            return zf.read(csv_names[0])

    # ------------------------------------------------------------------ #
    #  Cache paths                                                        #
    # ------------------------------------------------------------------ #

    def _trades_parquet_path(self, symbol: str, date: str) -> Path:
        return self.cache_dir / symbol / f"{date}_trades.parquet"

    def _depth_parquet_path(self, symbol: str, date: str) -> Path:
        return self.cache_dir / symbol / f"{date}_depth.parquet"

    def list_cached_dates(self, symbol: str) -> list[str]:
        """Return sorted list of dates with cached trade data."""
        symbol_dir = self.cache_dir / symbol
        if not symbol_dir.exists():
            return []
        dates = [
            p.stem.replace("_trades", "")
            for p in symbol_dir.glob("*_trades.parquet")
        ]
        return sorted(dates)

    def validate_parquet(self, symbol: str, date: str) -> dict:
        """Validate a cached Parquet trade file.

        Returns a dict with keys: valid, row_count, min_ts_ns, max_ts_ns,
        is_monotonic, has_nulls.
        """
        path = self._trades_parquet_path(symbol, date)
        if not path.exists():
            return {"valid": False, "reason": "file not found"}

        table = pq.read_table(str(path))

        if "transact_time" not in table.column_names:
            return {"valid": False, "reason": "missing transact_time column"}

        ts_col = table.column("transact_time").to_pylist()
        min_ts = min(ts_col) * 1_000_000
        max_ts = max(ts_col) * 1_000_000
        is_monotonic = all(ts_col[i] <= ts_col[i + 1] for i in range(len(ts_col) - 1))
        has_nulls = table.column("price").null_count > 0

        return {
            "valid": is_monotonic and not has_nulls,
            "row_count": len(table),
            "min_ts_ns": min_ts,
            "max_ts_ns": max_ts,
            "is_monotonic": is_monotonic,
            "has_nulls": has_nulls,
            "file_size_kb": round(path.stat().st_size / 1024, 1),
        }
