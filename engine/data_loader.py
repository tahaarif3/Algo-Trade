"""
Binance public REST API candle data loader.

Fetches historical kline (candlestick) data from Binance's public API.
No API key required. Data is cached locally as CSV to avoid redundant fetches.

Implements DataSourceProtocol for future swapability (e.g., MCP-based
data source, database loader, or alternative exchange APIs).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests


# Binance public klines endpoint — no authentication required
_BINANCE_KLINES_URL = "https://api.binance.us/api/v3/klines"

# Maximum candles per request (Binance limit)
_MAX_LIMIT = 1000

# Rate-limit courtesy pause between paginated requests (ms)
_REQUEST_DELAY_SEC = 0.25


class BinanceDataLoader:
    """Fetches and caches OHLCV candle data from Binance public API.

    Usage:
        loader = BinanceDataLoader(cache_dir="data")
        df = loader.fetch_candles("BTCUSDT", "4h", "2023-07-01", "2024-12-31")
    """

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, symbol: str, interval: str, start: str, end: str) -> str:
        """Generate a deterministic cache filename."""
        safe_name = f"{symbol}_{interval}_{start}_{end}.csv"
        return os.path.join(self.cache_dir, safe_name)

    def fetch_candles(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV data, using local cache if available.

        Args:
            symbol: Binance symbol (e.g., 'BTCUSDT').
            interval: Candle interval (e.g., '4h', '1h', '1d').
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD' (inclusive).

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            and a DatetimeIndex.
        """
        cache_file = self._cache_path(symbol, interval, start_date, end_date)

        if os.path.exists(cache_file):
            print(f"  [DataLoader] Using cached data: {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=["timestamp"])
            df.set_index("timestamp", inplace=True)
            return df

        print(f"  [DataLoader] Fetching {symbol} {interval} from Binance API...")
        print(f"  [DataLoader] Range: {start_date} -> {end_date}")

        # Convert dates to millisecond timestamps (UTC)
        start_ms = int(
            datetime.strptime(start_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        end_ms = int(
            datetime.strptime(end_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )

        all_candles = []
        current_start = start_ms
        page = 0

        while current_start < end_ms:
            page += 1
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": _MAX_LIMIT,
            }

            resp = requests.get(_BINANCE_KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            raw = resp.json()

            if not raw:
                break

            all_candles.extend(raw)
            print(
                f"  [DataLoader] Page {page}: fetched {len(raw)} candles "
                f"(total: {len(all_candles)})"
            )

            # Next page starts after the last candle's close time
            last_close_time = raw[-1][6]  # Close time field
            current_start = last_close_time + 1

            if len(raw) < _MAX_LIMIT:
                break  # No more data available

            time.sleep(_REQUEST_DELAY_SEC)

        if not all_candles:
            raise ValueError(
                f"No candle data returned for {symbol} {interval} "
                f"from {start_date} to {end_date}. "
                "Check symbol name and date range."
            )

        # Parse into DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ],
        )

        # Keep only OHLCV columns, convert types
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        # Remove any duplicate timestamps
        df = df[~df.index.duplicated(keep="first")]

        # Cache to disk
        df.to_csv(cache_file)
        print(
            f"  [DataLoader] Cached {len(df)} candles -> {cache_file}"
        )

        return df

    def validate_data(
        self,
        df: pd.DataFrame,
        expected_start: str,
        expected_end: str,
        min_candles: int = 100,
    ) -> dict:
        """Validate fetched data for completeness.

        Returns a dict with validation results (useful for MCP tool responses).
        """
        actual_start = df.index.min()
        actual_end = df.index.max()
        total = len(df)
        has_nulls = df[["open", "high", "low", "close"]].isnull().any().any()

        result = {
            "valid": total >= min_candles and not has_nulls,
            "total_candles": total,
            "actual_start": str(actual_start),
            "actual_end": str(actual_end),
            "expected_start": expected_start,
            "expected_end": expected_end,
            "has_null_prices": bool(has_nulls),
            "null_count": int(df[["open", "high", "low", "close"]].isnull().sum().sum()),
        }

        if not result["valid"]:
            reasons = []
            if total < min_candles:
                reasons.append(f"Only {total} candles (need ≥{min_candles})")
            if has_nulls:
                reasons.append(f"{result['null_count']} null price values")
            result["failure_reasons"] = reasons

        return result
