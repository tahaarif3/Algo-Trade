"""
TrendOscillatorStrategy — Trend-following momentum reversal system.

Architecture:
    Trend Filter:    200-period EMA establishes directional bias
    Momentum Trigger: RSI-14 (or Stochastic RSI) identifies overextended
                      counter-trend moves as entry opportunities
    Risk Management:  ATR-based dynamic SL/TP with structural 1:2 R:R
    Position Sizing:  Fixed percentage of available capital

This strategy buys oversold dips in uptrends and sells overbought rallies
in downtrends — a mean-reversion entry within a trend-following framework.

Satisfies StrategyProtocol. Stateless: all decisions are made from the
current candle row's indicator values. No internal state beyond config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import ta as ta_lib

if TYPE_CHECKING:
    from engine.config import StrategyConfig


class TrendOscillatorStrategy:
    """Trend + oscillator fusion strategy.

    Configurable via StrategyConfig — all thresholds, periods, and
    multipliers are externalized for optimization loop control.
    """

    def __init__(self, config: "StrategyConfig"):
        self.config = config

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicator columns to the OHLCV DataFrame.

        Columns added:
            ema_{period}:  Exponential Moving Average
            rsi_{period}:  Relative Strength Index
            atr_{period}:  Average True Range
            srsi_k:        Stochastic RSI %K (if use_srsi=True)
            srsi_d:        Stochastic RSI %D (if use_srsi=True)

        Returns a copy — does not mutate the input DataFrame.
        """
        result = df.copy()
        c = self.config

        # --- Trend Filter: EMA ---
        result[f"ema_{c.ema_period}"] = ta_lib.trend.ema_indicator(
            close=result["close"], window=c.ema_period
        )

        # --- Momentum: RSI ---
        result[f"rsi_{c.rsi_period}"] = ta_lib.momentum.rsi(
            close=result["close"], window=c.rsi_period
        )

        # --- Volatility: ATR ---
        result[f"atr_{c.atr_period}"] = ta_lib.volatility.average_true_range(
            high=result["high"],
            low=result["low"],
            close=result["close"],
            window=c.atr_period,
        )

        # --- Optional: Stochastic RSI ---
        if c.use_srsi:
            # ta library returns SRSI in 0-100 scale
            result["srsi_k"] = ta_lib.momentum.stochrsi_k(
                close=result["close"],
                window=c.srsi_window,
                smooth1=c.srsi_smooth1,
                smooth2=c.srsi_smooth2,
            )
            result["srsi_d"] = ta_lib.momentum.stochrsi_d(
                close=result["close"],
                window=c.srsi_window,
                smooth1=c.srsi_smooth1,
                smooth2=c.srsi_smooth2,
            )
            # Normalize to 0-100 if library returns 0-1
            if result["srsi_k"].max() <= 1.0:
                result["srsi_k"] = result["srsi_k"] * 100
                result["srsi_d"] = result["srsi_d"] * 100

        # --- Optional: ADX Regime Filter ---
        if c.use_adx_filter:
            result[f"adx_{c.adx_period}"] = ta_lib.trend.adx(
                high=result["high"],
                low=result["low"],
                close=result["close"],
                window=c.adx_period,
            )

        # --- Optional: RSI Ensemble ---
        if c.use_rsi_ensemble:
            for period in c.rsi_ensemble_periods:
                col = f"rsi_{period}"
                if col not in result.columns:
                    result[col] = ta_lib.momentum.rsi(
                        close=result["close"], window=period
                    )

        return result

    def should_long(self, row: pd.Series) -> bool:
        """Long entry signal: price above EMA + momentum oversold.

        Logic:
            1. Close STRICTLY above 200-EMA (uptrend confirmed)
            2. RSI below threshold (oversold dip in uptrend)
               OR Stochastic RSI %K below threshold (if SRSI mode)
        """
        c = self.config
        ema_col = f"ema_{c.ema_period}"
        rsi_col = f"rsi_{c.rsi_period}"

        # Guard against NaN indicators (warmup period)
        if pd.isna(row.get(ema_col)):
            return False

        if c.use_rsi_ensemble:
            if any(pd.isna(row.get(f"rsi_{p}")) for p in c.rsi_ensemble_periods):
                return False
        else:
            if pd.isna(row.get(rsi_col)):
                return False

        if c.use_adx_filter:
            adx_val = row.get(f"adx_{c.adx_period}")
            if pd.isna(adx_val) or adx_val > c.adx_threshold:
                return False

        # Trend filter: must be in uptrend
        if row["close"] <= row[ema_col]:
            return False

        # Momentum trigger
        if c.use_srsi:
            srsi_k = row.get("srsi_k")
            if pd.isna(srsi_k):
                return False
            return srsi_k < c.srsi_k_long
        elif c.use_rsi_ensemble:
            votes = sum(
                1 for p in c.rsi_ensemble_periods
                if row.get(f"rsi_{p}", float("nan")) < c.rsi_long_threshold
            )
            return votes >= c.rsi_ensemble_min_votes
        else:
            return row[rsi_col] < c.rsi_long_threshold

    def should_short(self, row: pd.Series) -> bool:
        """Short entry signal: price below EMA + momentum overbought.

        Logic:
            1. Close STRICTLY below 200-EMA (downtrend confirmed)
            2. RSI above threshold (overbought rally in downtrend)
               OR Stochastic RSI %K above threshold (if SRSI mode)
        """
        c = self.config
        ema_col = f"ema_{c.ema_period}"
        rsi_col = f"rsi_{c.rsi_period}"

        # Guard against NaN indicators (warmup period)
        if pd.isna(row.get(ema_col)):
            return False

        if c.use_rsi_ensemble:
            if any(pd.isna(row.get(f"rsi_{p}")) for p in c.rsi_ensemble_periods):
                return False
        else:
            if pd.isna(row.get(rsi_col)):
                return False

        if c.use_adx_filter:
            adx_val = row.get(f"adx_{c.adx_period}")
            if pd.isna(adx_val) or adx_val > c.adx_threshold:
                return False

        # Trend filter: must be in downtrend
        if row["close"] >= row[ema_col]:
            return False

        # Momentum trigger
        if c.use_srsi:
            srsi_k = row.get("srsi_k")
            if pd.isna(srsi_k):
                return False
            return srsi_k > c.srsi_k_short
        elif c.use_rsi_ensemble:
            votes = sum(
                1 for p in c.rsi_ensemble_periods
                if row.get(f"rsi_{p}", float("nan")) > c.rsi_short_threshold
            )
            return votes >= c.rsi_ensemble_min_votes
        else:
            return row[rsi_col] > c.rsi_short_threshold

    def get_stop_loss(self, entry_price: float, side: str, row: pd.Series) -> float:
        """Compute ATR-based stop-loss price."""
        c = self.config
        atr_col = f"atr_{c.atr_period}"
        atr_val = row[atr_col]

        mult = c.atr_mult_sl
        if c.use_adx_filter:
            adx_val = row.get(f"adx_{c.adx_period}", 0.0)
            if not pd.isna(adx_val):
                mult = c.atr_mult_sl * (1.0 + c.adx_sl_scale_factor * adx_val)

        if side == "long":
            return entry_price - (mult * atr_val)
        else:  # short
            return entry_price + (mult * atr_val)

    def get_take_profit(self, entry_price: float, side: str, row: pd.Series) -> float:
        """Compute ATR-based take-profit price."""
        c = self.config
        atr_col = f"atr_{c.atr_period}"
        atr_val = row[atr_col]

        if side == "long":
            return entry_price + (c.atr_mult_tp * atr_val)
        else:  # short
            return entry_price - (c.atr_mult_tp * atr_val)

    def get_position_qty(self, capital: float, entry_price: float) -> float:
        """Compute position size: fixed % of capital.

        Returns quantity in base asset units (e.g., BTC).
        """
        allocation = capital * self.config.position_pct
        return allocation / entry_price

    def get_entry_price(self, side: str, row: pd.Series) -> float | None:
        """Return a limit entry price, or None to use market (close).

        When limit pricing is active, the engine fills at this price
        only if the next bar's low (long) or high (short) reaches it.
        Otherwise the order is cancelled (no fill).
        """
        c = self.config
        if not c.use_limit_entry:
            return None  # Market order at close (backward compatible)
        atr_col = f"atr_{c.atr_period}"
        atr_val = row[atr_col]
        close = row["close"]
        if side == "long":
            return close - (atr_val * c.limit_atr_offset)  # Bid below close
        else:
            return close + (atr_val * c.limit_atr_offset)  # Ask above close
