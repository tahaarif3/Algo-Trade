"""
Configuration and data structures for the backtesting framework.

All structures are dataclasses with JSON serialization support,
making them ready for MCP tool parameter passing and result reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

import pandas as pd


@dataclass
class StrategyConfig:
    """All tunable parameters for a strategy + backtest run.

    Designed as a single flat config to simplify optimization loops —
    each run is fully described by one StrategyConfig instance.
    """

    # --- Indicator Parameters ---
    ema_period: int = 200
    rsi_period: int = 14
    atr_period: int = 14

    # --- Entry Thresholds (RSI mode) ---
    rsi_long_threshold: float = 30.0   # Long when RSI drops below this
    rsi_short_threshold: float = 70.0  # Short when RSI rises above this

    # --- Risk Management (ATR multipliers) ---
    atr_mult_sl: float = 2.5   # Stop-loss distance = ATR * this
    atr_mult_tp: float = 5.0   # Take-profit distance = ATR * this

    # --- Position Sizing ---
    position_pct: float = 0.10  # Fraction of capital per trade (10%)

    # --- Stochastic RSI Toggle (for Run 3 swap) ---
    use_srsi: bool = False
    srsi_window: int = 14
    srsi_smooth1: int = 3
    srsi_smooth2: int = 3
    srsi_k_long: float = 20.0   # Long when SRSI %K drops below this
    srsi_k_short: float = 80.0  # Short when SRSI %K rises above this

    # --- Regime Filter (ADX) ---
    use_adx_filter: bool = False
    adx_period: int = 14
    adx_threshold: float = 25.0       # Suppress entries when ADX > threshold (strong trend)
    adx_sl_scale_factor: float = 0.02 # SL widens by this * ADX value (dynamic risk)

    # --- Limit Order Execution ---
    use_limit_entry: bool = False
    limit_atr_offset: float = 0.2     # Entry = close ∓ (ATR * offset), passive fill

    # --- Parameter Ensembling (RSI Cluster) ---
    use_rsi_ensemble: bool = False
    rsi_ensemble_periods: list[int] = field(default_factory=lambda: [10, 14, 18])
    rsi_ensemble_min_votes: int = 2   # Minimum agreement count to trigger signal

    # --- Backtest Parameters ---
    starting_capital: float = 10_000.0
    fee_rate: float = 0.0004  # 0.04% per side (Binance futures taker)

    # --- Warmup ---
    warmup_candles: int = 210  # Must exceed ema_period + buffer

    def to_dict(self) -> dict:
        """Serialize to dict (JSON-safe for MCP transport)."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        """Deserialize from dict (e.g., from MCP tool params)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def describe_changes(self, baseline: "StrategyConfig") -> str:
        """Return a human-readable summary of parameters that differ from baseline."""
        changes = []
        for fld in self.__dataclass_fields__:
            v_self = getattr(self, fld)
            v_base = getattr(baseline, fld)
            if v_self != v_base:
                v_base_str = f"{v_base:.4f}".rstrip('0').rstrip('.') if isinstance(v_base, float) else str(v_base)
                v_self_str = f"{v_self:.4f}".rstrip('0').rstrip('.') if isinstance(v_self, float) else str(v_self)
                changes.append(f"{fld}: {v_base_str} -> {v_self_str}")
        return "; ".join(changes) if changes else "No changes (baseline)"


@dataclass
class Trade:
    """Record of a single completed trade."""

    entry_time: str          # ISO timestamp string
    exit_time: str           # ISO timestamp string
    side: str                # 'long' or 'short'
    entry_price: float
    exit_price: float
    qty: float               # Position size in base asset units
    pnl: float               # Realized profit/loss in quote currency
    pnl_pct: float           # P&L as percentage of trade value
    fee_paid: float          # Total fees (entry + exit)
    exit_reason: str         # 'sl' (stop-loss) or 'tp' (take-profit)
    duration_bars: int       # How many candles the position was held

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    """Complete output of a backtest run.

    Carries the full trade log, equity curve, configuration used,
    and computed performance metrics. Designed for serialization.
    """

    trades: list[Trade]
    equity_curve: list[dict]  # List of {timestamp, equity} dicts
    config: StrategyConfig
    metrics: dict             # Output from MetricsCalculator
    run_label: str = ""       # e.g., "Run 1 — Baseline"

    def to_dict(self) -> dict:
        return {
            "run_label": self.run_label,
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": self.equity_curve,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @property
    def summary(self) -> str:
        """One-line summary for console output."""
        m = self.metrics
        return (
            f"[{self.run_label}] "
            f"PnL: ${m.get('net_profit', 0):,.2f} | "
            f"Trades: {m.get('total_trades', 0)} | "
            f"Win: {m.get('win_rate_pct', 0):.1f}% | "
            f"PF: {m.get('profit_factor', 0):.2f} | "
            f"MaxDD: {m.get('max_drawdown_pct', 0):.2f}% | "
            f"Sharpe: {m.get('sharpe_ratio', 0):.3f}"
        )


# ---------------------------------------------------------------------------
# HFT Configuration  (additive — StrategyConfig is untouched)
# ---------------------------------------------------------------------------

@dataclass
class HFTConfig:
    """Configuration for the HFT Discrete Event Simulator.

    Covers latency simulation, order book parameters, capital model,
    and DES safety limits. JSON-serializable for MCP transport.

    Latency tier guidance:
        50_000  ns = 50µs   -- co-location / aggressive directional HFT
        100_000 ns = 100µs  -- fast VPS near exchange
        500_000 ns = 500µs  -- retail VPS, cross-region
    """

    # --- Latency simulation ---
    latency_ns: int = 50_000            # Signal-to-fill round-trip in nanoseconds
    latency_jitter_std: float = 0.0     # Std deviation of Gaussian noise on latency_ns (in nanoseconds)
    packet_drop_rate: float = 0.0       # Fraction of L2Delta events to randomly discard [0.0, 1.0]
    mc_random_seed: Optional[int] = None # Optional seed for reproducibility
    use_rust_lob: bool = False          # Use native Rust O(1) order book

    # --- Order book ---
    tick_size: float = 0.01             # Minimum price increment (BTC: $0.01)
    lot_size: float = 0.001             # Minimum order quantity (BTC: 0.001)

    # --- Capital ---
    starting_capital: float = 10_000.0
    position_pct: float = 0.10          # Fraction of capital per trade
    fee_rate_taker: float = 0.0004      # 0.04% taker (Binance futures default)
    fee_rate_maker: float = 0.0001      # 0.01% maker

    # --- DES limits ---
    max_events: int = 50_000_000        # Circuit breaker: stop after N events
    warmup_events: int = 10_000         # Events to process before recording stats

    # --- Data ---
    data_dir: str = "data/hft"          # Root dir for Parquet tick files
    symbol: str = "BTCUSDT"

    def to_dict(self) -> dict:
        """Serialize to dict (JSON-safe for MCP transport)."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "HFTConfig":
        """Deserialize from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def latency_us(self) -> float:
        """Latency in microseconds (human-readable)."""
        return self.latency_ns / 1_000.0

    def latency_ms(self) -> float:
        """Latency in milliseconds."""
        return self.latency_ns / 1_000_000.0

