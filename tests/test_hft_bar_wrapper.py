import pytest
import pandas as pd
from engine.config import StrategyConfig, HFTConfig
from engine.data_loader import BinanceDataLoader
from engine.hft_data_loader import HFTDataLoader
from engine.des_engine import DiscreteEventSimulator
from engine.hft_bar_wrapper import BarToHFTWrapper
from strategies.trend_oscillator import TrendOscillatorStrategy


def test_bar_to_hft_wrapper_execution():
    loader = BinanceDataLoader(cache_dir="data")
    df = loader.fetch_candles("BTCUSDT", "4h", "2023-07-01", "2025-01-01")

    # Initialize strategy and compute indicators
    strategy_config = StrategyConfig(
        starting_capital=10000.0,
        fee_rate=0.0004,
        warmup_candles=200,
    )
    strategy = TrendOscillatorStrategy(strategy_config)
    df_enriched = strategy.compute_indicators(df)

    # Load daily ticks
    hft_loader = HFTDataLoader(cache_dir="data/hft")
    symbol = "BTCUSDT"
    date = "2024-07-01"

    ticks = list(hft_loader.stream_trades(symbol, date))
    assert len(ticks) > 0, "No tick data loaded"

    # Slice first 50,000 ticks for quick testing
    ticks_slice = ticks[:50000]

    hft_config = HFTConfig(
        use_rust_lob=True,
        starting_capital=10000.0,
        latency_ns=50_000,
        fee_rate_taker=0.0004,
        fee_rate_maker=0.0001,
        warmup_events=0,
        max_events=100000,
    )
    des = DiscreteEventSimulator(hft_config)
    wrapper = BarToHFTWrapper(df_enriched, strategy, warmup_candles=200)
    wrapper.des = des

    result = des.run(iter(ticks_slice), wrapper)
    assert result is not None
    assert result.metrics is not None
