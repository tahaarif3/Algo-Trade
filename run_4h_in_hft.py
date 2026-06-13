import os
import sys
import json
import pandas as pd
import numpy as np
from engine.config import StrategyConfig, HFTConfig
from engine.data_loader import BinanceDataLoader
from engine.hft_data_loader import HFTDataLoader
from engine.des_engine import DiscreteEventSimulator
from engine.hft_bar_wrapper import BarToHFTWrapper
from strategies.trend_oscillator import TrendOscillatorStrategy
from engine.metrics import MetricsCalculator


def main():
    # 1. Load 4H candle data and compute indicators
    print("Loading 4H candles...")
    loader = BinanceDataLoader(cache_dir="data")
    df = loader.fetch_candles("BTCUSDT", "4h", "2023-07-01", "2025-01-01")

    # Slice to include warmup + backtest period
    bt_end = pd.Timestamp("2024-12-31", tz="UTC")
    df_backtest = df[df.index <= bt_end].copy()

    # Initialize strategy and compute indicators
    strategy_config = StrategyConfig(
        starting_capital=10000.0,
        fee_rate=0.0004,
        warmup_candles=200,
    )
    strategy = TrendOscillatorStrategy(strategy_config)
    df_enriched = strategy.compute_indicators(df_backtest)

    # 2. Load high-resolution tick data for the target range: 2024-07-01 to 2024-07-03
    print("Loading tick data...")
    hft_loader = HFTDataLoader(cache_dir="data/hft")
    dates = ["2024-07-01", "2024-07-02", "2024-07-03"]

    ticks = []
    for date in dates:
        print(f"Streaming trades for {date}...")
        ticks.extend(hft_loader.stream_trades("BTCUSDT", date))

    print(f"Loaded {len(ticks)} tick events.")

    # 3. Initialize DES with HFTConfig
    hft_config = HFTConfig(
        use_rust_lob=True,
        starting_capital=10000.0,
        latency_ns=50_000,  # 50 microseconds latency
        fee_rate_taker=0.0004,
        fee_rate_maker=0.0001,
        warmup_events=0,
        max_events=100_000_000,  # high limit to process all ticks
    )
    des = DiscreteEventSimulator(hft_config)

    # 4. Wrap the 4H strategy
    wrapper = BarToHFTWrapper(df_enriched, strategy, warmup_candles=200)
    wrapper.des = des

    # 5. Run the simulator
    print("Running simulator...")
    result = des.run(iter(ticks), wrapper)

    # Print summary
    print("\n" + "=" * 60)
    print("HFT BACKTEST RESULTS")
    print("=" * 60)
    print(MetricsCalculator.format_metrics_table(result.metrics))
    print(f"Summary: {result.summary}")

    print("\nTrades Executed in HFT:")
    for i, t in enumerate(result.trades):
        print(
            f"Trade {i+1}: {t.side} Entry: {t.entry_time} @ {t.entry_price:.2f}, Exit: {t.exit_time} @ {t.exit_price:.2f}, PnL: ${t.pnl:.2f} ({t.pnl_pct:.2f}%), Reason: {t.exit_reason}"
        )

    # Compare with Bar Engine
    # Let's load the baseline trades from run_1_baseline.json that fall in this date range
    baseline_path = os.path.join("runs", "4h", "run_1_baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)

        baseline_trades = baseline_data.get("trades", [])
        target_trades = []
        for t in baseline_trades:
            # Check if entry_time is between 2024-07-01 and 2024-07-03
            if (
                "2024-07-01" in t["entry_time"]
                or "2024-07-02" in t["entry_time"]
                or "2024-07-03" in t["entry_time"]
            ):
                target_trades.append(t)

        print("\n" + "=" * 60)
        print("BAR ENGINE BASELINE TRADES FOR SAME PERIOD")
        print("=" * 60)
        for i, t in enumerate(target_trades):
            print(
                f"Trade {i+1}: {t['side']} Entry: {t['entry_time']} @ {t['entry_price']:.2f}, Exit: {t['exit_time']} @ {t['exit_price']:.2f}, PnL: ${t['pnl']:.2f} ({t['pnl_pct']:.2f}%), Reason: {t['exit_reason']}"
            )


if __name__ == "__main__":
    main()
