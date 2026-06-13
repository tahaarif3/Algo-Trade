import pytest
import os
import pandas as pd
from engine.config import StrategyConfig
from engine.backtest import BacktestEngine
from strategies.trend_oscillator import TrendOscillatorStrategy
from engine.data_loader import BinanceDataLoader

@pytest.fixture
def real_data():
    """Load cached BTCUSDT candle data."""
    # The cache file BTCUSDT_4h_2023-07-01_2025-01-01.csv is expected in the data/ directory.
    # If not present (e.g. clean workspace), we raise a skip or error.
    cache_dir = "data"
    symbol = "BTCUSDT"
    interval = "4h"
    start_date = "2023-07-01"
    end_date = "2025-01-01"
    
    loader = BinanceDataLoader(cache_dir=cache_dir)
    # This will use cached file if available, or fetch it.
    df = loader.fetch_candles(symbol, interval, start_date, end_date)
    return df

def test_determinism(real_data):
    """Assert that running the same config twice produces identical results (bit-for-bit equivalence)."""
    config1 = StrategyConfig(
        ema_period=200,
        rsi_period=14,
        atr_period=14,
        starting_capital=10000.0
    )
    config2 = StrategyConfig(
        ema_period=200,
        rsi_period=14,
        atr_period=14,
        starting_capital=10000.0
    )
    
    strategy1 = TrendOscillatorStrategy(config1)
    strategy2 = TrendOscillatorStrategy(config2)
    
    engine = BacktestEngine()
    
    res1 = engine.run(real_data, strategy1, config1)
    res2 = engine.run(real_data, strategy2, config2)
    
    # Assert exact match of metrics
    assert res1.metrics == res2.metrics
    assert len(res1.trades) == len(res2.trades)
    assert len(res1.trades) > 0, "Should have trades to compare"
    
    # Assert exact trade log equivalence
    for t1, t2 in zip(res1.trades, res2.trades):
        assert t1.entry_time == t2.entry_time
        assert t1.exit_time == t2.exit_time
        assert t1.side == t2.side
        assert pytest.approx(t1.entry_price) == t2.entry_price
        assert pytest.approx(t1.exit_price) == t2.exit_price
        assert pytest.approx(t1.pnl) == t2.pnl
        assert t1.exit_reason == t2.exit_reason
        
    # Assert exact equity curve equivalence
    assert len(res1.equity_curve) == len(res2.equity_curve)
    for eq1, eq2 in zip(res1.equity_curve, res2.equity_curve):
        assert eq1["timestamp"] == eq2["timestamp"]
        assert pytest.approx(eq1["equity"]) == eq2["equity"]

def test_rsi_period_divergence(real_data):
    """Assert that altering the RSI period leads to different trades and net profit."""
    config_baseline = StrategyConfig(rsi_period=14)
    config_altered = StrategyConfig(rsi_period=5)
    
    strategy_baseline = TrendOscillatorStrategy(config_baseline)
    strategy_altered = TrendOscillatorStrategy(config_altered)
    
    engine = BacktestEngine()
    
    res_baseline = engine.run(real_data, strategy_baseline, config_baseline)
    res_altered = engine.run(real_data, strategy_altered, config_altered)
    
    # Verify that the two configurations produced distinct trade outputs
    assert len(res_baseline.trades) != len(res_altered.trades), (
        f"Trade counts should differ: baseline={len(res_baseline.trades)}, altered={len(res_altered.trades)}"
    )
    assert res_baseline.metrics["net_profit"] != res_altered.metrics["net_profit"], (
        f"Net profits should differ: baseline={res_baseline.metrics['net_profit']}, altered={res_altered.metrics['net_profit']}"
    )

def test_atr_mult_divergence(real_data):
    """Assert that altering ATR risk multipliers changes risk/reward, leading to divergent equity curves."""
    config_baseline = StrategyConfig(atr_mult_sl=2.5, atr_mult_tp=5.0)
    config_altered = StrategyConfig(atr_mult_sl=0.5, atr_mult_tp=1.0)
    
    strategy_baseline = TrendOscillatorStrategy(config_baseline)
    strategy_altered = TrendOscillatorStrategy(config_altered)
    
    engine = BacktestEngine()
    
    res_baseline = engine.run(real_data, strategy_baseline, config_baseline)
    res_altered = engine.run(real_data, strategy_altered, config_altered)
    
    # With tight SL/TP, trade count or duration and overall equity curves should diverge significantly
    assert len(res_baseline.trades) > 0
    assert len(res_altered.trades) > 0
    
    # Verify net profit difference
    assert res_baseline.metrics["net_profit"] != res_altered.metrics["net_profit"]
    
    # Assert exit reasons or trade count differ
    assert len(res_baseline.trades) != len(res_altered.trades) or \
           res_baseline.metrics["net_profit"] != res_altered.metrics["net_profit"]

def test_ema_period_divergence(real_data):
    """Assert that altering the EMA trend filter period alters exposure and trade counts."""
    config_baseline = StrategyConfig(ema_period=200)
    config_altered = StrategyConfig(ema_period=20)
    
    strategy_baseline = TrendOscillatorStrategy(config_baseline)
    strategy_altered = TrendOscillatorStrategy(config_altered)
    
    engine = BacktestEngine()
    
    res_baseline = engine.run(real_data, strategy_baseline, config_baseline)
    res_altered = engine.run(real_data, strategy_altered, config_altered)
    
    assert res_baseline.metrics["net_profit"] != res_altered.metrics["net_profit"]
    assert len(res_baseline.trades) != len(res_altered.trades)

def test_zero_trades_guard():
    """Assert that on a completely flat dataset, the engine reports zero trades and does not crash."""
    n_candles = 300
    dates = pd.date_range(start="2024-01-01", periods=n_candles, freq="4h", tz="UTC")
    # Completely flat price
    flat_data = pd.DataFrame({
        "open": [10000.0] * n_candles,
        "high": [10000.0] * n_candles,
        "low": [10000.0] * n_candles,
        "close": [10000.0] * n_candles,
        "volume": [10.0] * n_candles
    }, index=dates)
    
    config = StrategyConfig(ema_period=50, rsi_period=14)
    strategy = TrendOscillatorStrategy(config)
    
    engine = BacktestEngine()
    result = engine.run(flat_data, strategy, config)
    
    assert result.metrics["total_trades"] == 0
    assert result.metrics["net_profit"] == 0.0
    assert result.metrics["ending_balance"] == config.starting_capital
    assert len(result.trades) == 0
    assert len(result.equity_curve) == n_candles - config.warmup_candles
