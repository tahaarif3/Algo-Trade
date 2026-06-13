import pytest
import math
from engine.config import HFTConfig
from engine.des_engine import DiscreteEventSimulator
from engine.hft_data_loader import HFTDataLoader
from strategies.hft_market_maker import HFTMarketMaker


def test_e2e_hft_market_maker():
    # Load 1 day of Binance Vision trades data (cached during Phase 5)
    loader = HFTDataLoader(cache_dir="data/hft")
    symbol = "BTCUSDT"
    date = "2024-01-01"
    
    # Check if the cache file exists
    parquet_path = loader._trades_parquet_path(symbol, date)
    if not parquet_path.exists():
        pytest.skip("Binance Vision daily trades cache file not found. Skipping e2e test.")
        
    print("\nLoading events for end-to-end backtest...")
    # Stream first 10,000 events to keep the integration test fast but representative
    all_events = list(loader.stream_trades(symbol, date))
    events_slice = all_events[:20000]
    print(f"Loaded {len(events_slice)} tick events.")
    
    # 1. Run backtest with Python LOB
    config_py = HFTConfig(
        use_rust_lob=False,
        starting_capital=100000.0,
        latency_ns=50_000,
        fee_rate_taker=0.0004,
        fee_rate_maker=0.0001,
        warmup_events=0,
        max_events=100000
    )
    des_py = DiscreteEventSimulator(config_py)
    strategy_py = HFTMarketMaker(tick_size=0.01, profit_target=0.10, stop_loss=0.05, qty=0.1)
    
    print("Running backtest with Python LOB...")
    result_py = des_py.run(iter(events_slice), strategy_py)
    print(f"Python LOB: {len(result_py.trades)} trades executed. Final equity: ${result_py.equity_curve[-1]['equity']:.2f}")
    
    # 2. Run backtest with Rust LOB
    config_rust = HFTConfig(
        use_rust_lob=True,
        starting_capital=100000.0,
        latency_ns=50_000,
        fee_rate_taker=0.0004,
        fee_rate_maker=0.0001,
        warmup_events=0,
        max_events=100000
    )
    des_rust = DiscreteEventSimulator(config_rust)
    strategy_rust = HFTMarketMaker(tick_size=0.01, profit_target=0.10, stop_loss=0.05, qty=0.1)
    
    print("Running backtest with Rust LOB...")
    result_rust = des_rust.run(iter(events_slice), strategy_rust)
    print(f"Rust LOB: {len(result_rust.trades)} trades executed. Final equity: ${result_rust.equity_curve[-1]['equity']:.2f}")
    
    # 3. Assert identical results
    assert len(result_py.trades) == len(result_rust.trades), "Number of trades mismatch"
    assert len(result_py.trades) > 0, "No trades were executed"
    
    for i, (t_py, t_rust) in enumerate(zip(result_py.trades, result_rust.trades)):
        assert t_py.side == t_rust.side, f"Trade {i} side mismatch"
        assert math.isclose(t_py.entry_price, t_rust.entry_price), f"Trade {i} entry price mismatch: py={t_py.entry_price}, rust={t_rust.entry_price}"
        assert math.isclose(t_py.exit_price, t_rust.exit_price), f"Trade {i} exit price mismatch: py={t_py.exit_price}, rust={t_rust.exit_price}"
        assert math.isclose(t_py.pnl, t_rust.pnl), f"Trade {i} PnL mismatch: py={t_py.pnl}, rust={t_rust.pnl}"
        assert math.isclose(t_py.fee_paid, t_rust.fee_paid), f"Trade {i} fee mismatch: py={t_py.fee_paid}, rust={t_rust.fee_paid}"
        
    py_final_equity = result_py.equity_curve[-1]["equity"]
    rust_final_equity = result_rust.equity_curve[-1]["equity"]
    assert math.isclose(py_final_equity, rust_final_equity), f"Final equity mismatch: py={py_final_equity}, rust={rust_final_equity}"
    
    print("End-to-end HFT integration test completed successfully. Python and Rust LOB produce identical execution results.")
