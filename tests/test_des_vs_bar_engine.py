import pytest
import math
import pandas as pd
from engine.config import StrategyConfig, HFTConfig
from engine.backtest import BacktestEngine
from engine.des_engine import DiscreteEventSimulator
from engine.events import MarketTick, SignalEvent
from strategies.trend_oscillator import TrendOscillatorStrategy
from engine.data_loader import BinanceDataLoader


def candles_to_ticks(df: pd.DataFrame) -> list[MarketTick]:
    ticks = []
    
    for i, (ts, row) in enumerate(df.iterrows()):
        # ts is a pd.Timestamp, ts.value is nanoseconds since epoch
        candle_start_ns = ts.value
        
        # We generate 4 ticks: Open, Low, High, Close
        p_open = row["open"]
        p_high = row["high"]
        p_low = row["low"]
        p_close = row["close"]
        
        ticks.append(MarketTick(timestamp_ns=candle_start_ns, price=p_open, qty=1.0, side="buy"))
        ticks.append(MarketTick(timestamp_ns=candle_start_ns + 1 * 3600 * 1_000_000_000, price=p_low, qty=1.0, side="sell"))
        ticks.append(MarketTick(timestamp_ns=candle_start_ns + 2 * 3600 * 1_000_000_000, price=p_high, qty=1.0, side="buy"))
        ticks.append(MarketTick(timestamp_ns=candle_start_ns + 3 * 3600 * 1_000_000_000, price=p_close, qty=1.0, side="sell"))
        
    return ticks


class TrendOscillatorDESWrapper:
    def __init__(self, df_enriched, strategy_osc, warmup_candles):
        self.df = df_enriched
        self.strategy = strategy_osc
        self.warmup = warmup_candles
        self.position_side = None
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        self.qty = 0.0
        self.des = None
        self.tick_count = 0
        
    def on_market_tick(self, tick: MarketTick, book) -> list[SignalEvent]:
        bar_idx = self.tick_count // 4
        tick_idx_in_bar = self.tick_count % 4
        self.tick_count += 1
        
        if bar_idx < self.warmup or bar_idx >= len(self.df):
            return []
            
        row = self.df.iloc[bar_idx]
        
        # 1. Position management: check SL/TP exits
        if self.position_side == "long":
            if tick.price <= self.sl_price:
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="short",
                    price=self.sl_price,
                    qty=self.qty,
                    order_type="market"
                )]
            elif tick.price >= self.tp_price:
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="short",
                    price=self.tp_price,
                    qty=self.qty,
                    order_type="market"
                )]
        elif self.position_side == "short":
            if tick.price >= self.sl_price:
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="long",
                    price=self.sl_price,
                    qty=self.qty,
                    order_type="market"
                )]
            elif tick.price <= self.tp_price:
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="long",
                    price=self.tp_price,
                    qty=self.qty,
                    order_type="market"
                )]
                
        # 2. Entries: check at close tick of the bar (tick_idx_in_bar == 3)
        if self.position_side is None and tick_idx_in_bar == 3:
            if self.strategy.should_long(row):
                self.qty = self.strategy.get_position_qty(self.des._capital, tick.price)
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="long",
                    price=0.0,
                    qty=self.qty,
                    order_type="market"
                )]
            elif self.strategy.should_short(row):
                self.qty = self.strategy.get_position_qty(self.des._capital, tick.price)
                return [SignalEvent(
                    timestamp_ns=tick.timestamp_ns,
                    side="short",
                    price=0.0,
                    qty=self.qty,
                    order_type="market"
                )]
                
        return []
        
    def on_fill(self, fill, book) -> list[SignalEvent]:
        bar_idx = (self.tick_count - 1) // 4
        row = self.df.iloc[bar_idx]
        
        if self.position_side is None:
            # Entry fill
            self.position_side = "long" if fill.side == "bid" else "short"
            self.entry_price = fill.fill_price
            self.qty = fill.fill_qty
            self.sl_price = self.strategy.get_stop_loss(self.entry_price, self.position_side, row)
            self.tp_price = self.strategy.get_take_profit(self.entry_price, self.position_side, row)
        else:
            # Exit fill
            self.position_side = None
            self.entry_price = 0.0
            self.sl_price = 0.0
            self.tp_price = 0.0
            self.qty = 0.0
        return []
        
    def on_l2_delta(self, delta, book) -> list[SignalEvent]:
        return []
        
    def on_order_ack(self, ack) -> None:
        pass


def test_des_vs_bar_engine_equivalence():
    loader = BinanceDataLoader(cache_dir="data")
    df = loader.fetch_candles("BTCUSDT", "4h", "2023-07-01", "2025-01-01")
    
    bt_end = pd.Timestamp("2024-12-31", tz="UTC")
    df_backtest = df[df.index <= bt_end].copy()
    
    # 1. Run Bar Backtest
    strategy_config = StrategyConfig(
        starting_capital=10000.0,
        fee_rate=0.0004,
        warmup_candles=200,
    )
    strategy = TrendOscillatorStrategy(strategy_config)
    bar_engine = BacktestEngine()
    result_bar = bar_engine.run(df_backtest, strategy, strategy_config)
    
    # Enrich the candles data with indicators for the wrapper to use
    df_enriched = strategy.compute_indicators(df_backtest)
    
    # 2. Run DES Backtest
    hft_config = HFTConfig(
        use_rust_lob=False,
        starting_capital=10000.0,
        latency_ns=0,          # Zero latency to match instant bar close entry
        fee_rate_taker=0.0004,
        fee_rate_maker=0.0004,
        warmup_events=0,
        max_events=100000
    )
    des = DiscreteEventSimulator(hft_config)
    des_wrapper = TrendOscillatorDESWrapper(df_enriched, strategy, warmup_candles=200)
    des_wrapper.des = des
    
    ticks = candles_to_ticks(df_backtest)
    result_des = des.run(iter(ticks), des_wrapper)
    
    print(f"\nBar Engine: {len(result_bar.trades)} trades. Final Balance: ${result_bar.equity_curve[-1]['equity']:.2f}")
    print(f"DES Engine: {len(result_des.trades)} trades. Final Balance: ${result_des.equity_curve[-1]['equity']:.2f}")
    
    assert len(result_bar.trades) > 0, "Bar backtest made no trades"
    assert len(result_des.trades) > 0, "DES backtest made no trades"
    
    # Assert trade count is close
    assert abs(len(result_bar.trades) - len(result_des.trades)) <= 1, "Trade count mismatch"
    
    bar_pnl = result_bar.equity_curve[-1]["equity"] - 10000.0
    des_pnl = result_des.equity_curve[-1]["equity"] - 10000.0
    
    print(f"Bar Net PnL: ${bar_pnl:.2f}")
    print(f"DES Net PnL: ${des_pnl:.2f}")
    
    # Assert final equity is within 5% of starting capital
    assert abs(bar_pnl - des_pnl) < 0.05 * 10000.0, f"PnL mismatch exceeds 5% limit: bar={bar_pnl:.2f}, des={des_pnl:.2f}"
