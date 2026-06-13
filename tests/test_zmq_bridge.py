import pytest
import math
from engine.config import HFTConfig
from engine.events import MarketTick, SignalEvent
from engine.des_engine import DiscreteEventSimulator


class SlippageTestStrategy:
    def __init__(self, entry_time_ns):
        self.entry_time_ns = entry_time_ns
        self.triggered = False
        
    def on_market_tick(self, tick: MarketTick, book) -> list[SignalEvent]:
        # Trigger market buy at the specified entry timestamp
        if tick.timestamp_ns == self.entry_time_ns and not self.triggered:
            self.triggered = True
            return [SignalEvent(
                timestamp_ns=tick.timestamp_ns,
                side="long",
                price=0.0,  # market order
                qty=1.0,
                order_type="market"
            )]
        return []
        
    def on_fill(self, fill, book) -> list[SignalEvent]:
        return []
        
    def on_l2_delta(self, delta, book) -> list[SignalEvent]:
        return []
        
    def on_order_ack(self, ack) -> None:
        pass


def test_latency_bridge_slippage_model():
    # Scenario A: Slippage (Market moves in the 50us latency window)
    config_a = HFTConfig(
        latency_ns=50_000,          # 50 microseconds
        starting_capital=100_000.0,
        fee_rate_taker=0.0,         # Zero fees to isolate slippage PnL
        fee_rate_maker=0.0,
        warmup_events=0,
        max_events=100
    )
    des_a = DiscreteEventSimulator(config_a)
    strategy_a = SlippageTestStrategy(entry_time_ns=100_000_000)
    
    events_a = [
        # t=100ms: Tick at 50,000.0 (triggers buy signal)
        MarketTick(timestamp_ns=100_000_000, price=50000.0, qty=1.0, side="buy", trade_id=1),
        # t=100.02ms (inside window): Price moves up to 50,100.0
        MarketTick(timestamp_ns=100_020_000, price=50100.0, qty=1.0, side="buy", trade_id=2),
        # t=100.05ms: Delayed order executes (latency_ns=50,000)
        # It should fill at the new price 50,100.0
        # Then we exit flat using the close price at the end
        MarketTick(timestamp_ns=100_100_000, price=50200.0, qty=1.0, side="buy", trade_id=3)
    ]
    
    result_a = des_a.run(iter(events_a), strategy_a)
    
    assert len(result_a.trades) == 1, "Should have executed 1 trade"
    trade_a = result_a.trades[0]
    # Entry price should be 50,100.0 (slippage of 100.0 from trigger price 50,000.0)
    assert trade_a.entry_price == 50100.0
    assert trade_a.exit_price == 50200.0  # Force close at end of simulator
    
    
    # Scenario B: No Slippage (Market does not move in the latency window)
    config_b = HFTConfig(
        latency_ns=50_000,
        starting_capital=100_000.0,
        fee_rate_taker=0.0,
        fee_rate_maker=0.0,
        warmup_events=0,
        max_events=100
    )
    des_b = DiscreteEventSimulator(config_b)
    strategy_b = SlippageTestStrategy(entry_time_ns=100_000_000)
    
    events_b = [
        # t=100ms: Tick at 50,000.0 (triggers buy signal)
        MarketTick(timestamp_ns=100_000_000, price=50000.0, qty=1.0, side="buy", trade_id=1),
        # t=100.05ms: Delayed order executes at trigger price 50,000.0
        MarketTick(timestamp_ns=100_100_000, price=50200.0, qty=1.0, side="buy", trade_id=2)
    ]
    
    result_b = des_b.run(iter(events_b), strategy_b)
    
    assert len(result_b.trades) == 1
    trade_b = result_b.trades[0]
    # Entry price should be 50,000.0 (zero slippage)
    assert trade_b.entry_price == 50000.0
    assert trade_b.exit_price == 50200.0
