import pytest
import math
from engine.config import HFTConfig
from engine.events import MarketTick, L3OrderAdd, L2Delta, SignalEvent
from engine.des_engine import DiscreteEventSimulator
from engine.lob_python import LimitOrderBook

class MockHFTStrategy:
    def __init__(self):
        self.tick_count = 0
        self.fill_count = 0
        self.ack_count = 0
        self.l2_count = 0

    def on_market_tick(self, tick: MarketTick, lob: LimitOrderBook):
        self.tick_count += 1
        # On 3rd tick, buy 1 unit at market
        if self.tick_count == 3:
            return [SignalEvent(
                timestamp_ns=tick.timestamp_ns,
                side="long",
                price=0.0,  # market order
                qty=1.0,
                order_type="market"
            )]
        # On 5th tick, close the long position (sell 1 unit)
        elif self.tick_count == 5:
            return [SignalEvent(
                timestamp_ns=tick.timestamp_ns,
                side="short",
                price=0.0,  # market order
                qty=1.0,
                order_type="market"
            )]
        return []

    def on_l2_delta(self, delta: L2Delta, lob: LimitOrderBook):
        self.l2_count += 1
        return []

    def on_fill(self, fill, lob: LimitOrderBook):
        self.fill_count += 1
        return []

    def on_order_ack(self, ack):
        self.ack_count += 1

def test_des_integration():
    config = HFTConfig(
        latency_ns=50_000,          # 50 microseconds
        starting_capital=100_000.0,
        fee_rate_taker=0.0004,
        warmup_events=0,
        max_events=1000
    )
    
    des = DiscreteEventSimulator(config)
    strategy = MockHFTStrategy()
    
    # Create synthetic events:
    # 1. Market order book depth setup (resting asks/bids)
    # 2. Market tick 1
    # 3. Market tick 2
    # 4. Market tick 3 (triggers buy signal at t=300ms)
    # 5. Buy signal executes (at t = 300ms + 50us = 300,050,000 ns; fills at best ask 50100.0)
    # 6. We update best bid to 50200.0 (t=350ms)
    # 7. Market tick 4
    # 8. Market tick 5 (triggers sell signal at t=500ms)
    # 9. Sell signal executes (at t = 500ms + 50us = 500,050,000 ns; fills at best bid 50200.0)
    events = [
        L3OrderAdd(timestamp_ns=100_000_000, order_id=1001, price=50100.0, qty=10.0, side="ask"),
        L3OrderAdd(timestamp_ns=100_000_000, order_id=1002, price=49900.0, qty=10.0, side="bid"),
        
        MarketTick(timestamp_ns=200_000_000, price=50000.0, qty=1.0, side="buy", trade_id=1),
        MarketTick(timestamp_ns=250_000_000, price=50000.0, qty=1.0, side="buy", trade_id=2),
        MarketTick(timestamp_ns=300_000_000, price=50000.0, qty=1.0, side="buy", trade_id=3), # triggers buy SignalEvent
        
        L3OrderAdd(timestamp_ns=350_000_000, order_id=1004, price=50200.0, qty=10.0, side="bid"),
        
        MarketTick(timestamp_ns=400_000_000, price=50100.0, qty=1.0, side="buy", trade_id=4),
        MarketTick(timestamp_ns=500_000_000, price=50100.0, qty=1.0, side="buy", trade_id=5), # triggers sell SignalEvent
    ]
    
    result = des.run(iter(events), strategy)
    
    # Assertions
    assert strategy.tick_count == 5
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "long"
    assert trade.entry_price == 50100.0
    assert trade.exit_price == 50200.0
    assert trade.qty == 1.0
    
    # PnL math: (50200 - 50100) * 1.0 = 100.0 gross.
    # Taker fee: entry fee = 50100 * 0.0004 = 20.04, exit fee = 50200 * 0.0004 = 20.08
    # Net PnL: 100.0 - 20.04 - 20.08 = 59.88
    assert math.isclose(trade.pnl, 59.88)
