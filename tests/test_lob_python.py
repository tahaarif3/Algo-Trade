import pytest
import math
from engine.lob_python import LimitOrderBook

def test_lob_basic_add_cancel():
    lob = LimitOrderBook()
    
    # Add bid
    fills = lob.add_order(order_id=1, price=50000.0, qty=1.5, side="bid")
    assert len(fills) == 0
    assert lob.best_bid() == 50000.0
    assert lob.best_ask() == math.inf
    
    # Add ask
    fills = lob.add_order(order_id=2, price=50100.0, qty=2.0, side="ask")
    assert len(fills) == 0
    assert lob.best_ask() == 50100.0
    
    # Cancel bid
    cancelled = lob.cancel_order(order_id=1)
    assert cancelled is True
    assert lob.best_bid() == 0.0
    
    # Cancel non-existent
    cancelled = lob.cancel_order(order_id=999)
    assert cancelled is False

def test_lob_matching():
    lob = LimitOrderBook()
    
    # Resting ask
    lob.add_order(order_id=1, price=50000.0, qty=1.0, side="ask")
    
    # Incoming bid that crosses
    fills = lob.add_order(order_id=2, price=50000.0, qty=0.4, side="bid")
    assert len(fills) == 1
    fill = fills[0]
    assert fill.order_id == 2
    assert fill.fill_price == 50000.0
    assert fill.fill_qty == 0.4
    assert fill.side == "bid"
    assert fill.is_taker is True
    
    # Next bid that completely fills the remaining ask
    fills = lob.add_order(order_id=3, price=50100.0, qty=1.0, side="bid")
    assert len(fills) == 1
    fill = fills[0]
    assert fill.order_id == 3
    assert fill.fill_price == 50000.0  # Resting price
    assert fill.fill_qty == 0.6
    assert fill.side == "bid"
    
    # The ask should be fully filled and removed
    assert lob.best_ask() == math.inf
