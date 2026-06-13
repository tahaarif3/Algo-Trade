import random
import pytest
import math
import hft_engine
from engine.lob_python import LimitOrderBook


def test_rust_vs_python_lob_random_ops():
    random.seed(42)
    
    python_lob = LimitOrderBook()
    rust_lob = hft_engine.OrderBook()
    
    active_order_ids = []
    
    for i in range(10000):
        # 70% Add, 30% Cancel
        op = "add" if random.random() < 0.7 or not active_order_ids else "cancel"
        
        if op == "add":
            order_id = i + 1
            # Random price around $50,000, in steps of 0.01 (tick size)
            price = round(50000.0 + random.uniform(-10.0, 10.0), 2)
            # Random quantity between 0.001 and 1.0 (steps of 0.00001 lot size scale)
            qty = round(random.uniform(0.001, 1.0), 5)
            side = "bid" if random.random() < 0.5 else "ask"
            
            # Apply to Python LOB
            py_fills = python_lob.add_order(order_id, price, qty, side)
            
            # Apply to Rust LOB
            rust_fills = rust_lob.add_order(order_id, price, qty, side)
            
            # Compare fills count
            assert len(py_fills) == len(rust_fills), f"Fills length mismatch at step {i}: py={len(py_fills)}, rust={len(rust_fills)}"
            
            # Compare fill details
            for pf, rf in zip(py_fills, rust_fills):
                assert pf.order_id == rf.order_id
                assert math.isclose(pf.fill_price, rf.fill_price)
                assert math.isclose(pf.fill_qty, rf.fill_qty)
                assert pf.side == rf.side
                assert pf.is_taker == rf.is_taker
                
            active_order_ids.append(order_id)
            
        else:
            # Cancel a random active order
            order_to_cancel = random.choice(active_order_ids)
            active_order_ids.remove(order_to_cancel)
            
            py_cancelled = python_lob.cancel_order(order_to_cancel)
            rust_cancelled = rust_lob.cancel_order(order_to_cancel)
            
            assert py_cancelled == rust_cancelled, f"Cancel status mismatch for order {order_to_cancel}: py={py_cancelled}, rust={rust_cancelled}"
            
        # Compare best bid/ask, mid price, spread
        py_bb = python_lob.best_bid()
        rust_bb = rust_lob.best_bid()
        assert math.isclose(py_bb, rust_bb), f"Best bid mismatch at step {i}: py={py_bb}, rust={rust_bb}"
        
        py_ba = python_lob.best_ask()
        rust_ba = rust_lob.best_ask()
        if py_ba == math.inf:
            assert rust_ba == math.inf or rust_ba > 1e9
        else:
            assert math.isclose(py_ba, rust_ba), f"Best ask mismatch at step {i}: py={py_ba}, rust={rust_ba}"
            
        py_mid = python_lob.mid_price()
        rust_mid = rust_lob.mid_price()
        assert math.isclose(py_mid, rust_mid), f"Mid price mismatch at step {i}: py={py_mid}, rust={rust_mid}"
        
        py_spread = python_lob.spread()
        rust_spread = rust_lob.spread()
        if math.isinf(py_spread):
            assert math.isinf(rust_spread) or rust_spread > 1e9
        else:
            assert math.isclose(py_spread, rust_spread), f"Spread mismatch at step {i}: py={py_spread}, rust={rust_spread}"
            
    print("Cross-validation completed successfully: Python vs Rust LOB produce identical results on 10k random operations.")
