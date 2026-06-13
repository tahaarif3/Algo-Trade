"""
Cross-validation test: Python LOB vs Rust PyO3 LOB.

Feeds the same synthetic sequence of limit orders into both the reference
pure-Python LimitOrderBook and the compiled Rust PyOrderBook.
Verifies that both books produce identically matching fills (price, qty, side,
and order ID) and end with the exact same best bid/ask and spread.
"""

from engine.lob_python import LimitOrderBook as PythonLOB
from hft_engine import OrderBook as RustLOB

def run_cross_validation():
    print("Initializing both LOBs...")
    py_lob = PythonLOB()
    rs_lob = RustLOB()

    # Synthetic sequence of orders
    orders = [
        # (order_id, price, qty, side)
        (1, 50000.0, 1.0, "ask"),
        (2, 50010.0, 0.5, "ask"),
        (3, 49990.0, 0.5, "bid"),
        (4, 49980.0, 1.0, "bid"),
        # Crossing orders
        (5, 50000.0, 0.5, "bid"),  # Should fill 0.5 of order 1
        (6, 49990.0, 1.0, "ask"),  # Should fill all 0.5 of order 3, rest on book as ask
    ]

    print("\n--- Applying Synthetic Order Sequence ---")
    
    for oid, price, qty, side in orders:
        print(f"\n[Order {oid}] {side.upper()} {qty} @ {price}")
        
        # Apply to Python LOB
        py_fills = py_lob.add_order(oid, price, qty, side)
        
        # Apply to Rust LOB
        rs_fills = rs_lob.add_order(oid, price, qty, side)

        # Compare Fills
        assert len(py_fills) == len(rs_fills), f"Fill count mismatch! Py: {len(py_fills)}, Rust: {len(rs_fills)}"
        
        if py_fills:
            print("  Fills matched:")
            for pf, rf in zip(py_fills, rs_fills):
                print(f"    Py: {pf}")
                print(f"    Rs: {rf}")
                
                # Check data equivalence
                assert pf.order_id == rf.order_id
                assert abs(pf.fill_price - rf.fill_price) < 1e-9
                assert abs(pf.fill_qty - rf.fill_qty) < 1e-9
                # pf side might be an enum or string, handle accordingly
                assert str(pf.side).lower() == str(rf.side).lower()

        # Compare Book Top-of-Book State
        assert abs(py_lob.best_bid() - rs_lob.best_bid()) < 1e-9 or (py_lob.best_bid() == 0.0 and rs_lob.best_bid() == 0.0)
        assert abs(py_lob.best_ask() - rs_lob.best_ask()) < 1e-9 or (py_lob.best_ask() == float('inf') and rs_lob.best_ask() == float('inf'))
        
        print(f"  Top of book matched! Bid: {rs_lob.best_bid()} | Ask: {rs_lob.best_ask()}")

    print("\n--- Testing Cancels ---")
    print("Canceling remaining 0.5 of order 1...")
    py_c1 = py_lob.cancel_order(1)
    rs_c1 = rs_lob.cancel_order(1)
    assert py_c1 == rs_c1, f"Cancel mismatch! Py: {py_c1}, Rust: {rs_c1}"
    assert py_c1 is True, "Order 1 should have been canceled"
    
    print("Canceling order 6...")
    py_c6 = py_lob.cancel_order(6)
    rs_c6 = rs_lob.cancel_order(6)
    assert py_c6 == rs_c6
    
    print("Top of book after cancels:")
    print(f"  Py: Bid: {py_lob.best_bid()} | Ask: {py_lob.best_ask()}")
    print(f"  Rs: Bid: {rs_lob.best_bid()} | Ask: {rs_lob.best_ask()}")
    
    assert abs(py_lob.best_bid() - rs_lob.best_bid()) < 1e-9
    assert abs(py_lob.best_ask() - rs_lob.best_ask()) < 1e-9

    print("\n✅ CROSS-VALIDATION SUCCESSFUL: Python LOB exactly matches Rust LOB")

if __name__ == "__main__":
    run_cross_validation()
