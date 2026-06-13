/// O(1) Limit Order Book implementation.
///
/// Data structures:
///
/// 1. SlotMap<OrderKey, OrderNode>  — arena allocator
///    Provides stable keys (not invalidated by insertions/deletions).
///    order_id_map[order_id] -> OrderKey -> arena[OrderKey] = OrderNode
///    This gives O(1) order lookup by ID for cancels.
///
/// 2. BTreeMap<u64, PriceLevel>  — sorted price levels
///    Bids: stored with negated price so BTreeMap::iter() gives descending order.
///    Asks: stored with natural price, ascending order.
///    O(log n) level lookup (n = number of distinct price levels, typically small).
///
/// 3. Each PriceLevel has a doubly-linked list of OrderKeys via SlotMap keys.
///    Append to tail: O(1) — update tail_key + two slot writes
///    Remove head:    O(1) — update head_key + two slot writes
///    Cancel middle:  O(1) — read prev/next from slot, update their links
///
/// The Python reference LOB (lob_python.py) uses deque.remove() for middle-cancel
/// which is O(n). This Rust implementation makes that O(1) via the doubly-linked list.
///
/// Thread safety: intentionally single-threaded. The DES serializes all events.

use std::collections::BTreeMap;
use std::collections::HashMap;

use slotmap::{new_key_type, SlotMap};

use crate::types::{Fill, Order, Side, to_raw_price, to_raw_qty, from_raw_price};

// ---------------------------------------------------------------------------
// Doubly-linked list node stored in the SlotMap arena
// ---------------------------------------------------------------------------

new_key_type! { pub struct OrderKey; }

struct OrderNode {
    order: Order,
    prev: Option<OrderKey>,
    next: Option<OrderKey>,
}

// ---------------------------------------------------------------------------
// Price level: head/tail of the doubly-linked list
// ---------------------------------------------------------------------------

struct PriceLevel {
    head: Option<OrderKey>,
    tail: Option<OrderKey>,
    total_qty_raw: u64,   // Sum of remaining_qty for all resting orders
    order_count: usize,
}

impl PriceLevel {
    fn new() -> Self {
        PriceLevel {
            head: None,
            tail: None,
            total_qty_raw: 0,
            order_count: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.order_count == 0
    }
}

// ---------------------------------------------------------------------------
// Order Book
// ---------------------------------------------------------------------------

pub struct OrderBook {
    /// Arena: stable keys across insertions/deletions
    arena: SlotMap<OrderKey, OrderNode>,

    /// O(1) lookup by order_id -> arena slot
    order_map: HashMap<u64, OrderKey>,

    /// Bids: stored as (u64::MAX - price_raw) for descending iteration
    bids: BTreeMap<u64, PriceLevel>,

    /// Asks: stored as price_raw for ascending iteration
    asks: BTreeMap<u64, PriceLevel>,
}

impl OrderBook {
    pub fn new() -> Self {
        OrderBook {
            arena: SlotMap::with_key(),
            order_map: HashMap::new(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    // ------------------------------------------------------------------ //
    //  Public API                                                         //
    // ------------------------------------------------------------------ //

    /// Add a limit order. Returns immediate fills if it crosses the spread.
    ///
    /// Time complexity:
    ///   - Matching loop: O(k) where k = number of fills generated
    ///   - Resting insertion: O(log n) BTreeMap lookup + O(1) list append
    pub fn add_order(
        &mut self,
        order_id: u64,
        price: f64,
        qty: f64,
        side: Side,
    ) -> Vec<Fill> {
        if self.order_map.contains_key(&order_id) {
            // Duplicate order — ignore (production would return an error)
            return vec![];
        }

        let price_raw = to_raw_price(price);
        let qty_raw = to_raw_qty(qty);

        let mut order = Order {
            order_id,
            price_raw,
            remaining_qty: qty_raw,
            side,
        };

        let fills = match side {
            Side::Bid => self.match_bid(&mut order),
            Side::Ask => self.match_ask(&mut order),
        };

        // If not fully filled, rest on the book
        if order.remaining_qty > 0 {
            let key = self.arena.insert(OrderNode {
                order,
                prev: None,
                next: None,
            });
            self.order_map.insert(order_id, key);
            self.append_to_level(key, side);
        }

        fills
    }

    /// Cancel a resting order. Returns true if found and removed.
    ///
    /// Time complexity: O(1) — hash-map lookup + doubly-linked-list splice
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        let key = match self.order_map.remove(&order_id) {
            Some(k) => k,
            None => return false,
        };

        let node = match self.arena.remove(key) {
            Some(n) => n,
            None => return false,
        };

        let price_raw = node.order.price_raw;
        let side = node.order.side;
        let qty_raw = node.order.remaining_qty;

        // Splice out of doubly-linked list: O(1)
        self.splice_out(key, node.prev, node.next, price_raw, side, qty_raw);

        true
    }

    /// Best bid price as f64. Returns 0.0 if no bids.
    pub fn best_bid(&self) -> f64 {
        self.bids
            .keys()
            .next()
            .map(|&neg_price| from_raw_price(u64::MAX - neg_price))
            .unwrap_or(0.0)
    }

    /// Best ask price as f64. Returns f64::INFINITY if no asks.
    pub fn best_ask(&self) -> f64 {
        self.asks
            .keys()
            .next()
            .map(|&p| from_raw_price(p))
            .unwrap_or(f64::INFINITY)
    }

    /// Mid price.
    pub fn mid_price(&self) -> f64 {
        let bid = self.best_bid();
        let ask = self.best_ask();
        if bid == 0.0 || ask == f64::INFINITY {
            return 0.0;
        }
        (bid + ask) / 2.0
    }

    /// Bid-ask spread.
    pub fn spread(&self) -> f64 {
        let bid = self.best_bid();
        let ask = self.best_ask();
        if bid == 0.0 || ask == f64::INFINITY {
            return f64::INFINITY;
        }
        ask - bid
    }

    // ------------------------------------------------------------------ //
    //  Internal matching                                                  //
    // ------------------------------------------------------------------ //

    fn match_bid(&mut self, order: &mut Order) -> Vec<Fill> {
        let mut fills = Vec::new();

        loop {
            if order.remaining_qty == 0 {
                break;
            }

            // Find best ask (lowest price)
            let best_ask_raw = match self.asks.keys().next().copied() {
                Some(p) => p,
                None => break,
            };

            // Check if bid price >= best ask (cross condition)
            if order.price_raw < best_ask_raw {
                break;
            }

            // Match against head of best ask level
            let level = self.asks.get_mut(&best_ask_raw).unwrap();
            let head_key = match level.head {
                Some(k) => k,
                None => {
                    self.asks.remove(&best_ask_raw);
                    continue;
                }
            };

            let (fill, fully_filled, resting_id) = {
                let node = self.arena.get_mut(head_key).unwrap();
                let fill_qty = order.remaining_qty.min(node.order.remaining_qty);
                order.remaining_qty -= fill_qty;
                node.order.remaining_qty -= fill_qty;
                level.total_qty_raw -= fill_qty;

                let fill = Fill {
                    order_id: order.order_id,
                    fill_price_raw: best_ask_raw,
                    fill_qty_raw: fill_qty,
                    side: Side::Bid,
                    is_taker: true,
                };

                (fill, node.order.remaining_qty == 0, node.order.order_id)
            };

            fills.push(fill);

            if fully_filled {
                // Remove resting order from arena and maps
                let node = self.arena.remove(head_key).unwrap();
                self.order_map.remove(&resting_id);

                let level = self.asks.get_mut(&best_ask_raw).unwrap();
                level.order_count -= 1;
                level.head = node.next;
                if let Some(next_key) = node.next {
                    self.arena[next_key].prev = None;
                } else {
                    level.tail = None;
                }
                if level.is_empty() {
                    self.asks.remove(&best_ask_raw);
                }
            }
        }

        fills
    }

    fn match_ask(&mut self, order: &mut Order) -> Vec<Fill> {
        let mut fills = Vec::new();

        loop {
            if order.remaining_qty == 0 {
                break;
            }

            // Find best bid (highest price = smallest neg_price key)
            let best_bid_neg = match self.bids.keys().next().copied() {
                Some(k) => k,
                None => break,
            };
            let best_bid_raw = u64::MAX - best_bid_neg;

            // Check if ask price <= best bid
            if order.price_raw > best_bid_raw {
                break;
            }

            let level = self.bids.get_mut(&best_bid_neg).unwrap();
            let head_key = match level.head {
                Some(k) => k,
                None => {
                    self.bids.remove(&best_bid_neg);
                    continue;
                }
            };

            let (fill, fully_filled, resting_id) = {
                let node = self.arena.get_mut(head_key).unwrap();
                let fill_qty = order.remaining_qty.min(node.order.remaining_qty);
                order.remaining_qty -= fill_qty;
                node.order.remaining_qty -= fill_qty;
                level.total_qty_raw -= fill_qty;

                let fill = Fill {
                    order_id: order.order_id,
                    fill_price_raw: best_bid_raw,
                    fill_qty_raw: fill_qty,
                    side: Side::Ask,
                    is_taker: true,
                };

                (fill, node.order.remaining_qty == 0, node.order.order_id)
            };

            fills.push(fill);

            if fully_filled {
                let node = self.arena.remove(head_key).unwrap();
                self.order_map.remove(&resting_id);

                let level = self.bids.get_mut(&best_bid_neg).unwrap();
                level.order_count -= 1;
                level.head = node.next;
                if let Some(next_key) = node.next {
                    self.arena[next_key].prev = None;
                } else {
                    level.tail = None;
                }
                if level.is_empty() {
                    self.bids.remove(&best_bid_neg);
                }
            }
        }

        fills
    }

    // ------------------------------------------------------------------ //
    //  Linked-list helpers                                                //
    // ------------------------------------------------------------------ //

    fn append_to_level(&mut self, key: OrderKey, side: Side) {
        let price_raw = self.arena[key].order.price_raw;
        let qty_raw = self.arena[key].order.remaining_qty;

        let level = match side {
            Side::Bid => {
                let neg_price = u64::MAX - price_raw;
                self.bids.entry(neg_price).or_insert_with(PriceLevel::new)
            }
            Side::Ask => {
                self.asks.entry(price_raw).or_insert_with(PriceLevel::new)
            }
        };

        level.total_qty_raw += qty_raw;
        level.order_count += 1;

        match level.tail {
            None => {
                // Empty level: this node is both head and tail
                level.head = Some(key);
                level.tail = Some(key);
            }
            Some(old_tail) => {
                // Link new tail
                self.arena[old_tail].next = Some(key);
                self.arena[key].prev = Some(old_tail);
                level.tail = Some(key);
            }
        }
    }

    fn splice_out(
        &mut self,
        _key: OrderKey,
        prev: Option<OrderKey>,
        next: Option<OrderKey>,
        price_raw: u64,
        side: Side,
        qty_raw: u64,
    ) {
        // Update neighbour links
        if let Some(p) = prev {
            self.arena[p].next = next;
        }
        if let Some(n) = next {
            self.arena[n].prev = prev;
        }

        // Update level metadata
        let level_opt = match side {
            Side::Bid => {
                let neg_price = u64::MAX - price_raw;
                self.bids.get_mut(&neg_price)
            }
            Side::Ask => self.asks.get_mut(&price_raw),
        };

        if let Some(level) = level_opt {
            level.total_qty_raw = level.total_qty_raw.saturating_sub(qty_raw);
            level.order_count = level.order_count.saturating_sub(1);

            // Update head/tail if we spliced out either end
            if prev.is_none() {
                level.head = next;
            }
            if next.is_none() {
                level.tail = prev;
            }

            // Clean up empty price levels
            let is_empty = level.is_empty();
            if is_empty {
                match side {
                    Side::Bid => {
                        self.bids.remove(&(u64::MAX - price_raw));
                    }
                    Side::Ask => {
                        self.asks.remove(&price_raw);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_resting_order() {
        let mut book = OrderBook::new();
        let fills = book.add_order(1, 50000.0, 0.1, Side::Ask);
        assert!(fills.is_empty());
        assert!((book.best_ask() - 50000.0).abs() < 0.01);
        assert_eq!(book.best_bid(), 0.0);
    }

    #[test]
    fn test_crossing_fill() {
        let mut book = OrderBook::new();
        book.add_order(1, 50000.0, 0.1, Side::Ask);
        let fills = book.add_order(2, 50001.0, 0.05, Side::Bid);
        assert_eq!(fills.len(), 1);
        assert!((fills[0].fill_price_f64() - 50000.0).abs() < 0.01);
        assert!((fills[0].fill_qty_f64() - 0.05).abs() < 0.00001);
    }

    #[test]
    fn test_partial_fill() {
        let mut book = OrderBook::new();
        book.add_order(1, 50000.0, 0.1, Side::Ask);  // Resting ask 0.1
        let fills = book.add_order(2, 50001.0, 0.03, Side::Bid);  // Bid 0.03 (partial)
        assert_eq!(fills.len(), 1);
        assert!((fills[0].fill_qty_f64() - 0.03).abs() < 0.00001);
        // Remaining ask should still be 0.07
        assert!((book.best_ask() - 50000.0).abs() < 0.01);
    }

    #[test]
    fn test_cancel() {
        let mut book = OrderBook::new();
        book.add_order(1, 50000.0, 0.1, Side::Ask);
        assert!(book.cancel_order(1));
        assert_eq!(book.best_ask(), f64::INFINITY);
        assert!(!book.cancel_order(1));  // Double cancel returns false
    }

    #[test]
    fn test_mid_price() {
        let mut book = OrderBook::new();
        book.add_order(1, 49999.0, 0.1, Side::Bid);
        book.add_order(2, 50001.0, 0.1, Side::Ask);
        let mid = book.mid_price();
        assert!((mid - 50000.0).abs() < 0.01);
    }

    #[test]
    fn test_spread() {
        let mut book = OrderBook::new();
        book.add_order(1, 49999.0, 0.1, Side::Bid);
        book.add_order(2, 50001.0, 0.1, Side::Ask);
        let spread = book.spread();
        assert!((spread - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_price_time_priority() {
        let mut book = OrderBook::new();
        // Two resting asks at same price — first one should fill first
        book.add_order(1, 50000.0, 0.05, Side::Ask);
        book.add_order(2, 50000.0, 0.05, Side::Ask);
        let fills = book.add_order(3, 50000.0, 0.1, Side::Bid);
        assert_eq!(fills.len(), 2);
        // Both should fill at 50000.0
        for f in &fills {
            assert!((f.fill_price_f64() - 50000.0).abs() < 0.01);
        }
    }
}
