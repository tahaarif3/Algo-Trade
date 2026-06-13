/// Shared types for the HFT engine.
///
/// All types are repr(C)-compatible where needed for PyO3 and use
/// fixed-point integer prices (price_raw = price * PRICE_SCALE) to
/// avoid floating-point comparison issues at nanosecond resolution.

/// Price scale factor: store prices as integer ticks.
/// 1 BTC price tick = 0.01 USD => scale by 100 to get integer cents.
/// For BTCUSDT at ~$50,000: max price_raw = 50_000_00 = 5_000_000 (fits in u64 easily)
pub const PRICE_SCALE: u64 = 100;

/// Convert a float price to the internal fixed-point representation.
#[inline(always)]
pub fn to_raw_price(price: f64) -> u64 {
    (price * PRICE_SCALE as f64).round() as u64
}

/// Convert a fixed-point price back to float.
#[inline(always)]
pub fn from_raw_price(raw: u64) -> f64 {
    raw as f64 / PRICE_SCALE as f64
}

/// Quantity scale: store quantities as integer units of 0.00001 BTC.
pub const QTY_SCALE: u64 = 100_000;

#[inline(always)]
pub fn to_raw_qty(qty: f64) -> u64 {
    (qty * QTY_SCALE as f64).round() as u64
}

#[inline(always)]
pub fn from_raw_qty(raw: u64) -> f64 {
    raw as f64 / QTY_SCALE as f64
}

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

impl Side {
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "bid" | "buy" | "long" => Side::Bid,
            _ => Side::Ask,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            Side::Bid => "bid",
            Side::Ask => "ask",
        }
    }
}

/// A single limit order resting in the book.
#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: u64,
    pub price_raw: u64,       // Fixed-point price
    pub remaining_qty: u64,   // Fixed-point quantity remaining
    pub side: Side,
}

/// A fill record: created when a new order crosses the spread.
#[derive(Debug, Clone)]
pub struct Fill {
    pub order_id: u64,
    pub fill_price_raw: u64,
    pub fill_qty_raw: u64,
    pub side: Side,
    pub is_taker: bool,
}

impl Fill {
    pub fn fill_price_f64(&self) -> f64 {
        from_raw_price(self.fill_price_raw)
    }

    pub fn fill_qty_f64(&self) -> f64 {
        from_raw_qty(self.fill_qty_raw)
    }
}
