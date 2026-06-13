/// PyO3 module entry point: hft_engine Python extension.
///
/// Exposes:
///   PyOrderBook  — Python-callable wrapper around the Rust OrderBook
///   PyFill       — Python-readable fill record
///
/// Import from Python:
///   from hft_engine import PyOrderBook, PyFill

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

mod types;
mod order_book;

use types::Side;
use order_book::OrderBook;


// ---------------------------------------------------------------------------
// Python-visible fill record
// ---------------------------------------------------------------------------

#[pyclass(name = "Fill")]
#[derive(Clone, Debug)]
pub struct PyFill {
    #[pyo3(get)]
    pub order_id: u64,

    #[pyo3(get)]
    pub fill_price: f64,

    #[pyo3(get)]
    pub fill_qty: f64,

    #[pyo3(get)]
    pub side: String,

    #[pyo3(get)]
    pub is_taker: bool,
}

#[pymethods]
impl PyFill {
    fn __repr__(&self) -> String {
        format!(
            "Fill(order_id={}, price={:.2}, qty={:.6}, side={}, taker={})",
            self.order_id, self.fill_price, self.fill_qty, self.side, self.is_taker
        )
    }
}


// ---------------------------------------------------------------------------
// Python-visible Order Book
// ---------------------------------------------------------------------------

/// O(1) Limit Order Book backed by a Rust doubly-linked list + SlotMap arena.
///
/// Example:
///     from hft_engine import PyOrderBook
///     book = PyOrderBook()
///     fills = book.add_order(1, 50000.0, 0.1, 'ask')   # resting ask
///     fills = book.add_order(2, 50001.0, 0.05, 'bid')  # crossing bid -> fill
///     print(fills[0].fill_price)  # 50000.0
///     print(book.best_bid())      # 0.0 (no resting bids)
///     print(book.best_ask())      # 50000.0 (0.05 remaining)
#[pyclass(name = "OrderBook")]
pub struct PyOrderBook {
    inner: OrderBook,
}

#[pymethods]
impl PyOrderBook {
    #[new]
    pub fn new() -> Self {
        PyOrderBook {
            inner: OrderBook::new(),
        }
    }

    /// Add a limit order to the book.
    ///
    /// Args:
    ///     order_id (int): Unique order identifier.
    ///     price (float):  Limit price in USD.
    ///     qty (float):    Quantity in base asset (e.g. BTC).
    ///     side (str):     'bid' or 'ask'.
    ///     timestamp_ns (int, optional): Event timestamp.
    ///
    /// Returns:
    ///     list[Fill]: Immediate fills if the order crossed the spread.
    #[pyo3(signature = (order_id, price, qty, side, timestamp_ns=None))]
    pub fn add_order(
        &mut self,
        order_id: u64,
        price: f64,
        qty: f64,
        side: &str,
        timestamp_ns: Option<u64>,
    ) -> PyResult<Vec<PyFill>> {
        if price <= 0.0 {
            return Err(PyValueError::new_err("price must be positive"));
        }
        if qty <= 0.0 {
            return Err(PyValueError::new_err("qty must be positive"));
        }

        let rust_side = Side::from_str(side);
        let fills = self.inner.add_order(order_id, price, qty, rust_side);

        Ok(fills.into_iter().map(|f| PyFill {
            order_id: f.order_id,
            fill_price: f.fill_price_f64(),
            fill_qty: f.fill_qty_f64(),
            side: f.side.to_str().to_string(),
            is_taker: f.is_taker,
        }).collect())
    }

    /// Cancel a resting order by ID.
    ///
    /// Returns:
    ///     bool: True if the order was found and cancelled.
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        self.inner.cancel_order(order_id)
    }

    /// Best bid price (highest resting bid). Returns 0.0 if no bids.
    pub fn best_bid(&self) -> f64 {
        self.inner.best_bid()
    }

    /// Best ask price (lowest resting ask). Returns float('inf') if no asks.
    pub fn best_ask(&self) -> f64 {
        self.inner.best_ask()
    }

    /// Mid price: (best_bid + best_ask) / 2. Returns 0.0 if book is one-sided.
    pub fn mid_price(&self) -> f64 {
        self.inner.mid_price()
    }

    /// Bid-ask spread in price units.
    pub fn spread(&self) -> f64 {
        self.inner.spread()
    }

    fn __repr__(&self) -> String {
        format!(
            "OrderBook(best_bid={:.2}, best_ask={:.2}, spread={:.2})",
            self.best_bid(),
            self.best_ask(),
            self.spread(),
        )
    }
}


// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn hft_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrderBook>()?;
    m.add_class::<PyFill>()?;
    Ok(())
}
