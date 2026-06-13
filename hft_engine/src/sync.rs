use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use pyo3::prelude::*;

// 1. Position Status Enum
#[pyclass(name = "PositionStatus")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PositionStatus {
    Flat = 0,
    Long = 1,
    Short = 2,
    PendingClose = 3,
    PendingOpen = 4,
}

#[pymethods]
impl PositionStatus {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl From<u8> for PositionStatus {
    fn from(val: u8) -> Self {
        match val {
            0 => PositionStatus::Flat,
            1 => PositionStatus::Long,
            2 => PositionStatus::Short,
            3 => PositionStatus::PendingClose,
            _ => PositionStatus::PendingOpen,
        }
    }
}

// 2. Execution Report
#[pyclass(name = "ExecutionReport")]
#[derive(Clone, Debug)]
pub struct PyExecutionReport {
    #[pyo3(get, set)]
    pub timestamp_ns: u64,
    #[pyo3(get, set)]
    pub order_id: i64,
    #[pyo3(get, set)]
    pub fill_price: f64,
    #[pyo3(get, set)]
    pub fill_qty: f64,
    #[pyo3(get, set)]
    pub net_pnl: f64,
    #[pyo3(get, set)]
    pub status: PositionStatus,
}

#[pymethods]
impl PyExecutionReport {
    #[new]
    pub fn new(
        timestamp_ns: u64,
        order_id: i64,
        fill_price: f64,
        fill_qty: f64,
        net_pnl: f64,
        status: PositionStatus,
    ) -> Self {
        Self {
            timestamp_ns,
            order_id,
            fill_price,
            fill_qty,
            net_pnl,
            status,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecutionReport(ts={}, order_id={}, price={:.2}, qty={:.6}, pnl={:.2}, status={:?})",
            self.timestamp_ns,
            self.order_id,
            self.fill_price,
            self.fill_qty,
            self.net_pnl,
            self.status
        )
    }
}

// 3. Real-Time State Broker
#[pyclass(name = "RealTimeStateBroker")]
#[derive(Clone)]
pub struct PyRealTimeStateBroker {
    status_raw: Arc<AtomicU8>,
}

#[pymethods]
impl PyRealTimeStateBroker {
    #[new]
    pub fn new() -> Self {
        Self {
            status_raw: Arc::new(AtomicU8::new(PositionStatus::Flat as u8)),
        }
    }

    pub fn update_status(&self, status: PositionStatus) {
        self.status_raw.store(status as u8, Ordering::Release);
    }

    pub fn get_status(&self) -> PositionStatus {
        PositionStatus::from(self.status_raw.load(Ordering::Acquire))
    }
}

// 4. Trace Event for Testing
#[pyclass(name = "TraceEvent")]
#[derive(Clone, Debug)]
pub struct PyTraceEvent {
    #[pyo3(get)]
    pub timestamp_ns: u64,
    #[pyo3(get)]
    pub thread_id: usize,
    #[pyo3(get)]
    pub event_type: String,
    #[pyo3(get)]
    pub metadata: String,
}

#[pymethods]
impl PyTraceEvent {
    fn __repr__(&self) -> String {
        format!(
            "TraceEvent(ts={}, thread={}, type={}, meta={})",
            self.timestamp_ns, self.thread_id, self.event_type, self.metadata
        )
    }
}

// 5. Trace Logger
#[pyclass(name = "TraceLogger")]
#[derive(Clone)]
pub struct PyTraceLogger {
    events: Arc<Mutex<Vec<PyTraceEvent>>>,
}

#[pymethods]
impl PyTraceLogger {
    #[new]
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn log_event(
        &self,
        timestamp_ns: u64,
        thread_id: usize,
        event_type: &str,
        metadata: &str,
    ) {
        let mut events = self.events.lock().unwrap();
        events.push(PyTraceEvent {
            timestamp_ns,
            thread_id,
            event_type: event_type.to_string(),
            metadata: metadata.to_string(),
        });
    }

    pub fn get_events(&self) -> Vec<PyTraceEvent> {
        let events = self.events.lock().unwrap();
        events.clone()
    }

    pub fn clear(&self) {
        let mut events = self.events.lock().unwrap();
        events.clear();
    }
}

// 6. Bar Boundary Gatekeeper
#[pyclass(name = "BarBoundaryGatekeeper")]
pub struct PyBarBoundaryGatekeeper {
    latest_processed_tick_ns: Arc<AtomicU64>,
    state_broker: PyRealTimeStateBroker,
    reports: Arc<Mutex<Vec<PyExecutionReport>>>,
}

#[pymethods]
impl PyBarBoundaryGatekeeper {
    #[new]
    pub fn new(state_broker: PyRealTimeStateBroker) -> Self {
        Self {
            latest_processed_tick_ns: Arc::new(AtomicU64::new(0)),
            state_broker,
            reports: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn update_processed_tick(&self, timestamp_ns: u64) {
        self.latest_processed_tick_ns
            .store(timestamp_ns, Ordering::Release);
    }

    pub fn get_latest_tick(&self) -> u64 {
        self.latest_processed_tick_ns.load(Ordering::Acquire)
    }

    pub fn push_report(&self, report: PyExecutionReport) {
        let mut reports = self.reports.lock().unwrap();
        reports.push(report);
    }

    pub fn pull_reports(&self) -> Vec<PyExecutionReport> {
        let mut reports = self.reports.lock().unwrap();
        std::mem::take(&mut *reports)
    }

    pub fn reconcile_boundary_state(
        &self,
        boundary_ts_ns: u64,
        timeout_ms: u64,
    ) -> PyResult<bool> {
        let start = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);

        // 1. Wait until the Micro Engine has processed all ticks up to boundary
        while self.latest_processed_tick_ns.load(Ordering::Acquire) < boundary_ts_ns {
            if start.elapsed() > timeout {
                return Ok(false); // timeout
            }
            thread::yield_now();
        }

        // 2. Wait until state is no longer PendingClose
        while self.state_broker.get_status() == PositionStatus::PendingClose {
            if start.elapsed() > timeout {
                return Ok(false); // timeout
            }
            std::hint::spin_loop();
        }

        Ok(true)
    }
}
