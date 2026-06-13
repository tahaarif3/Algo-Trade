# low-Level Blueprint: State Visibility and Sequence Alignment in Rust HFT Core

This engineering blueprint details the architectural fixes required to resolve the state-leak and synchronization lag between the **Macro Strategy Layer** (operating on 4-hour bar boundaries) and the **Micro Execution Engine** (operating on tick-level event streams).

---

## Phase 1: Temporal Ordering & Event Sequencing Analysis

### 1.1 The Boundary Timeline ($T_{boundary}$)

The sequence of events near a 4-hour bar boundary is highly sensitive to network latency, CPU execution delays, and thread scheduling. Below is the temporal trace of a stop-out event occurring near the boundary $T_{boundary}$:

```
     Tick N (SL/TP Trigger)      T_boundary (4H Bar Closes)
            |                              |
------------+------------------------------+----------------------------> Time (ns)
            |                              |
      T_tick = T_boundary - 1ms            T_eval = T_boundary
            |                              |
            v                              v
   [Execution Thread]             [Strategy Thread]
   processes Tick N               wakes up to run
   and submits Exit order         on_bar_close()
            |                              |
            |   Order Latency (e.g., 50µs) |
            +---------------------> T_fill (Fill arrives)
                                  (Exit executed)
```

At $T_{tick} = T_{boundary} - 1\text{ ms}$, Tick $N$ enters the engine. The price matches the stop-loss (SL) or take-profit (TP) threshold, triggering an immediate market exit order. 

Two critical issues occur in this sequence:

1. **State Lag (Asynchronous Latency)**: The exit market order is processed by the execution engine and sent to the matching engine. It takes $50\ \mu\text{s}$ (round-trip + matching latency) to execute. The fill event `FillEvent` arrives back at the engine at $T_{fill}$. However, the Strategy Layer thread wakes up to evaluate the next 4H signal at $T_{eval} = T_{boundary}$. Because $T_{boundary} < T_{fill}$, the strategy thread reads the stale position state (believing it is still in a trade) and fails to generate a new entry signal.
2. **Memory Visibility Race**: Even if the execution engine processes the `FillEvent` slightly before $T_{boundary}$ (i.e., $T_{fill} < T_{boundary}$), the memory write updating the portfolio state in the execution thread might not be immediately visible to the strategy thread due to CPU cache coherency lag or lack of memory barriers.

---

## Phase 2: Design of a Thread-Safe, Real-Time State Broker

To establish low-latency, lock-free communication between the execution loop and the strategy evaluator, we implement a **State Broker** using atomic primitives and a Single-Producer Single-Consumer (`spsc`) ring buffer.

### 2.1 Atomic Position Status (`AtomicU8`)

We represent the current position status using a lock-free `AtomicU8` register. This bypasses heavy mutex locks during hot-path tick processing.

```rust
use std::sync::atomic::{AtomicU8, Ordering};

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PositionStatus {
    Flat = 0,
    Long = 1,
    Short = 2,
    PendingClose = 3,
    PendingOpen = 4,
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
```

### 2.2 Memory Ordering Guarantees

To ensure absolute cache visibility without the overhead of `SeqCst` (sequentially consistent) memory ordering, we utilize `Acquire-Release` semantics:

* **Writer (Micro Execution Engine)**: Updates the state register using `Release` ordering. This guarantees that all prior memory writes (updates to Cash Balance, Realized PnL, margin requirements, etc.) are committed to cache and visible to other threads before the status byte changes.
* **Reader (Macro Strategy Layer)**: Reads the state register using `Acquire` ordering. This guarantees that when the strategy thread reads the updated status (e.g., `PositionStatus::Flat`), all memory modifications made by the execution thread prior to writing that status are synchronized and visible.

```rust
pub struct RealTimeStateBroker {
    position_status: AtomicU8,
}

impl RealTimeStateBroker {
    pub fn new() -> Self {
        Self {
            position_status: AtomicU8::new(PositionStatus::Flat as u8),
        }
    }

    #[inline(always)]
    pub fn update_status(&self, status: PositionStatus) {
        // Release ordering commits all preceding portfolio writes before updating status
        self.position_status.store(status as u8, Ordering::Release);
    }

    #[inline(always)]
    pub fn get_status(&self) -> PositionStatus {
        // Acquire ordering guarantees visibility of all writes made prior to the store
        PositionStatus::from(self.position_status.load(Ordering::Acquire))
    }
}
```

### 2.3 SPSC Ring Buffer for Execution Reports

We broadcast detailed transaction fills from the execution loop using a lock-free, zero-copy `spsc` ring buffer (e.g., utilizing the `ringbuf` or `crossbeam-channel` crates).

```rust
pub struct ExecutionReport {
    pub timestamp_ns: u64,
    pub order_id: u64,
    pub fill_price: f64,
    pub fill_qty: f64,
    pub net_pnl: f64,
    pub status: PositionStatus,
}
```

The execution loop acts as the sole **Producer**, pushing `ExecutionReport` events on fill occurrences. The strategy layer acts as the sole **Consumer**, draining this channel at the boundary gatekeeper.

---

## Phase 3: Implementing Boundary Gatekeepers / Barriers

Before the Macro Strategy Layer is permitted to evaluate signals on `on_bar_close()`, it must pass through a synchronization barrier. The barrier's job is to forcefully drain the execution loop's outbound report queue and wait for any pending execution states to reconcile.

### 3.1 Gatekeeper Interface & Signatures

```rust
pub trait BoundaryGatekeeper {
    /// Reconciles state up to a specific boundary timestamp.
    /// Blocks until all ticks before `boundary_ts_ns` have been processed,
    /// and all pending exit orders have been resolved (filled/cancelled).
    fn reconcile_boundary_state(&self, boundary_ts_ns: u64) -> Result<(), GatekeeperError>;
}

#[derive(Debug)]
pub enum GatekeeperError {
    Timeout,
    QueueCorrupted,
}
```

### 3.2 Gatekeeper Implementation

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub struct BarBoundaryGatekeeper {
    // Monotonically increasing sequence or timestamp of the latest processed tick in the micro engine
    latest_processed_tick_ns: Arc<AtomicU64>,
    state_broker: Arc<RealTimeStateBroker>,
    // Consumer end of the SPSC queue for execution reports
    report_consumer: ringbuf::Consumer<ExecutionReport>,
}

impl BoundaryGatekeeper for BarBoundaryGatekeeper {
    fn reconcile_boundary_state(&self, boundary_ts_ns: u64) -> Result<(), GatekeeperError> {
        let timeout_limit = Duration::from_millis(50); // 50ms safety timeout
        let start_time = std::time::Instant::now();

        // 1. Spin/wait until the Micro Engine has processed all ticks up to the boundary timestamp
        while self.latest_processed_tick_ns.load(Ordering::Acquire) < boundary_ts_ns {
            if start_time.elapsed() > timeout_limit {
                return Err(GatekeeperError::Timeout);
            }
            // Yield CPU to let the execution thread finish processing ticks
            thread::yield_now();
        }

        // 2. Drain all pending execution reports in the SPSC ring buffer
        // This updates the local portfolio cache inside the strategy memory space
        while let Some(report) = self.report_consumer.pop() {
            // Reconcile portfolio cache locally
            self.reconcile_portfolio_cache(&report);
        }

        // 3. Re-evaluate any pending order states.
        // If the state is PendingClose, we must block until the fill arrives
        while self.state_broker.get_status() == PositionStatus::PendingClose {
            if start_time.elapsed() > timeout_limit {
                return Err(GatekeeperError::Timeout);
            }
            
            // Drain SPSC queue again in case the fill arrives during spin
            while let Some(report) = self.report_consumer.pop() {
                self.reconcile_portfolio_cache(&report);
            }
            
            std::hint::spin_loop();
        }

        Ok(())
    }
}

impl BarBoundaryGatekeeper {
    fn reconcile_portfolio_cache(&self, report: &ExecutionReport) {
        // Zero-copy update of strategy's local copy of margins/balances
        // e.g., portfolio.cash += report.net_pnl;
        //       portfolio.position = report.status;
    }
}
```

---

## Phase 4: Regression Testing & State-Trace Debugging

To verify that the boundary gatekeeper completely eliminates state visibility lag, we implement a trace-logging test harness.

### 4.1 Structuring the Trace Log

Every key event in both the Execution Loop and the Strategy Layer is recorded into a shared lock-free memory ring buffer with nanosecond timestamps:

```rust
#[derive(Debug, Clone)]
pub enum TraceEventType {
    TickReceived { price: f64 },
    StopTriggered { order_id: u64, limit_price: f64 },
    FillReceived { order_id: u64, fill_price: f64 },
    StateBrokerUpdate { new_state: PositionStatus },
    BoundarySyncStart { boundary_ts_ns: u64 },
    BoundarySyncEnd { boundary_ts_ns: u64, final_state: PositionStatus },
    StrategySignalCheck { signal_type: &'static str },
}

pub struct TraceEvent {
    pub timestamp_ns: u64,
    pub thread_id: usize,
    pub event_type: TraceEventType,
}
```

### 4.2 Automated Trace Assertions

We write integration tests that simulate a stop-out happening $1\text{ ms}$ before the bar close. We then assert the following conditions on the collected event trace:

1. **Monotonicity**: The end of the boundary sync ($T_{sync\_end}$) must occur *before* the strategy checks for signals ($T_{signal\_check}$).
2. **State Concurrency Assert**: If a stop-out fill event ($T_{fill}$) occurs before the boundary sync end ($T_{sync\_end}$), then the final visible state checked by the strategy must be `Flat`.

```rust
#[test]
fn test_boundary_state_visibility_concurrency() {
    let trace: Vec<TraceEvent> = run_backtest_with_tracing();

    let mut exit_fill_time: Option<u64> = None;
    let mut sync_end_time: Option<u64> = None;
    let mut strategy_check_state: Option<PositionStatus> = None;

    for event in trace {
        match event.event_type {
            TraceEventType::FillReceived { .. } => {
                exit_fill_time = Some(event.timestamp_ns);
            }
            TraceEventType::BoundarySyncEnd { final_state, .. } => {
                sync_end_time = Some(event.timestamp_ns);
                strategy_check_state = Some(final_state);
            }
            TraceEventType::StrategySignalCheck { .. } => {
                assert!(sync_end_time.is_some(), "Strategy signal check ran before boundary gatekeeper reconciled!");
                if let Some(fill_t) = exit_fill_time {
                    assert_eq!(
                        strategy_check_state.unwrap(),
                        PositionStatus::Flat,
                        "State leak detected! Strategy evaluated indicators while position was still marked active."
                    );
                }
            }
            _ => {}
        }
    }
}
```
