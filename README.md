# Algo-Trade: Multi-Tier Quantitative Backtesting & HFT Simulation Framework

Algo-Trade is a high-performance quantitative backtesting pipeline designed for research and execution verification of systematic trading strategies. The framework operates on two distinct tiers: a macro-level swing-trading optimizer and a microsecond-level discrete event simulator.

---

## 1. System Architecture

The project features a **two-tier architecture** that separates strategy prototyping from latency-sensitive execution modeling:

1. **Swing-Trading Tier ([BacktestEngine](file:///c:/Users/bigbo/Algo-Trade/engine/backtest.py))**:
   * Event-driven bar-by-bar backtesting.
   * Dynamic stop-loss (SL) and take-profit (TP) calculation based on ATR volatility.
   * Auto-tuning loop executing baseline runs, parameter optimization, momentum trigger swaps, and drawdown-capped capital sizing.
2. **High-Frequency Tier ([DiscreteEventSimulator](file:///c:/Users/bigbo/Algo-Trade/engine/des_engine.py))**:
   * Nanosecond-resolution discrete event simulator (DES) backed by a priority queue heap.
   * In-process network delay and execution slippage bridge.
   * Double-ended queue Python Limit Order Book correctness reference ([LimitOrderBook](file:///c:/Users/bigbo/Algo-Trade/engine/lob_python.py)).
   * Native Rust Limit Order Book extension ([hft_engine](file:///c:/Users/bigbo/Algo-Trade/hft_engine/src/lib.rs)) backed by a doubly-linked list `SlotMap` arena to achieve O(1) cancels.

For a detailed breakdown of the engine internals, see [ARCHITECTURE.md](file:///c:/Users/bigbo/Algo-Trade/ARCHITECTURE.md).

---

## 2. Directory Structure

```bash
├── hft_engine/          # Rust native PyO3 core LOB library
│   ├── src/
│   │   ├── lib.rs       # PyO3 bindings & wrapper interfaces
│   │   ├── order_book.rs# O(1) doubly-linked list & SlotMap matcher
│   │   └── types.rs     # Fixed-point raw representations & Side enums
│   └── Cargo.toml
├── engine/              # Python orchestration & reference models
│   ├── backtest.py      # Swing-trading bar backtest engine
│   ├── des_engine.py    # Discrete Event Simulator loop
│   ├── lob_python.py    # Python LOB reference implementation
│   ├── tick_parser.py   # PyArrow record-batch tick data parser
│   ├── hft_data_loader.py# Binance Vision Snappy Parquet data streams
│   ├── latency_bridge.py # Network delay & execution slippage modeling
│   ├── metrics.py       # Performance & risk metric calculations
│   ├── data_loader.py   # REST/Local kline data management
│   ├── walk_forward.py  # Walk-forward optimization logic
│   ├── monte_carlo.py   # Strategy robustness simulation
│   ├── cross_validate.py# Parameter cross-validation
│   ├── events.py        # Event type definitions for DES
│   └── protocols.py     # Structural typing protocol definitions
├── strategies/          # Trading strategies
│   ├── trend_oscillator.py# Trend following + Oscillator entry
│   └── hft_market_maker.py# High-frequency bid/ask quoting strategy
├── tests/               # Test suite
│   ├── test_des_engine.py    # Priority queue & order matching unit tests
│   ├── test_des_vs_bar_engine.py# Tick-to-candle PnL equivalence validation
│   ├── test_rust_lob.py      # 10k random-ops differential LOB checks
│   ├── test_zmq_bridge.py    # Slippage and latency bridge verification
│   └── test_e2e_hft.py       # End-to-end Python/Rust LOB MM simulation
├── scratch/             # Development scripts and temporary debug tools
├── requirements.txt     # Python runtime dependencies
├── runner.py            # Swing-trade optimization loop entrypoint
└── ARCHITECTURE.md      # Detailed system design documentation
```

---

## 3. Installation & Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.14)
* Rust toolchain (stable-x86_64-pc-windows-msvc)
* Git

### Installation Steps
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Compile and link the native Rust order book:
   ```bash
   cd hft_engine
   maturin develop
   cd ..
   ```

---

## 4. Usage

### Running the Swing-Trading Optimizer
To execute the autonomous 4-run parameter optimization research pipeline, fetch daily/hourly historical klines, and output the strategy research report:
```bash
python runner.py
```
This writes the final output to `strategy_research_report.md` and exports the winning configuration to `strategies/winning_config.json`.

### Running the Test Suite
To verify both simulation engines, slippage models, LOB compliance, and regression safety:
```bash
python -m pytest
```

---

## 5. Technology Stack & Specs
* **Execution Engines**: Python 3.14 + `pandas` + `ta`
* **Tick Core**: Rust (`pyo3` / `maturin` bindings)
* **Serialization**: `msgpack` / `pyarrow` (Snappy compression)
* **Data Sources**: Public Binance Kline REST API & Binance Vision Daily Trade Archive
* **Floating-Point Guard**: Rust fixed-point integer scaling (Prices scaled by 100, quantities scaled by 100,000)
