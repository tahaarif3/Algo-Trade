"""
Algo-Trade Backtest Engine
==========================

Modular, protocol-driven backtesting engine designed for extensibility.
The engine layer is intentionally decoupled from strategy logic and data sources
via Protocol classes, enabling future replacement of any component (e.g.,
swapping the backtest engine for a vectorized variant, or wiring data
sources through an MCP server).

Public API:
    - BacktestEngine:   Event-driven backtest runner
    - BinanceDataLoader: Free public API candle fetcher
    - MetricsCalculator: Performance analytics from trade logs
    - StrategyConfig:    Dataclass for all tunable parameters
    - BacktestResult:    Structured output from a backtest run
    - Trade:             Single trade record
"""

from engine.config import StrategyConfig, Trade, BacktestResult
from engine.backtest import BacktestEngine
from engine.data_loader import BinanceDataLoader
from engine.metrics import MetricsCalculator
from engine.protocols import (
    DataSourceProtocol,
    BacktestEngineProtocol,
    StrategyProtocol,
)
from engine.monte_carlo import MonteCarloRunner, MonteCarloResult
from engine.walk_forward import WalkForwardAnalyzer, WFAResult
from engine.hft_bar_wrapper import BarToHFTWrapper

__all__ = [
    "StrategyConfig",
    "Trade",
    "BacktestResult",
    "BacktestEngine",
    "BinanceDataLoader",
    "MetricsCalculator",
    "DataSourceProtocol",
    "BacktestEngineProtocol",
    "StrategyProtocol",
    "MonteCarloRunner",
    "MonteCarloResult",
    "WalkForwardAnalyzer",
    "WFAResult",
    "BarToHFTWrapper",
]
