from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable, Iterator, Any, Optional
import concurrent.futures

from engine.config import HFTConfig, BacktestResult, Trade
from engine.des_engine import DiscreteEventSimulator
from engine.events import BaseEvent

@dataclass
class MonteCarloResult:
    """Carries the statistical output and distributions of a Monte Carlo simulation run."""
    n_runs: int
    net_profit_distribution: list[float]
    max_drawdown_distribution: list[float]
    total_trades_distribution: list[int]
    sharpe_distribution: list[float]
    summary: dict
    equity_curves: list[list[dict]]

    def to_dict(self) -> dict:
        """Serialize results to a dictionary."""
        return {
            "n_runs": self.n_runs,
            "net_profit_distribution": self.net_profit_distribution,
            "max_drawdown_distribution": self.max_drawdown_distribution,
            "total_trades_distribution": self.total_trades_distribution,
            "sharpe_distribution": self.sharpe_distribution,
            "summary": self.summary,
            "equity_curves": self.equity_curves,
        }


def _run_single_mc(
    hft_config_dict: dict,
    seed: int,
    event_factory: Callable[[], Iterator[BaseEvent]],
    strategy_factory: Callable[[], Any]
) -> dict:
    """Top-level helper function to run a single DES iteration in subprocesses.
    
    Required for process pickling in Windows environments.
    """
    config = HFTConfig.from_dict(hft_config_dict)
    config.mc_random_seed = seed
    
    events = event_factory()
    strategy = strategy_factory()
    
    des = DiscreteEventSimulator(config)
    # Redirect stdout to suppress print logs during parallel execution
    import sys
    import os
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = des.run(events, strategy)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
        
    return {
        "net_profit": result.metrics.get("net_profit", 0.0),
        "max_drawdown": result.metrics.get("max_drawdown_pct", 0.0),
        "total_trades": result.metrics.get("total_trades", 0),
        "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
        "equity_curve": result.equity_curve,
    }


class MonteCarloRunner:
    """Orchestrates parallel stochastic simulations with latency jitter and packet drop scenarios."""
    
    def __init__(self, hft_config: HFTConfig, n_runs: int = 100, n_workers: int = 4) -> None:
        self.hft_config = hft_config
        self.n_runs = n_runs
        self.n_workers = n_workers

    def run(
        self,
        event_factory: Callable[[], Iterator[BaseEvent]],
        strategy_factory: Callable[[], Any]
    ) -> MonteCarloResult:
        """Executes the simulation suite.
        
        Uses ProcessPoolExecutor for concurrent runs, falling back to sequential execution
        if n_workers is set to 1.
        """
        # Determine starting seed
        base_seed = self.hft_config.mc_random_seed if self.hft_config.mc_random_seed is not None else 1337
        seeds = [base_seed + i for i in range(self.n_runs)]
        
        config_dict = self.hft_config.to_dict()
        results = []
        
        if self.n_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(_run_single_mc, config_dict, seed, event_factory, strategy_factory)
                    for seed in seeds
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        raise RuntimeError(f"Monte Carlo simulation worker failed: {e}") from e
        else:
            # Sequential execution for debugging or clean pytest runs
            for seed in seeds:
                res = _run_single_mc(config_dict, seed, event_factory, strategy_factory)
                results.append(res)
                
        return self._aggregate(results)

    def _aggregate(self, results: list[dict]) -> MonteCarloResult:
        net_profits = [r["net_profit"] for r in results]
        max_drawdowns = [r["max_drawdown"] for r in results]
        total_trades = [r["total_trades"] for r in results]
        sharpes = [r["sharpe_ratio"] for r in results]
        equity_curves = [r["equity_curve"] for r in results]
        
        summary = {}
        for name, dist in [
            ("net_profit", net_profits),
            ("max_drawdown", max_drawdowns),
            ("total_trades", total_trades),
            ("sharpe_ratio", sharpes)
        ]:
            summary[name] = {
                "mean": float(np.mean(dist)),
                "median": float(np.median(dist)),
                "p5": float(np.percentile(dist, 5)),
                "p95": float(np.percentile(dist, 95)),
            }
            
        return MonteCarloResult(
            n_runs=self.n_runs,
            net_profit_distribution=net_profits,
            max_drawdown_distribution=max_drawdowns,
            total_trades_distribution=total_trades,
            sharpe_distribution=sharpes,
            summary=summary,
            equity_curves=equity_curves
        )

    def plot_fan_chart(self, result: MonteCarloResult, save_path: Optional[str] = None) -> None:
        """Plot a fan chart of the equity curves showing percentiles over time.
        
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required to plot the fan chart. Please install it using 'pip install matplotlib'.")
            return
            
        if not result.equity_curves:
            print("No equity curves available to plot.")
            return
            
        # Determine the length of the shortest equity curve to align curves
        min_len = min(len(curve) for curve in result.equity_curves)
        if min_len == 0:
            print("Equity curves are empty.")
            return
            
        # Extract equities as a numpy matrix of shape (n_runs, min_len)
        equities = np.array([
            [point["equity"] for point in curve[:min_len]]
            for curve in result.equity_curves
        ])
        
        # Compute percentiles at each step
        p5 = np.percentile(equities, 5, axis=0)
        p25 = np.percentile(equities, 25, axis=0)
        p50 = np.percentile(equities, 50, axis=0)
        p75 = np.percentile(equities, 75, axis=0)
        p95 = np.percentile(equities, 95, axis=0)
        
        x = np.arange(min_len)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, p50, color="blue", label="Median (P50)", linewidth=2)
        plt.fill_between(x, p25, p75, color="blue", alpha=0.3, label="Interquartile Range (P25-P75)")
        plt.fill_between(x, p5, p95, color="blue", alpha=0.1, label="90% Confidence Interval (P5-P95)")
        
        plt.title(f"Monte Carlo Equity Curves ({result.n_runs} runs)")
        plt.xlabel("Simulation Steps")
        plt.ylabel("Equity")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Fan chart saved to {save_path}")
        else:
            plt.show()
