from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd
import numpy as np

from engine.config import StrategyConfig, BacktestResult, Trade
from engine.backtest import BacktestEngine
from engine.metrics import MetricsCalculator

@dataclass
class WFAFold:
    """Represents the results and metadata for a single train/test fold."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_result: BacktestResult
    test_result: BacktestResult
    robustness_score: float


@dataclass
class WFAResult:
    """Aggregated results across all walk-forward folds."""
    n_folds: int
    folds: list[WFAFold]
    oos_equity_curve: list[dict]
    robustness_scores: list[float]
    aggregate_metrics: dict
    summary: str


class WalkForwardAnalyzer:
    """Performs chronological walk-forward analysis (WFA) splits, runs backtests, and stitches OOS equity."""
    
    def __init__(self, n_folds: int = 5, train_pct: float = 0.70) -> None:
        self.n_folds = n_folds
        self.train_pct = train_pct

    def split(self, df: pd.DataFrame, warmup_candles: int) -> list[dict]:
        """Divide DataFrame into Train/Test slice indices.
        
        Uses anchored train windows (starts at 0, expands) followed by non-overlapping,
        adjacent chronological test windows.
        """
        n_candles = len(df)
        test_pct = 1.0 - self.train_pct
        test_segment_size = int(n_candles * test_pct / self.n_folds)
        
        if test_segment_size <= 0:
            raise ValueError(
                f"Dataset length ({n_candles}) is too small for {self.n_folds} folds "
                f"with {self.train_pct:.1%} train percentage."
            )
            
        folds = []
        for i in range(self.n_folds):
            train_start = 0
            train_end = int(n_candles * self.train_pct) + i * test_segment_size
            test_start = train_end
            test_end = test_start + test_segment_size
            
            # Ensure the final fold covers exactly up to the end of the dataset
            if i == self.n_folds - 1:
                test_end = n_candles
                
            train_len = train_end - train_start
            if train_len <= warmup_candles:
                raise ValueError(
                    f"Train window length ({train_len}) is smaller than or equal to warmup_candles ({warmup_candles}) "
                    f"for fold {i}. Increase dataset size or reduce warmup requirements."
                )
                
            if test_start < warmup_candles:
                raise ValueError(
                    f"Test start index ({test_start}) is less than warmup_candles ({warmup_candles}) "
                    f"for fold {i}. The test window cannot resolve warmup candles."
                )
                
            folds.append({
                "fold_idx": i,
                "train_start_idx": train_start,
                "train_end_idx": train_end,
                "test_start_idx": test_start,
                "test_end_idx": test_end,
            })
        return folds

    def run(self, df: pd.DataFrame, strategy_cls: Any, config: StrategyConfig) -> WFAResult:
        """Execute walk-forward simulation across all folds."""
        warmup = config.warmup_candles
        fold_specs = self.split(df, warmup)
        
        folds = []
        engine = BacktestEngine()
        
        for spec in fold_specs:
            idx = spec["fold_idx"]
            
            # Slice In-Sample (Train) split
            train_df = df.iloc[spec["train_start_idx"] : spec["train_end_idx"]]
            strategy_train = strategy_cls(config)
            train_res = engine.run(train_df, strategy_train, config)
            
            # Slice Out-of-Sample (Test) split with warmup prepended
            # This is critical so the BacktestEngine has warmup historical data
            # to compute indicators and starts trading exactly at test_start_idx
            test_df_with_warmup = df.iloc[spec["test_start_idx"] - warmup : spec["test_end_idx"]]
            strategy_test = strategy_cls(config)
            test_res = engine.run(test_df_with_warmup, strategy_test, config)
            
            # Compute robustness score (Out-of-Sample PnL / In-Sample PnL)
            train_pnl = train_res.metrics.get("net_profit", 0.0)
            test_pnl = test_res.metrics.get("net_profit", 0.0)
            
            if train_pnl == 0.0:
                robustness = float("nan")
            else:
                robustness = test_pnl / train_pnl
                
            folds.append(WFAFold(
                fold_idx=idx,
                train_start=df.index[spec["train_start_idx"]],
                train_end=df.index[spec["train_end_idx"] - 1],
                test_start=df.index[spec["test_start_idx"]],
                test_end=df.index[spec["test_end_idx"] - 1],
                train_result=train_res,
                test_result=test_res,
                robustness_score=robustness
            ))
            
        # Stitch Out-of-Sample equity curves and compile OOS trades
        stitched_equity_curve = []
        current_offset = 0.0
        all_oos_trades = []
        
        for fold in folds:
            test_curve = fold.test_result.equity_curve
            
            # Accumulate all trades executed in OOS period
            all_oos_trades.extend(fold.test_result.trades)
            
            # Offset the OOS equity curve to make it contiguous
            for point in test_curve:
                stitched_equity_curve.append({
                    "timestamp": point["timestamp"],
                    "equity": point["equity"] + current_offset
                })
                
            # Compound/offset increment for next fold
            if test_curve:
                current_offset += test_curve[-1]["equity"] - config.starting_capital
                
        # Compute aggregate metrics over the whole stitched OOS run
        aggregate_metrics = MetricsCalculator.compute_all(
            all_oos_trades,
            stitched_equity_curve,
            config.starting_capital
        )
        
        # Calculate mean robustness of non-nan scores
        valid_scores = [f.robustness_score for f in folds if not math.isnan(f.robustness_score)]
        mean_robustness = float(np.mean(valid_scores)) if valid_scores else 0.0
        
        summary = (
            f"Walk-Forward Analysis Completed ({self.n_folds} folds)\n"
            f"OOS Net Profit: ${aggregate_metrics.get('net_profit', 0.0):,.2f}\n"
            f"OOS Total Trades: {aggregate_metrics.get('total_trades', 0)}\n"
            f"OOS Sharpe Ratio: {aggregate_metrics.get('sharpe_ratio', 0.0):.3f}\n"
            f"Mean Robustness Score: {mean_robustness:.3f}"
        )
        
        return WFAResult(
            n_folds=self.n_folds,
            folds=folds,
            oos_equity_curve=stitched_equity_curve,
            robustness_scores=[f.robustness_score for f in folds],
            aggregate_metrics=aggregate_metrics,
            summary=summary
        )

    def plot_oos_curve(self, result: WFAResult, save_path: Optional[str] = None) -> None:
        """Plot the stitched Out-of-Sample equity curve with vertical lines at fold splits."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required to plot the OOS curve. Please install it using 'pip install matplotlib'.")
            return
            
        if not result.oos_equity_curve:
            print("No OOS equity curve data to plot.")
            return
            
        dates = [pd.to_datetime(p["timestamp"]) for p in result.oos_equity_curve]
        equities = [p["equity"] for p in result.oos_equity_curve]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equities, color="green", label="Stitched OOS Equity", linewidth=2)
        
        # Draw split boundary lines
        min_eq = min(equities)
        for fold in result.folds:
            boundary_date = pd.to_datetime(fold.test_start)
            plt.axvline(boundary_date, color="grey", linestyle="--", alpha=0.5)
            plt.text(boundary_date, min_eq, f"Fold {fold.fold_idx} OOS", 
                     rotation=90, verticalalignment='bottom', alpha=0.7)
                     
        plt.title("Walk-Forward Analysis — Stitched Out-of-Sample Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"OOS curve plot saved to {save_path}")
        else:
            plt.show()
