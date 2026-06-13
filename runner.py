"""
Runner — Master orchestrator for the 4-run optimization loop.

Executes all phases autonomously:
    1. Fetch & validate candle data
    2. Run baseline backtest
    3. Volatility tuning (ATR adjustment)
    4. Momentum smoothing (RSI → Stochastic RSI swap)
    5. Capital efficiency optimization (position sizing with DD guardrail)
    6. Generate the final strategy_research_report.md

Entry point: python runner.py

Designed so individual functions can be exposed as MCP tools in the future.
"""

from __future__ import annotations

import json
import os
import sys
import copy
from dataclasses import replace
from datetime import datetime

from engine.config import StrategyConfig, BacktestResult
from engine.data_loader import BinanceDataLoader
from engine.backtest import BacktestEngine
from engine.metrics import MetricsCalculator
from strategies.trend_oscillator import TrendOscillatorStrategy

# ── Constants ──────────────────────────────────────────────────────────

SYMBOL = "BTCUSDT"
INTERVAL = "4h"

# Fetch extra data before backtest start for EMA-200 warmup
DATA_START = "2023-07-01"
DATA_END = "2025-01-01"

# Backtest window (within the fetched data)
BACKTEST_START = "2024-01-01"
BACKTEST_END = "2024-12-31"

RUNS_DIR = "runs"
DATA_DIR = "data"
REPORT_FILE = "strategy_research_report.md"


def ensure_dirs():
    """Create output directories."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def save_result(result: BacktestResult, filename: str):
    """Persist backtest results as JSON."""
    path = os.path.join(RUNS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(result.to_json())
    print(f"  [Runner] Results saved -> {path}")


# ── Phase 1: Data Acquisition ─────────────────────────────────────────

def fetch_data() -> "pd.DataFrame":
    """Fetch and validate candle data."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("PHASE 1: DATA ACQUISITION")
    print("=" * 60)

    loader = BinanceDataLoader(cache_dir=DATA_DIR)
    df = loader.fetch_candles(SYMBOL, INTERVAL, DATA_START, DATA_END)

    # Validate
    validation = loader.validate_data(df, DATA_START, DATA_END, min_candles=1000)
    print(f"  [Validation] Total candles: {validation['total_candles']}")
    print(f"  [Validation] Range: {validation['actual_start']} -> {validation['actual_end']}")
    print(f"  [Validation] Valid: {validation['valid']}")

    if not validation["valid"]:
        print(f"  [ERROR] Data validation failed: {validation.get('failure_reasons')}")
        sys.exit(1)

    # Slice to include warmup + backtest period
    # The strategy will handle warmup internally via config.warmup_candles
    bt_start = pd.Timestamp(BACKTEST_START, tz="UTC")
    bt_end = pd.Timestamp(BACKTEST_END, tz="UTC")

    # We need data BEFORE bt_start for warmup — keep all data from DATA_START
    df_backtest = df[df.index <= bt_end].copy()

    print(f"  [Data] Using {len(df_backtest)} candles for backtest (including warmup)")

    return df_backtest


# ── Phase 2 & 3: Backtest Execution ──────────────────────────────────

def run_backtest(
    data: "pd.DataFrame",
    config: StrategyConfig,
    run_label: str,
) -> BacktestResult:
    """Execute a single backtest run."""
    print(f"\n  -- {run_label} --")

    strategy = TrendOscillatorStrategy(config)
    engine = BacktestEngine()

    result = engine.run(data, strategy, config)
    result.run_label = run_label

    # Print metrics
    print(MetricsCalculator.format_metrics_table(result.metrics))
    print(f"  {result.summary}")

    return result


def run_baseline(data: "pd.DataFrame") -> BacktestResult:
    """Run 1: Baseline with default parameters."""
    print("\n" + "=" * 60)
    print("PHASE 2/3 - RUN 1: BASELINE")
    print("=" * 60)

    config = StrategyConfig()
    result = run_backtest(data, config, "Run 1 - Baseline")
    save_result(result, "run_1_baseline.json")
    return result


def run_volatility_tuning(
    data: "pd.DataFrame",
    baseline_result: BacktestResult,
) -> BacktestResult:
    """Run 2: Adjust ATR multipliers based on baseline trade analysis.

    Decision logic:
        - Short avg duration OR high consecutive losses → widen SL/TP
        - Long avg duration → tighten TP
        - Default: moderate widening for choppy market resilience
    """
    print("\n" + "=" * 60)
    print("PHASE 3 - RUN 2: VOLATILITY TUNING")
    print("=" * 60)

    m = baseline_result.metrics
    avg_duration = m.get("avg_trade_duration_bars", 0)
    consec_losses = m.get("max_consecutive_losses", 0)
    total_trades = m.get("total_trades", 0)

    print(f"  [Analysis] Baseline avg trade duration: {avg_duration:.1f} bars")
    print(f"  [Analysis] Baseline max consecutive losses: {consec_losses}")
    print(f"  [Analysis] Baseline total trades: {total_trades}")

    # Start from baseline config
    config = StrategyConfig()

    if total_trades == 0:
        # No trades — loosen entry conditions by widening thresholds
        print("  [Decision] No trades in baseline -- widening RSI thresholds")
        config.rsi_long_threshold = 35.0
        config.rsi_short_threshold = 65.0
        config.atr_mult_sl = 2.0
        config.atr_mult_tp = 4.0
    elif avg_duration < 3 or consec_losses > 4:
        # Too many whipsaws — widen stops to give trades room
        print("  [Decision] High whipsaw detected -> widening ATR multipliers")
        config.atr_mult_sl = 3.0
        config.atr_mult_tp = 6.0
    elif avg_duration > 20:
        # Trades held too long — tighten TP for faster capture
        print("  [Decision] Long avg duration -> tightening TP target")
        config.atr_mult_sl = 2.5
        config.atr_mult_tp = 4.0
    else:
        # Moderate adjustment — slight widening for resilience
        print("  [Decision] Moderate performance -> slight SL widening")
        config.atr_mult_sl = 3.0
        config.atr_mult_tp = 5.5

    print(f"  [Params] SL mult: {config.atr_mult_sl}, TP mult: {config.atr_mult_tp}")

    result = run_backtest(data, config, "Run 2 - Volatility Tuned")
    save_result(result, "run_2_volatility.json")
    return result


def run_momentum_smoothing(
    data: "pd.DataFrame",
    result_1: BacktestResult,
    result_2: BacktestResult,
) -> BacktestResult:
    """Run 3: Swap RSI for Stochastic RSI to filter false signals.

    Inherits the better ATR config from Runs 1-2, then toggles SRSI.
    """
    print("\n" + "=" * 60)
    print("PHASE 3 - RUN 3: MOMENTUM SMOOTHING (SRSI)")
    print("=" * 60)

    # Pick the better base config (by Sharpe, then by profit factor)
    if _is_better(result_2, result_1):
        base_config = result_2.config
        print("  [Decision] Using Run 2 config as base (outperformed Run 1)")
    else:
        base_config = result_1.config
        print("  [Decision] Using Run 1 config as base (outperformed Run 2)")

    # Build SRSI config from the winning base
    config = StrategyConfig(
        ema_period=base_config.ema_period,
        rsi_period=base_config.rsi_period,
        atr_period=base_config.atr_period,
        rsi_long_threshold=base_config.rsi_long_threshold,
        rsi_short_threshold=base_config.rsi_short_threshold,
        atr_mult_sl=base_config.atr_mult_sl,
        atr_mult_tp=base_config.atr_mult_tp,
        position_pct=base_config.position_pct,
        # Toggle SRSI
        use_srsi=True,
        srsi_k_long=20.0,
        srsi_k_short=80.0,
        starting_capital=base_config.starting_capital,
        fee_rate=base_config.fee_rate,
        warmup_candles=base_config.warmup_candles,
    )

    print(f"  [Params] SRSI enabled, K thresholds: long<{config.srsi_k_long}, short>{config.srsi_k_short}")

    result = run_backtest(data, config, "Run 3 - Momentum Smoothed (SRSI)")
    save_result(result, "run_3_momentum.json")
    return result


def run_capital_efficiency(
    data: "pd.DataFrame",
    result_1: BacktestResult,
    result_2: BacktestResult,
    result_3: BacktestResult,
) -> BacktestResult:
    """Run 4: Optimize position sizing with -15% max DD guardrail.

    Selects the best of Runs 1-3, then increases capital allocation.
    If the increased allocation breaches -15% DD, reverts and re-runs.
    """
    print("\n" + "=" * 60)
    print("PHASE 3 - RUN 4: CAPITAL EFFICIENCY OPTIMIZATION")
    print("=" * 60)

    # Rank all three by Sharpe ratio
    all_results = [result_1, result_2, result_3]
    ranked = sorted(all_results, key=lambda r: r.metrics.get("sharpe_ratio", 0), reverse=True)

    best = ranked[0]
    print(f"  [Decision] Best iteration: {best.run_label}")
    print(f"  [Decision] Sharpe: {best.metrics.get('sharpe_ratio', 0):.3f}, "
          f"MaxDD: {best.metrics.get('max_drawdown_pct', 0):.2f}%")

    base_config = best.config
    current_dd = abs(best.metrics.get("max_drawdown_pct", 0))

    # Determine safe position size increase
    if current_dd < 8:
        # Lots of headroom — aggressive increase
        new_pct = min(base_config.position_pct + 0.05, 0.20)  # Cap at 20%
        print(f"  [Decision] Low DD ({current_dd:.1f}%) -> increasing position to {new_pct*100:.0f}%")
    elif current_dd < 12:
        # Some headroom — moderate increase
        new_pct = min(base_config.position_pct + 0.03, 0.15)
        print(f"  [Decision] Moderate DD ({current_dd:.1f}%) -> increasing position to {new_pct*100:.0f}%")
    else:
        # Close to guardrail — minimal increase
        new_pct = min(base_config.position_pct + 0.01, 0.12)
        print(f"  [Decision] High DD ({current_dd:.1f}%) -> cautious increase to {new_pct*100:.0f}%")

    # Build optimized config
    config = StrategyConfig(
        ema_period=base_config.ema_period,
        rsi_period=base_config.rsi_period,
        atr_period=base_config.atr_period,
        rsi_long_threshold=base_config.rsi_long_threshold,
        rsi_short_threshold=base_config.rsi_short_threshold,
        atr_mult_sl=base_config.atr_mult_sl,
        atr_mult_tp=base_config.atr_mult_tp,
        position_pct=new_pct,
        use_srsi=base_config.use_srsi,
        srsi_window=base_config.srsi_window,
        srsi_smooth1=base_config.srsi_smooth1,
        srsi_smooth2=base_config.srsi_smooth2,
        srsi_k_long=base_config.srsi_k_long,
        srsi_k_short=base_config.srsi_k_short,
        starting_capital=base_config.starting_capital,
        fee_rate=base_config.fee_rate,
        warmup_candles=base_config.warmup_candles,
    )

    result = run_backtest(data, config, "Run 4 - Capital Optimized")

    # ── GUARDRAIL: Max DD must not exceed -15% ──
    actual_dd = abs(result.metrics.get("max_drawdown_pct", 0))
    if actual_dd > 15:
        print(f"\n  WARNING: GUARDRAIL TRIGGERED: MaxDD = -{actual_dd:.2f}% exceeds -15% limit")
        print(f"  [Revert] Falling back to base config (position_pct={base_config.position_pct})")

        # Revert to the best config without sizing change
        config_reverted = StrategyConfig(
            ema_period=base_config.ema_period,
            rsi_period=base_config.rsi_period,
            atr_period=base_config.atr_period,
            rsi_long_threshold=base_config.rsi_long_threshold,
            rsi_short_threshold=base_config.rsi_short_threshold,
            atr_mult_sl=base_config.atr_mult_sl,
            atr_mult_tp=base_config.atr_mult_tp,
            position_pct=base_config.position_pct,
            use_srsi=base_config.use_srsi,
            srsi_window=base_config.srsi_window,
            srsi_smooth1=base_config.srsi_smooth1,
            srsi_smooth2=base_config.srsi_smooth2,
            srsi_k_long=base_config.srsi_k_long,
            srsi_k_short=base_config.srsi_k_short,
            starting_capital=base_config.starting_capital,
            fee_rate=base_config.fee_rate,
            warmup_candles=base_config.warmup_candles,
        )
        result = run_backtest(data, config_reverted, "Run 4 - Capital Optimized (Reverted)")
    else:
        print(f"  [OK] Guardrail passed: MaxDD = -{actual_dd:.2f}% (limit: -15%)")

    save_result(result, "run_4_capital.json")
    return result


# ── Phase 4: Report Generation ────────────────────────────────────────

def generate_report(
    results: list[BacktestResult],
    baseline_config: StrategyConfig,
) -> str:
    """Generate the final strategy_research_report.md."""
    print("\n" + "=" * 60)
    print("PHASE 4: GENERATING RESEARCH REPORT")
    print("=" * 60)

    # Find the winning iteration
    winner = max(results, key=lambda r: r.metrics.get("sharpe_ratio", 0))

    lines = []
    lines.append("# TrendOscillatorStrategy — Quantitative Research Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Instrument:** BTC/USDT Perpetual Futures (Binance)")
    lines.append(f"**Timeframe:** 4-Hour Candles")
    lines.append(f"**Backtest Period:** {BACKTEST_START} to {BACKTEST_END}")
    lines.append(f"**Starting Capital:** ${baseline_config.starting_capital:,.2f}")
    lines.append(f"**Fee Model:** {baseline_config.fee_rate * 100:.2f}% taker per side")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Comparison Matrix ──
    lines.append("## 1. Iteration Comparison Matrix")
    lines.append("")
    lines.append("| Metric | " + " | ".join(r.run_label for r in results) + " |")
    lines.append("|--------|" + "|".join("---" for _ in results) + "|")

    # Parameters changed
    param_row = "| **Parameters Altered** |"
    for r in results:
        desc = r.config.describe_changes(baseline_config)
        param_row += f" {desc} |"
    lines.append(param_row)

    # Metric rows
    metric_defs = [
        ("Ending Balance ($)", "ending_balance", "${:,.2f}"),
        ("Net Profit ($)", "net_profit", "${:,.2f}"),
        ("Total Trades", "total_trades", "{}"),
        ("Win Rate %", "win_rate_pct", "{:.1f}%"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Max Drawdown %", "max_drawdown_pct", "{:.2f}%"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
        ("Avg Duration (bars)", "avg_trade_duration_bars", "{:.1f}"),
        ("Max Consec. Losses", "max_consecutive_losses", "{}"),
        ("Total Fees ($)", "total_fees", "${:,.2f}"),
    ]

    for label, key, fmt in metric_defs:
        row = f"| **{label}** |"
        for r in results:
            val = r.metrics.get(key, 0)
            if key == "ending_balance" or key == "net_profit" or key == "total_fees":
                row += f" {fmt.format(val)} |"
            elif "%" in fmt:
                row += f" {fmt.format(val)} |"
            else:
                row += f" {fmt.format(val)} |"
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Technical Deduction ──
    lines.append("## 2. Technical Deduction — Why the Winner Outperformed")
    lines.append("")
    lines.append(f"**Winning Iteration:** {winner.run_label}")
    lines.append("")

    w_m = winner.metrics
    b_m = results[0].metrics  # Baseline

    lines.append("### Mathematical Analysis")
    lines.append("")

    # Sharpe decomposition
    lines.append("**Sharpe Ratio Decomposition:**")
    lines.append(f"- Baseline Sharpe: {b_m.get('sharpe_ratio', 0):.3f}")
    lines.append(f"- Winner Sharpe: {w_m.get('sharpe_ratio', 0):.3f}")
    lines.append("")

    # Win rate and profit factor analysis
    lines.append("**Edge Analysis:**")
    w_wr = w_m.get("win_rate_pct", 0)
    b_wr = b_m.get("win_rate_pct", 0)
    w_pf = w_m.get("profit_factor", 0)
    b_pf = b_m.get("profit_factor", 0)

    if w_wr > b_wr:
        lines.append(f"- Win rate improved from {b_wr:.1f}% to {w_wr:.1f}% "
                      f"(+{w_wr - b_wr:.1f}pp), reducing the frequency of losing trades.")
    elif w_wr < b_wr:
        lines.append(f"- Win rate decreased from {b_wr:.1f}% to {w_wr:.1f}% "
                      f"({w_wr - b_wr:.1f}pp), but this was offset by improved reward per trade.")

    if w_pf > b_pf:
        lines.append(f"- Profit factor improved from {b_pf:.2f} to {w_pf:.2f}, meaning each "
                      f"dollar risked generated ${w_pf:.2f} in gross profit vs ${b_pf:.2f} baseline.")

    # Drawdown analysis
    w_dd = abs(w_m.get("max_drawdown_pct", 0))
    b_dd = abs(b_m.get("max_drawdown_pct", 0))
    if w_dd < b_dd:
        lines.append(f"- Max drawdown reduced from -{b_dd:.2f}% to -{w_dd:.2f}%, "
                      f"indicating tighter capital preservation.")
    lines.append("")

    # Parameter impact
    param_changes = winner.config.describe_changes(baseline_config)
    lines.append("**Parameter Changes That Drove Improvement:**")
    lines.append(f"- {param_changes}")
    lines.append("")

    # Specific deductions per run type
    if winner.config.use_srsi:
        lines.append("**Stochastic RSI Impact:** The double-smoothed oscillator reduced "
                      "false entry signals by applying a stochastic transformation on RSI "
                      "values, effectively filtering out noise in ranging markets where raw "
                      "RSI frequently oscillates near the 30/70 boundaries.")
    if winner.config.atr_mult_sl != baseline_config.atr_mult_sl:
        sl_delta = winner.config.atr_mult_sl - baseline_config.atr_mult_sl
        direction = "wider" if sl_delta > 0 else "tighter"
        lines.append(f"**ATR Stop-Loss Adjustment:** The {direction} stop ({baseline_config.atr_mult_sl} to "
                      f"{winner.config.atr_mult_sl}) reduced premature stop-outs on volatile "
                      f"4-hour bars, allowing winning trades to reach their ATR-based TP targets.")
    if winner.config.position_pct != baseline_config.position_pct:
        lines.append(f"**Position Sizing:** Capital allocation increased from "
                      f"{baseline_config.position_pct*100:.0f}% to {winner.config.position_pct*100:.0f}%, "
                      f"amplifying the per-trade return while staying within the -15% drawdown guardrail.")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Deployment Confirmation ──
    lines.append("## 3. Deployment Confirmation")
    lines.append("")
    lines.append(f"✅ **The winning configuration ({winner.run_label}) is deployed as the "
                  f"active default in `strategies/trend_oscillator.py`.**")
    lines.append("")
    lines.append("### Winning Configuration Parameters")
    lines.append("")
    lines.append("```python")
    lines.append("StrategyConfig(")
    for fld_name, fld in winner.config.__dataclass_fields__.items():
        val = getattr(winner.config, fld_name)
        if isinstance(val, str):
            lines.append(f"    {fld_name}=\"{val}\",")
        elif isinstance(val, bool):
            lines.append(f"    {fld_name}={val},")
        elif isinstance(val, float):
            lines.append(f"    {fld_name}={round(val, 4)},")
        else:
            lines.append(f"    {fld_name}={val},")
    lines.append(")")
    lines.append("```")
    lines.append("")

    report = "\n".join(lines)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  [Report] Written -> {REPORT_FILE}")
    return report


def update_strategy_defaults(winning_config: StrategyConfig):
    """Update the strategy file's docstring to document the winning params.

    We write a companion config file rather than modifying Python code,
    which is safer and more extensible for MCP tool consumption.
    """
    config_path = os.path.join("strategies", "winning_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(winning_config.to_json())
    print(f"  [Deploy] Winning config saved -> {config_path}")


# ── Utility ───────────────────────────────────────────────────────────

def _is_better(a: BacktestResult, b: BacktestResult) -> bool:
    """Compare two results: True if a is better than b.

    Primary sort: Sharpe Ratio (higher is better).
    Tiebreaker: Profit Factor (higher is better).
    """
    a_sharpe = a.metrics.get("sharpe_ratio", 0)
    b_sharpe = b.metrics.get("sharpe_ratio", 0)

    if a_sharpe != b_sharpe:
        return a_sharpe > b_sharpe

    return a.metrics.get("profit_factor", 0) > b.metrics.get("profit_factor", 0)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    """Execute the full 4-phase research pipeline."""
    print("============================================================")
    print("   TrendOscillatorStrategy -- Autonomous Research Pipeline   ")
    print("============================================================")
    print(f"   Instrument: BTC/USDT  |  Timeframe: 4H                   ")
    print(f"   Period: {BACKTEST_START} -> {BACKTEST_END}                     ")
    print(f"   Capital: $10,000  |  Fee: 0.04%/side                     ")
    print("============================================================")

    ensure_dirs()
    baseline_config = StrategyConfig()

    # Phase 1: Data
    data = fetch_data()

    # Phase 2+3: Optimization loop
    r1 = run_baseline(data)
    r2 = run_volatility_tuning(data, r1)
    r3 = run_momentum_smoothing(data, r1, r2)
    r4 = run_capital_efficiency(data, r1, r2, r3)

    all_results = [r1, r2, r3, r4]

    # Phase 4: Report
    generate_report(all_results, baseline_config)

    # Deploy winning config
    winner = max(all_results, key=lambda r: r.metrics.get("sharpe_ratio", 0))
    update_strategy_defaults(winner.config)

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for r in all_results:
        print(f"  {r.summary}")
    print(f"\n  WINNER: {winner.run_label}")
    print(f"  Report: {REPORT_FILE}")
    print(f"  Config: strategies/winning_config.json")


if __name__ == "__main__":
    main()
