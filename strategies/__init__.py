"""
Strategies package.

Houses all trading strategy implementations. Each strategy satisfies
StrategyProtocol and is a self-contained, stateless signal generator.
"""

from strategies.trend_oscillator import TrendOscillatorStrategy

__all__ = ["TrendOscillatorStrategy"]
