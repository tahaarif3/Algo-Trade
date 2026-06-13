from __future__ import annotations

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from engine.des_engine import DiscreteEventSimulator
    from engine.events import SignalEvent

from engine.events import LatencyDelayEvent


class InProcessLatencyBridge:
    """Injects configurable nanosecond delay into the DES event queue.

    Slippage emerges organically from intervening events processed
    between signal generation and the delayed execution.
    """

    def __init__(
        self,
        des: DiscreteEventSimulator,
        latency_ns: int = 50_000,
        jitter_std: float = 0.0,
        rng: Optional["np.random.Generator"] = None,
    ) -> None:
        self.des = des
        self.latency_ns = latency_ns
        self.jitter_std = jitter_std
        self.rng = rng

    def send_order(self, signal: SignalEvent) -> None:
        """Wrap signal in a LatencyDelayEvent and schedule it in the future."""
        delay = self.latency_ns
        if self.jitter_std > 0.0 and self.rng is not None:
            noise = self.rng.normal(0, self.jitter_std)
            # Latency cannot be negative; clamp noise so total delay >= 0
            delay = max(0, int(self.latency_ns + noise))
            
        delayed = LatencyDelayEvent(
            timestamp_ns=signal.timestamp_ns + delay,
            inner_event=signal,
        )
        self.des.schedule(delayed)
