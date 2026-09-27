"""Async, Gymnasium-style environment for cocotb-controlled 2x2 mesh NoCs."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from cocotb_tb.noc_interface import NoCInterface
from cocotb_tb.traffic_driver import Flit, TrafficDriver
from rl_env.action_decoder import decode_action
from rl_env.config import NoCConfig
from rl_env.coverage_stub import CoverageCollector
from rl_env.state_builder import StateBuilder


class AsyncNoCEnv:
    """Cocotb-safe environment whose reset and step methods always await time.

    The public names and `action_space`/`observation_space` follow Gymnasium,
    but this is deliberately not a synchronous Gymnasium `Env`: cocotb owns the
    simulator's clock. `step` returns `(state, reward, done, info)`, where a
    positive reward is the fraction of newly covered traffic bins.
    """

    def __init__(self, dut: Any, config: NoCConfig | None = None, interface: NoCInterface | None = None) -> None:
        self.config = config or NoCConfig()
        self.interface = interface or NoCInterface(dut)
        self.traffic_driver = TrafficDriver(self.interface)
        self.state_builder = StateBuilder(self.interface, self.config)
        self.coverage = CoverageCollector()
        self.action_space = self.config.action_space
        self.observation_space = self.config.observation_space
        self._cycles = 0
        self._injected_packets = 0
        self._transferred_flits = 0
        self._stalled_cycles = 0
        self._completed_packets = 0
        self._total_latency = 0
        self._done = False

    def reset_coverage(self) -> None:
        """Explicitly begin a new coverage campaign across one or more episodes."""
        self.coverage.reset()
    

    async def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset DUT and traffic state, then return `(state, info)` after clock edges."""
        del seed, options  # Reserved for Gymnasium compatibility.
        self._cycles = self._injected_packets = self._transferred_flits = 0
        self._stalled_cycles = self._completed_packets = self._total_latency = 0
        self._done = False
        self.traffic_driver.reset()
        await self.interface.reset(self.config.reset_cycles)
        state = self._build_state()
        return state, {"coverage": self.coverage.snapshot()}

    async def step(self, action: int | Sequence[int] | np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Drive one traffic action, await a rising edge, and return async RL data."""
        if self._done:
            raise RuntimeError("Episode is done; call and await reset() before another step().")

        params = decode_action(action)
        accepted_injection = self.traffic_driver.submit_action(params, self._cycles)
        if accepted_injection:
            self._injected_packets += 1
        driven_flit = self.traffic_driver.drive_current_flit()
        await self.interface.rising_edge()
        transferred = self.traffic_driver.complete_cycle()
        if driven_flit is not None and transferred is None:
            self._stalled_cycles += 1
        if transferred is not None:
            self._record_transfer(transferred)

        self._cycles += 1
        self._done = self._cycles >= self.config.max_steps
        state = self._build_state()
        coverage = self.coverage.collect(
            action=params,
            flit=transferred,
            accepted_injection=accepted_injection,
            transferred=transferred is not None,
        )
        # Coverage gain is bounded to [0, 1], making it safe for Q-learning and DQN.
        reward = coverage["newly_covered_count"] / coverage["total_bins"]
        info = {
            "action": params,
            "accepted_injection": accepted_injection,
            "transferred": transferred is not None,
            "cycles": self._cycles,
            "coverage": coverage,
        }
        return state, float(reward), self._done, info

    def _record_transfer(self, flit: Flit) -> None:
        self._transferred_flits += 1
        if flit.eop:
            self._completed_packets += 1
            self._total_latency += self._cycles - flit.created_cycle + 1

    def _performance(self) -> dict[str, float]:
        denominator = max(self._cycles, 1)
        latency = (self._total_latency / self._completed_packets) if self._completed_packets else 0.0
        # Clip latency at a two-hop reference window so every observation remains in [0, 1].
        return {
            "latency": min(latency / 2.0, 1.0),
            "injection_rate": min(self._injected_packets / denominator, 1.0),
            "throughput": min(self._transferred_flits / denominator, 1.0),
            "stall_ratio": min(self._stalled_cycles / denominator, 1.0),
        }

    def _build_state(self) -> np.ndarray:
        return self.state_builder.build(self.traffic_driver.current_traffic(), self._performance())
