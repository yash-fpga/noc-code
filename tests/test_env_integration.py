"""Signal-free asyncio smoke test for the async environment integration seam."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rl_env.config import NoCConfig
from rl_env.coverage_stub import CoverageCollector
from rl_env.noc_async_env import AsyncNoCEnv


class MockDUT:
    """Minimal DUT-shaped object; `rising_edge` replaces cocotb's clock trigger."""

    def __init__(self) -> None:
        self.edges = 0
        self.rst_n = 1
        self.packet_valid_i = 0
        self.packet_ready_o = 1
        self.packet_src_i = 0
        self.packet_dest_i = 0
        self.packet_type_i = 0
        self.packet_length_i = 0
        self.packet_vc_i = 0
        self.packet_priority_i = 0
        self.packet_sop_i = 0
        self.packet_eop_i = 0
        self.packet_flit_type_i = 0
        self.router_state = [[[0 for _ in range(6)] for _ in range(5)] for _ in range(4)]

    async def rising_edge(self) -> None:
        self.edges += 1
        # Make one router feature observable, proving StateBuilder reads the mock.
        self.router_state[0][0][0] = int(bool(self.packet_valid_i))


@pytest.mark.asyncio
async def test_async_reset_and_step_with_mock_dut() -> None:
    dut = MockDUT()
    env = AsyncNoCEnv(dut, NoCConfig(coverage_len=3, max_steps=3, reset_cycles=2))

    state, info = await env.reset()
    assert state.shape == (141,)
    assert state.dtype == np.float32
    assert info["coverage"]["coverage_percent"] == 0.0
    assert dut.edges == 3

    next_state, reward, done, step_info = await env.step(np.asarray([1, 0, 3, 0, 1, 1, 1, 0]))
    assert next_state.shape == (141,)
    assert reward > 0.0 and not done
    assert step_info["accepted_injection"] and step_info["transferred"]
    assert dut.packet_valid_i == 1
    assert dut.packet_src_i == 0 and dut.packet_dest_i == 3
    assert dut.packet_length_i == 2 and dut.packet_sop_i == 1
    assert next_state[-3:].tolist() == [0.0, 0.0, 0.0]
    assert step_info["coverage"]["coverage_percent"] > 0.0
    assert step_info["coverage"]["hit_bins"] > 0
    _, _, done, _ = await env.step([0, 0, 0, 0, 0, 0, 0, 0])
    _, _, done, _ = await env.step([0, 0, 0, 0, 0, 0, 0, 0])
    assert done
    with pytest.raises(RuntimeError):
        await env.step([0, 0, 0, 0, 0, 0, 0, 0])

def test_traffic_coverage_reaches_100_percent_for_all_defined_bins() -> None:
    collector = CoverageCollector()
    base = {"src": 0, "dest": 0, "packet_type": 0, "packet_length": 1, "vc": 0, "priority": 0, "injection_gap": 0}

    def sample(**overrides: int) -> None:
        collector.collect(action={**base, **overrides}, accepted_injection=True)

    for src in range(4):
        sample(src=src)
    for dest in range(4):
        sample(dest=dest)
    for packet_type in range(2):
        sample(packet_type=packet_type)
    for packet_length in range(1, 5):
        sample(packet_length=packet_length)
    for vc in range(2):
        sample(vc=vc)
    for priority in range(2):
        sample(priority=priority)
    for injection_gap in range(4):
        sample(injection_gap=injection_gap)

    # N, S, E, W, L and all Manhattan distances 0, 1, 2.
    for src, dest, packet_type in ((2, 0, 0), (0, 2, 0), (0, 1, 0), (1, 0, 0), (0, 0, 0), (0, 3, 0)):
        sample(src=src, dest=dest, packet_type=packet_type)
    for flit_type in range(4):
        collector.collect(flit=SimpleNamespace(flit_type=flit_type), transferred=True)

    coverage = collector.snapshot()
    assert coverage["total_bins"] == 34
    assert coverage["hit_bins"] == 34
    assert coverage["coverage_percent"] == 100.0