"""The sole cocotb test entry point: it owns both training and simulation time."""

from __future__ import annotations

import cocotb
from cocotb.clock import Clock

from rl_env.noc_async_env import AsyncNoCEnv
from training.train_q_learning import train_q_learning


@cocotb.test()
async def noc_rl_end_to_end_smoke(dut):
    """Prove cocotb -> AsyncNoCEnv -> traffic driver -> DUT works end to end."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    env = AsyncNoCEnv(dut)
    await env.reset()
    _, _, _, injection_info = await env.step((1, 0, 3, 0, 0, 0, 0, 0))
    assert injection_info["accepted_injection"]
    await train_q_learning(env, episodes=1, max_steps=4)
