"""Standalone coverage-closure campaign for the 2x2 NoC RL environment.

Unlike `cocotb_test_entry.py` (a short wiring smoke test), this file is meant
to actually run long enough, and inject often enough, to close all 34
functional coverage bins. Two changes vs. your existing train_q_learning loop:

  1. Runs many more episodes/steps.
  2. Forces `inject=1` on every action, since a random "don't inject" action
     wastes a step that could have been sampling a new bin.

Everything else (Q-table, epsilon, update rule) is untouched -- this script
just wraps your existing agent and gives it a fair amount of work to do.

HOW TO RUN:

  Option A -- quick local sanity check, no simulator needed:
    Uses the MockDUT already defined in your test file, so you can see this
    work in seconds without Icarus/Verilator.

      pytest -q -s coverage_campaign.py

  Option B -- for real, against your actual DUT/simulator:
    Copy the `run_coverage_campaign()` function's body into a new cocotb
    @cocotb.test() function (same pattern as cocotb_test_entry.py), passing
    it a real `dut` handle instead of MockDUT. See the bottom of this file
    for a ready-to-paste example.
"""

from __future__ import annotations

import asyncio

from rl_env.action_decoder import action_to_index
from rl_env.config import NoCConfig
from rl_env.noc_async_env import AsyncNoCEnv
from agents.q_learning_agent import QLearningAgent
from training.checkpoint_utils import save_q_agent


async def run_coverage_campaign(
    env: AsyncNoCEnv,
    agent: QLearningAgent | None = None,
    max_episodes: int = 300,
    max_steps_per_episode: int = 100,
    force_injection: bool = True,
    checkpoint_dir: str = "checkpoints",
) -> QLearningAgent:
    """Run episodes until coverage hits 100% (or max_episodes runs out).

    Saves the Q-table to `checkpoint_dir` every time coverage crosses a new
    25% milestone, plus once more at the very end -- so a Colab disconnect
    never loses more than a partial milestone of progress.
    """
    learner = agent or QLearningAgent()
    env.reset_coverage()

    last_categories = {}
    milestones_saved: set[int] = set()
    for episode in range(1, max_episodes + 1):
        state, _ = await env.reset()
        coverage = env.coverage.snapshot()

        for _ in range(max_steps_per_episode):
            action = list(learner.select_action(state))

            if force_injection:
                action[0] = 1  # index 0 is the "inject" field; always send a packet

            action = tuple(action)
            next_state, reward, done, info = await env.step(action)
            learner.update(state, action_to_index(action), reward, next_state, done)
            state = next_state
            coverage = info["coverage"]
            if done:
                break

        last_categories = coverage["categories"]

        if episode % 10 == 0 or coverage["coverage_percent"] >= 100.0:
            print(
                f"Episode {episode:>4}/{max_episodes} | "
                f"coverage {coverage['coverage_percent']:6.2f}% "
                f"({coverage['hit_bins']}/{coverage['total_bins']} bins)"
            )

        # Save a checkpoint the moment we cross each 25/50/75/100% milestone.
        for milestone in (25, 50, 75, 100):
            if coverage["coverage_percent"] >= milestone and milestone not in milestones_saved:
                milestones_saved.add(milestone)
                save_q_agent(learner, f"{checkpoint_dir}/q_table_{milestone}pct.pkl")
                print(f"  -> saved checkpoint at {milestone}% coverage")

        if coverage["coverage_percent"] >= 100.0:
            print(f"\n100% coverage reached after {episode} episodes.")
            break
    else:
        print(f"\nStopped after {max_episodes} episodes without reaching 100%.")

    save_q_agent(learner, f"{checkpoint_dir}/q_table_final.pkl")
    print(f"\nFinal Q-table saved to {checkpoint_dir}/q_table_final.pkl")

    print("\nPer-category breakdown:")
    for name, stats in last_categories.items():
        marker = "OK" if stats["hit"] == stats["total"] else "--"
        print(f"  [{marker}] {name:<14} {stats['hit']}/{stats['total']}")

    return learner


# ---------------------------------------------------------------------------
# Option A: quick local run against the MockDUT from your existing test file.
# ---------------------------------------------------------------------------

def test_coverage_campaign_hits_100_percent_with_mock_dut() -> None:
    """Run under: pytest -q -s coverage_campaign.py"""
    from tests.test_env_integration import MockDUT  # reuse your existing mock

    dut = MockDUT()
    env = AsyncNoCEnv(dut, NoCConfig(max_steps=100))
    asyncio.run(run_coverage_campaign(env, max_episodes=300, max_steps_per_episode=100))


# ---------------------------------------------------------------------------
# Option B: paste this into a real cocotb test file (same shape as
# cocotb_test_entry.py) to run against your actual simulator/DUT.
# ---------------------------------------------------------------------------
#
# import cocotb
# from cocotb.clock import Clock
# from rl_env.noc_async_env import AsyncNoCEnv
# from coverage_campaign import run_coverage_campaign
#
# @cocotb.test()
# async def noc_rl_coverage_campaign(dut):
#     cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
#     env = AsyncNoCEnv(dut)
#     await run_coverage_campaign(env, max_episodes=300, max_steps_per_episode=100)
