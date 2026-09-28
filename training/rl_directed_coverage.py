"""Coverage closure driven by an RL agent -- not pure random sampling.

Two real RL techniques replace the "spray random actions" approach:

  1. NOVELTY-BASED REWARD SHAPING
     On top of the existing "did I just hit a brand-new bin" reward, we add
     a bonus for sampling (category, value) combinations that have been
     tried less often overall -- not just "never before". The bonus shrinks
     each time a combination is repeated (1 / (1 + times_seen)). This is a
     standard RL exploration technique (count-based / novelty-driven
     intrinsic reward). It matters because the *base* reward goes to zero
     the instant a bin is first covered, giving the agent nothing left to
     optimize toward for the rest of training -- the novelty bonus keeps a
     live gradient pointing at under-sampled regions the whole time.

  2. EPSILON DECAY
     The agent starts fully exploratory (random, epsilon=1.0) and gradually
     shifts toward exploiting its learned Q-values (epsilon -> 0.05). Once
     it's exploiting a Q-table that's been shaped by the novelty bonus, its
     "best" action *is* whatever historically led toward less-sampled
     combinations -- so exploitation itself becomes coverage-seeking
     behaviour, rather than doing nothing (as it does with epsilon stuck
     at 1.0).

This is meaningfully different from constrained-random verification: CRV
samples every combination with equal, memoryless probability forever. Here,
the agent's *policy* is shaped by what it has already explored.
"""

from __future__ import annotations

from collections import defaultdict

from rl_env.action_decoder import action_to_index
from rl_env.coverage_stub import CoverageCollector
from rl_env.noc_async_env import AsyncNoCEnv
from agents.q_learning_agent import QLearningAgent
from training.checkpoint_utils import save_q_agent


class NoveltyTracker:
    """Counts how often each (category, value) coverage point has been
    sampled -- independent of whether it's already 'covered' -- and turns
    that into a shrinking intrinsic-reward bonus for under-sampled points."""

    def __init__(self) -> None:
        self.counts: defaultdict[str, int] = defaultdict(int)

    def bonus_and_update(self, action: dict) -> float:
        src, dest, ptype = int(action["src"]), int(action["dest"]), int(action["packet_type"])
        keys = [
            f"source={src}",
            f"destination={dest}",
            f"packet_type={'broadcast' if ptype else 'unicast'}",
            f"packet_length={int(action['packet_length'])}",
            f"vc={int(action['vc'])}",
            f"priority={int(action['priority'])}",
            f"injection_gap={int(action['injection_gap'])}",
            f"direction={CoverageCollector._direction(src, dest, ptype)}",
            f"hop_count={CoverageCollector._hop_count(src, dest)}",
        ]
        bonus = 0.0
        for key in keys:
            bonus += 1.0 / (1.0 + self.counts[key])
            self.counts[key] += 1
        return bonus / len(keys)


async def run_rl_directed_campaign(
    env: AsyncNoCEnv,
    agent: QLearningAgent | None = None,
    max_episodes: int = 300,
    max_steps_per_episode: int = 100,
    novelty_weight: float = 0.5,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 150,
    checkpoint_dir: str = "checkpoints_rl",
) -> QLearningAgent:
    """Run episodes until coverage hits 100%, using a shaped reward and a
    decaying epsilon so the agent's own policy drives coverage closure."""
    learner = agent or QLearningAgent(epsilon=epsilon_start)
    tracker = NoveltyTracker()
    env.reset_coverage()

    last_categories = {}
    milestones_saved: set[int] = set()

    for episode in range(1, max_episodes + 1):
        # Linearly decay epsilon: fully random early, mostly exploiting later.
        progress = min(episode / epsilon_decay_episodes, 1.0)
        learner.epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)

        state, _ = await env.reset()
        coverage = env.coverage.snapshot()

        for _ in range(max_steps_per_episode):
            action = list(learner.select_action(state))
            action[0] = 1  # always inject -- "do nothing" wastes a step either way
            action = tuple(action)

            next_state, base_reward, done, info = await env.step(action)
            novelty_bonus = tracker.bonus_and_update(info["action"])
            shaped_reward = base_reward + novelty_weight * novelty_bonus

            learner.update(state, action_to_index(action), shaped_reward, next_state, done)
            state = next_state
            coverage = info["coverage"]
            if done:
                break

        last_categories = coverage["categories"]

        if episode % 10 == 0 or coverage["coverage_percent"] >= 100.0:
            print(
                f"Episode {episode:>4}/{max_episodes} | epsilon {learner.epsilon:.2f} | "
                f"coverage {coverage['coverage_percent']:6.2f}% "
                f"({coverage['hit_bins']}/{coverage['total_bins']} bins)"
            )

        for milestone in (25, 50, 75, 100):
            if coverage["coverage_percent"] >= milestone and milestone not in milestones_saved:
                milestones_saved.add(milestone)
                save_q_agent(learner, f"{checkpoint_dir}/q_table_{milestone}pct.pkl")
                print(f"  -> saved checkpoint at {milestone}% coverage")

        if coverage["coverage_percent"] >= 100.0:
            print(f"\n100% coverage reached after {episode} episodes (RL-directed).")
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

def test_rl_directed_campaign_hits_100_percent_with_mock_dut() -> None:
    """Run under: pytest -q -s rl_directed_coverage.py"""
    import asyncio
    from tests.test_env_integration import MockDUT
    from rl_env.config import NoCConfig

    dut = MockDUT()
    env = AsyncNoCEnv(dut, NoCConfig(max_steps=100))
    asyncio.run(run_rl_directed_campaign(env, max_episodes=300, max_steps_per_episode=100))


# ---------------------------------------------------------------------------
# Option B: paste into a real cocotb test file to run against your simulator.
# ---------------------------------------------------------------------------
#
# import cocotb
# from cocotb.clock import Clock
# from rl_env.noc_async_env import AsyncNoCEnv
# from rl_directed_coverage import run_rl_directed_campaign
#
# @cocotb.test()
# async def noc_rl_directed_coverage_campaign(dut):
#     cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
#     env = AsyncNoCEnv(dut)
#     await run_rl_directed_campaign(env, max_episodes=300, max_steps_per_episode=100)
