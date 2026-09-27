"""Async DQN loop that keeps simulator time under the cocotb test's control."""

from __future__ import annotations

from agents.dqn_agent import DQNAgent
from rl_env.action_decoder import action_to_index
from rl_env.noc_async_env import AsyncNoCEnv
from utils.logger import get_logger, log_coverage_progress

async def train_dqn(
    env: AsyncNoCEnv,
    episodes: int = 2,
    max_steps: int = 16,
    agent: DQNAgent | None = None,
    coverage_target: float = 100.0,
) -> DQNAgent:
    """Run DQN on incremental traffic coverage inside the cocotb coroutine."""
    learner = agent or DQNAgent(env.config.observation_dim)
    logger = get_logger(__name__)
    env.reset_coverage()
    for episode in range(1, episodes + 1):
        state, _ = await env.reset()
        coverage = env.coverage.snapshot()
        for _ in range(max_steps):
            action = learner.select_action(state)
            next_state, reward, done, info = await env.step(action)
            learner.remember(state, action_to_index(action), reward, next_state, done)
            learner.optimize()
            state = next_state
            coverage = info["coverage"]
            if done:
                break
        log_coverage_progress(logger, coverage, f"DQN episode {episode}/{episodes}")
        if coverage["coverage_percent"] >= coverage_target:
            break
    return learner
