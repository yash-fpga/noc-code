"""Async tabular training loop for use only inside a cocotb coroutine."""

from __future__ import annotations

from agents.q_learning_agent import QLearningAgent
from rl_env.noc_async_env import AsyncNoCEnv
from utils.logger import get_logger, log_coverage_progress

async def train_q_learning(
    env: AsyncNoCEnv,
    episodes: int = 2,
    max_steps: int = 16,
    agent: QLearningAgent | None = None,
    coverage_target: float = 100.0,
) -> QLearningAgent:
    """Train on incremental traffic coverage and report it after every episode."""
    learner = agent or QLearningAgent()
    logger = get_logger(__name__)
    env.reset_coverage()
    for episode in range(1, episodes + 1):
        state, _ = await env.reset()
        coverage = env.coverage.snapshot()
        for _ in range(max_steps):
            action = learner.select_action(state)
            next_state, reward, done, info = await env.step(action)
            learner.update(state, action, reward, next_state, done)
            state = next_state
            coverage = info["coverage"]
            if done:
                break
        log_coverage_progress(logger, coverage, f"Q-learning episode {episode}/{episodes}")
        if coverage["coverage_percent"] >= coverage_target:
            break
    return learner
