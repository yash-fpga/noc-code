"""Save/load helpers so a training run's progress survives a Colab disconnect."""

from __future__ import annotations

import pickle
from pathlib import Path

from agents.q_learning_agent import QLearningAgent


def save_q_agent(agent: QLearningAgent, path: str | Path) -> None:
    """Pickle the Q-table (a plain dict of bytes -> float32 array) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(agent.q_table), f)


def load_q_agent(path: str | Path, **agent_kwargs) -> QLearningAgent:
    """Rebuild a QLearningAgent from a saved Q-table."""
    agent = QLearningAgent(**agent_kwargs)
    with open(path, "rb") as f:
        table = pickle.load(f)
    for key, values in table.items():
        agent.q_table[key][:] = values
    return agent


def save_dqn_agent(agent, path: str | Path) -> None:
    """Save DQN weights (only the online network is needed to resume/evaluate)."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.online.state_dict(), path)


def load_dqn_agent(agent, path: str | Path) -> None:
    """Load DQN weights into an already-constructed agent, in place."""
    import torch

    agent.online.load_state_dict(torch.load(path))
    agent.target.load_state_dict(agent.online.state_dict())
