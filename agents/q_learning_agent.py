"""Tabular Q-learning over a discretized NoC state and flattened action space."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from rl_env.action_decoder import index_to_action
from rl_env.config import FLATTENED_ACTION_DIM


class QLearningAgent:
    """A small tabular baseline; Q-table keys are discretized state byte strings."""

    def __init__(self, learning_rate: float = 0.1, discount: float = 0.99, epsilon: float = 1.0, bins: int = 8, seed: int | None = None) -> None:
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.bins = bins
        self.rng = np.random.default_rng(seed)
        self.q_table: defaultdict[bytes, np.ndarray] = defaultdict(lambda: np.zeros(FLATTENED_ACTION_DIM, dtype=np.float32))

    def state_key(self, state: np.ndarray) -> bytes:
        """Hash-ready state representation: values clipped and quantized to `bins`."""
        discrete = np.clip(np.asarray(state, dtype=np.float32), 0.0, 1.0) * (self.bins - 1)
        return np.rint(discrete).astype(np.uint8).tobytes()

    def select_action_index(self, state: np.ndarray, explore: bool = True) -> int:
        key = self.state_key(state)
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(FLATTENED_ACTION_DIM))
        return int(np.argmax(self.q_table[key]))

    def select_action(self, state: np.ndarray, explore: bool = True) -> tuple[int, ...]:
        """Return a MultiDiscrete tuple usable directly by `AsyncNoCEnv.step`."""
        return index_to_action(self.select_action_index(state, explore))

    def update(self, state: np.ndarray, action: int | tuple[int, ...], reward: float, next_state: np.ndarray, done: bool) -> None:
        """Apply the one-step tabular Q-learning update."""
        from rl_env.action_decoder import action_to_index

        action_index = int(action) if isinstance(action, (int, np.integer)) else action_to_index(action)
        key, next_key = self.state_key(state), self.state_key(next_state)
        target = reward if done else reward + self.discount * float(np.max(self.q_table[next_key]))
        self.q_table[key][action_index] += self.learning_rate * (target - self.q_table[key][action_index])
