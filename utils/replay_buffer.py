"""Simple replay buffer used by the DQN scaffold."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Bounded random-sampling replay memory."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._items: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self._items):
            raise ValueError("Cannot sample more transitions than are present.")
        return random.sample(self._items, batch_size)
