"""Dimensions and Gymnasium-style space definitions for the 2x2 mesh."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np

try:
    from gymnasium import spaces
except ImportError:  # Lets the signal-free scaffold be imported before installation.
    class _MultiDiscrete:
        def __init__(self, nvec: list[int]) -> None:
            self.nvec = np.asarray(nvec, dtype=np.int64)

        def contains(self, value: object) -> bool:
            array = np.asarray(value)
            return array.shape == self.nvec.shape and bool(np.all((0 <= array) & (array < self.nvec)))

    class _Box:
        def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: type[np.float32]) -> None:
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class _Spaces:
        MultiDiscrete = _MultiDiscrete
        Box = _Box

    spaces = _Spaces()


ROUTER_COUNT = 4
PORT_COUNT = 5
FEATURES_PER_PORT = 6
ROUTER_STATE_DIM = ROUTER_COUNT * PORT_COUNT * FEATURES_PER_PORT
TRAFFIC_STATE_DIM = 14
PERFORMANCE_STATE_DIM = 4
ACTION_NVECS = (2, 4, 4, 2, 4, 2, 2, 4)
FLATTENED_ACTION_DIM = prod(ACTION_NVECS)


@dataclass(frozen=True)
class NoCConfig:
    """Fixed mesh dimensions plus adjustable episode and coverage lengths."""

    coverage_len: int = 0
    max_steps: int = 100
    reset_cycles: int = 2

    @property
    def observation_dim(self) -> int:
        return ROUTER_STATE_DIM + TRAFFIC_STATE_DIM + PERFORMANCE_STATE_DIM + self.coverage_len

    @property
    def action_space(self):
        return spaces.MultiDiscrete(np.asarray(ACTION_NVECS, dtype=np.int64))

    @property
    def observation_space(self):
        return spaces.Box(low=0.0, high=1.0, shape=(self.observation_dim,), dtype=np.float32)
