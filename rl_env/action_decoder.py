"""Conversions between a MultiDiscrete action, flat index, and traffic fields."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rl_env.config import ACTION_NVECS, FLATTENED_ACTION_DIM


def _validate_action_tuple(action: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    values = tuple(int(value) for value in action)
    if len(values) != len(ACTION_NVECS):
        raise ValueError(f"Expected {len(ACTION_NVECS)} action values, got {len(values)}")
    if any(value < 0 or value >= limit for value, limit in zip(values, ACTION_NVECS)):
        raise ValueError(f"Action {values} is outside MultiDiscrete{ACTION_NVECS}")
    return values


def action_to_index(action: Sequence[int] | np.ndarray) -> int:
    """Flatten a valid MultiDiscrete action to one of 4,096 Q-value indices."""
    index = 0
    for value, radix in zip(_validate_action_tuple(action), ACTION_NVECS):
        index = index * radix + value
    return index


def index_to_action(index: int) -> tuple[int, ...]:
    """Invert :func:`action_to_index` without depending on Gymnasium helpers."""
    if not 0 <= int(index) < FLATTENED_ACTION_DIM:
        raise ValueError(f"Action index must be in [0, {FLATTENED_ACTION_DIM}), got {index}")
    result = [0] * len(ACTION_NVECS)
    remaining = int(index)
    for position in range(len(ACTION_NVECS) - 1, -1, -1):
        remaining, result[position] = divmod(remaining, ACTION_NVECS[position])
    return tuple(result)


def decode_action(action: int | Sequence[int] | np.ndarray) -> dict[str, int | str]:
    """Decode an index or action tuple into packet-generator parameters.

    Packet length is stored by the action as 0--3 and exposed as 1--4 flits.
    """
    values = index_to_action(action) if isinstance(action, (int, np.integer)) else _validate_action_tuple(action)
    inject, src, dest, packet_type, length_code, vc, priority, injection_gap = values
    return {
        "inject": inject,
        "src": src,
        "dest": dest,
        "packet_type": packet_type,
        "packet_type_name": "broadcast" if packet_type else "unicast",
        "packet_length": length_code + 1,
        "vc": vc,
        "priority": priority,
        "injection_gap": injection_gap,
    }
