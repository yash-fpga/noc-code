"""PyTorch DQN scaffold that predicts all flattened traffic actions."""

from __future__ import annotations

import numpy as np

from rl_env.action_decoder import index_to_action
from rl_env.config import FLATTENED_ACTION_DIM
from utils.replay_buffer import ReplayBuffer, Transition

try:
    import torch
    from torch import nn
except ImportError:  # Keep imports usable until optional runtime dependencies are installed.
    torch = None
    nn = None


if nn is not None:
    class DQNNetwork(nn.Module):
        """Small MLP mapping the observation vector to 4,096 action values."""

        def __init__(self, observation_dim: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(observation_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, FLATTENED_ACTION_DIM),
            )

        def forward(self, state):
            return self.layers(state)
else:
    class DQNNetwork:  # type: ignore[no-redef]
        def __init__(self, observation_dim: int) -> None:
            del observation_dim
            raise RuntimeError("Install torch to use DQNNetwork.")


class DQNAgent:
    """Replay-buffer DQN reference implementation for future reward shaping."""

    def __init__(self, observation_dim: int, learning_rate: float = 1e-3, discount: float = 0.99, epsilon: float = 0.1, target_sync_interval: int = 100) -> None:
        if torch is None or nn is None:
            raise RuntimeError("Install torch to use DQNAgent.")
        self.discount, self.epsilon, self.target_sync_interval = discount, epsilon, target_sync_interval
        self.online = DQNNetwork(observation_dim)
        self.target = DQNNetwork(observation_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = ReplayBuffer()
        self._updates = 0

    def select_action_index(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return int(np.random.randint(FLATTENED_ACTION_DIM))
        with torch.no_grad():
            values = self.online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(torch.argmax(values, dim=1).item())

    def select_action(self, state: np.ndarray, explore: bool = True) -> tuple[int, ...]:
        return index_to_action(self.select_action_index(state, explore))

    def remember(self, state: np.ndarray, action_index: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.append(Transition(state.copy(), action_index, reward, next_state.copy(), done))

    def optimize(self, batch_size: int = 32) -> float | None:
        """Run one DQN update once enough replay items are available."""
        if len(self.replay) < batch_size:
            return None
        batch = self.replay.sample(batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32)
        actions = torch.as_tensor([item.action_index for item in batch], dtype=torch.int64)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32)
        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            targets = rewards + self.discount * (1.0 - dones) * self.target(next_states).max(dim=1).values
        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._updates += 1
        if self._updates % self.target_sync_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
