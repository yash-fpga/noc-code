"""Assemble the fixed-layout 2x2 NoC observation vector."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from cocotb_tb.noc_interface import PORTS, ROUTER_FEATURES, NoCInterface
from rl_env.config import NoCConfig


class StateBuilder:
    """Read DUT signals through :class:`NoCInterface` and produce float32 state."""

    def __init__(self, interface: NoCInterface, config: NoCConfig) -> None:
        self.interface = interface
        self.config = config

    @staticmethod
    def _direction(src: int, dest: int, packet_type: int) -> int:
        """Return N/S/E/W/L as 0/1/2/3/4 using row-major router ids."""
        if packet_type or src == dest:  # Broadcast and local routes have no single outgoing direction.
            return 4
        src_row, src_col = divmod(src, 2)
        dest_row, dest_col = divmod(dest, 2)
        if dest_row < src_row:
            return 0
        if dest_row > src_row:
            return 1
        if dest_col > src_col:
            return 2
        return 3

    @staticmethod
    def _hop_count(src: int, dest: int) -> int:
        src_row, src_col = divmod(src, 2)
        dest_row, dest_col = divmod(dest, 2)
        return abs(dest_row - src_row) + abs(dest_col - src_col)

    def router_features(self) -> np.ndarray:
        """Return 120 binary port/handshake/buffer features in the documented order."""
        values = [
            self.interface.router_port_feature(router, port, feature)
            for router in range(4)
            for port in PORTS
            for feature in ROUTER_FEATURES
        ]
        return np.asarray(values, dtype=np.float32)

    def traffic_features(self, traffic: Mapping[str, int] | None) -> np.ndarray:
        """Return 14 metadata values for the flit currently awaiting transfer."""
        if traffic is None:
            return np.zeros(14, dtype=np.float32)
        src, dest = int(traffic["src"]), int(traffic["dest"])
        direction = self._direction(src, dest, int(traffic["packet_type"]))
        direction_one_hot = np.zeros(5, dtype=np.float32)
        direction_one_hot[direction] = 1.0
        packet_one_hot = np.zeros(2, dtype=np.float32)
        packet_one_hot[int(traffic["packet_type"])] = 1.0
        return np.asarray(
            [src / 3.0, dest / 3.0, *direction_one_hot, self._hop_count(src, dest) / 2.0,
             *packet_one_hot, int(traffic["vc"]), int(traffic["priority"]),
             int(traffic["sop"]), int(traffic["eop"])],
            dtype=np.float32,
        )

    def build(self, traffic: Mapping[str, int] | None, performance: Mapping[str, float]) -> np.ndarray:
        """Concatenate router, traffic, performance, and future coverage segments."""
        performance_values = np.asarray(
            [performance["latency"], performance["injection_rate"], performance["throughput"], performance["stall_ratio"]],
            dtype=np.float32,
        )
        coverage = np.zeros(self.config.coverage_len, dtype=np.float32)
        state = np.concatenate((self.router_features(), self.traffic_features(traffic), performance_values, coverage))
        assert state.shape == (self.config.observation_dim,)
        return state.astype(np.float32, copy=False)
