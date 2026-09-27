"""Small adapter around DUT handles; it intentionally contains no RL policy logic."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Iterable


PORTS = ("N", "S", "E", "W", "L")
ROUTER_FEATURES = ("VALID", "READY", "TRANSFER", "STALL", "FULL", "EMPTY")


class NoCInterface:
    """Read and write optional DUT signals while keeping RTL naming isolated.

    The candidate names are deliberately placeholders.  For example, a real
    integration might replace them with `dut.n_valid_i`, `dut.n_ready_o`, and
    the actual router-indexed signal names from the RTL.  Missing state signals
    read as zero so the scaffold can run against the included placeholder DUT.
    """

    def __init__(self, dut: Any) -> None:
        self.dut = dut

    @staticmethod
    def _int_value(value: Any, default: int = 0) -> int:
        """Convert a cocotb handle/value or Python scalar to an integer."""
        try:
            raw = value.value if hasattr(value, "value") else value
            return int(raw)
        except (TypeError, ValueError, AttributeError):
            return default

    def _lookup(self, name: str) -> Any | None:
        try:
            return getattr(self.dut, name)
        except (AttributeError, TypeError):
            return None

    def read_optional(self, names: Iterable[str], default: int = 0) -> int:
        """Read the first present signal from *names*, or return *default*."""
        for name in names:
            handle = self._lookup(name)
            if handle is not None:
                return self._int_value(handle, default)
        return default

    def write_optional(self, names: Iterable[str], value: int) -> bool:
        """Drive the first present signal from *names* and report success."""
        for name in names:
            handle = self._lookup(name)
            if handle is None:
                continue
            try:
                if hasattr(handle, "value"):
                    handle.value = value
                else:
                    setattr(self.dut, name, value)
                return True
            except (AttributeError, TypeError, ValueError):
                continue
        return False

    async def rising_edge(self) -> None:
        """Await one DUT clock edge, or yield once for a plain asyncio mock."""
        mock_edge = self._lookup("rising_edge")
        if callable(mock_edge):
            result = mock_edge()
            if inspect.isawaitable(result):
                await result
            return

        clk = self._lookup("clk")
        if clk is not None:
            try:
                from cocotb.triggers import RisingEdge

                await RisingEdge(clk)
                return
            except ImportError:
                # Unit tests intentionally run without cocotb installed.
                pass
        await asyncio.sleep(0)

    async def reset(self, cycles: int = 2) -> None:
        """Apply an active-low or active-high reset using conventional names."""
        active_low = self.write_optional(("rst_n", "reset_n"), 0)
        active_high = False if active_low else self.write_optional(("reset", "reset_i"), 1)
        for _ in range(cycles):
            await self.rising_edge()
        if active_low:
            self.write_optional(("rst_n", "reset_n"), 1)
        elif active_high:
            self.write_optional(("reset", "reset_i"), 0)
        await self.rising_edge()

    def router_port_feature(self, router: int, port: str, feature: str) -> int:
        """Read a binary router/port feature from a mock matrix or signal names."""
        matrix = self._lookup("router_state")
        if matrix is not None:
            try:
                return int(matrix[router][PORTS.index(port)][ROUTER_FEATURES.index(feature)])
            except (IndexError, KeyError, TypeError, ValueError):
                pass

        stem = f"r{router}_{port.lower()}_{feature.lower()}"
        # These names are examples only; map this list to the actual RTL port list.
        return int(bool(self.read_optional((stem, f"{port.lower()}_{feature.lower()}_i", f"{port.lower()}_{feature.lower()}_o"))))

    def packet_ready(self) -> bool:
        """Return injection readiness; the fallback keeps the placeholder live."""
        return bool(self.read_optional(("packet_ready_o", "injection_ready_o", "local_ready_o"), default=1))
