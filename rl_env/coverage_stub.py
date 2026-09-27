"""Traffic functional-coverage collection for the RL-driven NoC stimulus."""


from __future__ import annotations
from collections.abc import Mapping
from typing import Any


class CoverageCollector:
    """Track achievable traffic-generator coverage bins for a 2x2 mesh.

    This first collector measures *stimulus* coverage, rather than claiming
    complete RTL functional coverage. Add RTL response, arbitration, credit,
    and error bins once the actual DUT signals are defined.
    """

    _BIN_VALUES = {
        "source": range(4),
        "destination": range(4),
        "packet_type": ("unicast", "broadcast"),
        "packet_length": range(1, 5),
        "vc": range(2),
        "priority": range(2),
        "injection_gap": range(4),
        "direction": ("N", "S", "E", "W", "L"),
        "hop_count": range(3),
        "flit_type": ("head", "body", "tail", "single"),
    }
    _FLIT_TYPE_NAMES = {0: "head", 1: "body", 2: "tail", 3: "single"}

    def __init__(self) -> None:
        self._hit_bins: dict[str, set[Any]] = {name: set() for name in self._BIN_VALUES}

    @property
    def total_bins(self) -> int:
        """Return the number of currently defined, reachable coverage bins."""
        return sum(len(values) for values in self._BIN_VALUES.values())

    def reset(self) -> None:
        """Start a new coverage campaign without changing the environment state."""
        for values in self._hit_bins.values():
            values.clear()

    def collect(
        self,
        *,
        action: Mapping[str, Any] | None = None,
        flit: Any | None = None,
        accepted_injection: bool = False,
        transferred: bool = False,
    ) -> dict[str, Any]:
        """Sample an accepted action and transferred flit, returning a snapshot."""
        newly_covered: list[str] = []
        if accepted_injection and action is not None:
            src = int(action["src"])
            dest = int(action["dest"])
            packet_type = int(action["packet_type"])
            self._hit("source", src, newly_covered)
            self._hit("destination", dest, newly_covered)
            self._hit("packet_type", "broadcast" if packet_type else "unicast", newly_covered)
            self._hit("packet_length", int(action["packet_length"]), newly_covered)
            self._hit("vc", int(action["vc"]), newly_covered)
            self._hit("priority", int(action["priority"]), newly_covered)
            self._hit("injection_gap", int(action["injection_gap"]), newly_covered)
            self._hit("direction", self._direction(src, dest, packet_type), newly_covered)
            self._hit("hop_count", self._hop_count(src, dest), newly_covered)
        if transferred and flit is not None:
            self._hit("flit_type", self._FLIT_TYPE_NAMES.get(int(flit.flit_type), "unknown"), newly_covered)
        return self.snapshot(newly_covered)

    def snapshot(self, newly_covered: list[str] | None = None) -> dict[str, Any]:
        """Return a serializable summary suitable for environment info and logs."""
        hit_bins = sum(len(values) for values in self._hit_bins.values())
        total_bins = self.total_bins
        categories = {
            name: {"hit": len(self._hit_bins[name]), "total": len(values)}
            for name, values in self._BIN_VALUES.items()
        }
        return {
            "coverage_percent": (100.0 * hit_bins / total_bins) if total_bins else 100.0,
            "hit_bins": hit_bins,
            "total_bins": total_bins,
            "newly_covered": tuple(newly_covered or ()),
            "newly_covered_count": len(newly_covered or ()),
            "categories": categories,
        }

    def _hit(self, point: str, value: Any, newly_covered: list[str]) -> None:
        if value in self._BIN_VALUES[point] and value not in self._hit_bins[point]:
            self._hit_bins[point].add(value)
            newly_covered.append(f"{point}={value}")

    @staticmethod
    def _hop_count(src: int, dest: int) -> int:
        src_row, src_col = divmod(src, 2)
        dest_row, dest_col = divmod(dest, 2)
        return abs(dest_row - src_row) + abs(dest_col - src_col)

    @staticmethod
    def _direction(src: int, dest: int, packet_type: int) -> str:
        if packet_type or src == dest:
            return "L"
        src_row, src_col = divmod(src, 2)
        dest_row, dest_col = divmod(dest, 2)
        if dest_row < src_row:
            return "N"
        if dest_row > src_row:
            return "S"
        if dest_col > src_col:
            return "E"
        return "W"