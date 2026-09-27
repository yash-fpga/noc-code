"""Translate decoded traffic actions into packet/flit signals on the DUT."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from cocotb_tb.noc_interface import NoCInterface


FLIT_HEAD = 0
FLIT_BODY = 1
FLIT_TAIL = 2
FLIT_SINGLE = 3


@dataclass(frozen=True)
class Flit:
    """One internally derived flit; flit type is not part of the RL action."""

    packet_id: int
    src: int
    dest: int
    packet_type: int
    packet_length: int
    vc: int
    priority: int
    position: int
    created_cycle: int

    @property
    def sop(self) -> int:
        return int(self.position == 0)

    @property
    def eop(self) -> int:
        return int(self.position == self.packet_length - 1)

    @property
    def flit_type(self) -> int:
        if self.packet_length == 1:
            return FLIT_SINGLE
        if self.sop:
            return FLIT_HEAD
        if self.eop:
            return FLIT_TAIL
        return FLIT_BODY


class TrafficDriver:
    """Queues packet flits and holds each one until the DUT reports ready.

    Signal names such as `packet_valid_i` are placeholders.  Replace the
    candidates in `_drive_flit` with the real local-injection interface names.
    """

    def __init__(self, interface: NoCInterface) -> None:
        self.interface = interface
        self._pending: deque[Flit] = deque()
        self._next_packet_id = 0
        self._gap_remaining = 0
        self.last_submit_accepted = False

    def reset(self) -> None:
        self._pending.clear()
        self._next_packet_id = 0
        self._gap_remaining = 0
        self.last_submit_accepted = False
        self._drive_idle()

    @property
    def gap_remaining(self) -> int:
        return self._gap_remaining

    @property
    def current_flit(self) -> Flit | None:
        return self._pending[0] if self._pending else None

    def submit_action(self, params: dict[str, Any], cycle: int) -> bool:
        """Accept a requested packet unless an earlier injection gap blocks it."""
        self.last_submit_accepted = False
        if self._gap_remaining:
            self._gap_remaining -= 1
            return False
        if not params["inject"]:
            return False

        packet_id = self._next_packet_id
        self._next_packet_id += 1
        for position in range(params["packet_length"]):
            self._pending.append(
                Flit(
                    packet_id=packet_id,
                    src=params["src"],
                    dest=params["dest"],
                    packet_type=params["packet_type"],
                    packet_length=params["packet_length"],
                    vc=params["vc"],
                    priority=params["priority"],
                    position=position,
                    created_cycle=cycle,
                )
            )
        self._gap_remaining = params["injection_gap"]
        self.last_submit_accepted = True
        return True

    def drive_current_flit(self) -> Flit | None:
        """Drive the queued head flit before the environment awaits a clock edge."""
        flit = self.current_flit
        if flit is None:
            self._drive_idle()
            return None
        self._drive_flit(flit)
        return flit

    def complete_cycle(self) -> Flit | None:
        """Pop and return the driven flit only when the DUT accepted it."""
        flit = self.current_flit
        if flit is not None and self.interface.packet_ready():
            return self._pending.popleft()
        return None

    def current_traffic(self) -> dict[str, int] | None:
        """Return metadata for the current in-flight flit for state construction."""
        flit = self.current_flit
        if flit is None:
            return None
        return {
            "src": flit.src,
            "dest": flit.dest,
            "packet_type": flit.packet_type,
            "vc": flit.vc,
            "priority": flit.priority,
            "sop": flit.sop,
            "eop": flit.eop,
        }

    def _drive_idle(self) -> None:
        self.interface.write_optional(("packet_valid_i", "local_valid_i"), 0)

    def _drive_flit(self, flit: Flit) -> None:
        self.interface.write_optional(("packet_valid_i", "local_valid_i"), 1)
        self.interface.write_optional(("packet_src_i", "local_src_i"), flit.src)
        self.interface.write_optional(("packet_dest_i", "local_dest_i"), flit.dest)
        self.interface.write_optional(("packet_type_i", "local_packet_type_i"), flit.packet_type)
        self.interface.write_optional(("packet_length_i", "local_length_i"), flit.packet_length)
        self.interface.write_optional(("packet_vc_i", "local_vc_i"), flit.vc)
        self.interface.write_optional(("packet_priority_i", "local_priority_i"), flit.priority)
        self.interface.write_optional(("packet_sop_i", "local_sop_i"), flit.sop)
        self.interface.write_optional(("packet_eop_i", "local_eop_i"), flit.eop)
        self.interface.write_optional(("packet_flit_type_i", "local_flit_type_i"), flit.flit_type)
