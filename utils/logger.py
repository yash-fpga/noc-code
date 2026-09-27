"""Consistent logger factory for training entry points."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Mapping

def get_logger(name: str) -> logging.Logger:
    """Return a logger with a minimal stream handler when none is configured."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Cocotb also forwards Python logging through its simulator logger.
        # Keep one direct, human-readable line rather than emitting duplicates.
        logger.propagate = False
    return logger


def log_coverage_progress(logger: logging.Logger, coverage: Mapping[str, Any], label: str) -> None:
    """Emit a local, human-readable coverage measurement to the console."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info(
        "%s | %s | coverage %.2f%% (%d/%d bins; +%d new)",
        timestamp,
        label,
        float(coverage["coverage_percent"]),
        int(coverage["hit_bins"]),
        int(coverage["total_bins"]),
        int(coverage["newly_covered_count"]),
    )