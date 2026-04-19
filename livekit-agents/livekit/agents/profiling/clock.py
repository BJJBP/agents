from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from os import getpid


def now_mono_ns() -> int:
    return time.monotonic_ns()


def now_wall_time_ns() -> int:
    return time.time_ns()


@dataclass(slots=True, frozen=True)
class ClockSnapshot:
    mono_ns: int
    wall_time_ns: int


def clock_snapshot() -> ClockSnapshot:
    return ClockSnapshot(mono_ns=now_mono_ns(), wall_time_ns=now_wall_time_ns())


def make_clock_anchor(*, component: str, run_id: str) -> dict[str, object]:
    snap = clock_snapshot()
    return {
        "schema_version": 1,
        "record_type": "clock_anchor",
        "component": component,
        "host": socket.gethostname(),
        "pid": getpid(),
        "run_id": run_id,
        "mono_ns": snap.mono_ns,
        "wall_time_ns": snap.wall_time_ns,
    }
