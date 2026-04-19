from __future__ import annotations

import atexit
import os
import threading
from os import getpid

from prometheus_client import Counter, Gauge, Histogram, multiprocess


_LATENCY_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
)


FIREREDCHAT_TIMESPAN_LATENCY = Histogram(
    "fireredchat_timespan_latency_seconds",
    "Latency of FireRedChat profiling spans.",
    ["span"],
    buckets=_LATENCY_BUCKETS,
)

FIREREDCHAT_E2E_TRACES_STARTED = Counter(
    "fireredchat_e2e_traces_started_total",
    "Total number of turn-level traces started.",
)

FIREREDCHAT_E2E_TRACES_FINISHED = Counter(
    "fireredchat_e2e_traces_finished_total",
    "Total number of turn-level traces finished.",
    ["status"],
)

FIREREDCHAT_E2E_TRACES_FIRST_AUDIO = Counter(
    "fireredchat_e2e_traces_first_audio_total",
    "Total number of turn-level traces that reached first playable agent audio.",
)

FIREREDCHAT_AUDIO_STALLED_SESSIONS = Gauge(
    "fireredchat_audio_stalled_sessions",
    "Number of sessions currently in audio stall.",
    multiprocess_mode="livesum",
)

FIREREDCHAT_TTS_SCHED_WAITING = Gauge(
    "fireredchat_tts_scheduler_waiting_tasks",
    "Number of tasks waiting in the shared TTS scheduler.",
    multiprocess_mode="livesum",
)

FIREREDCHAT_TTS_SCHED_RUNNING = Gauge(
    "fireredchat_tts_scheduler_running_tasks",
    "Number of tasks currently authorized by the shared TTS scheduler.",
    multiprocess_mode="livesum",
)

_STALL_LOCK = threading.Lock()
_ACTIVE_STALLED_SESSIONS: set[str] = set()


def _sync_stall_gauge() -> None:
    FIREREDCHAT_AUDIO_STALLED_SESSIONS.set(float(len(_ACTIVE_STALLED_SESSIONS)))


def _cleanup_multiprocess_state() -> None:
    clear_session_stalls()
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        try:
            multiprocess.mark_process_dead(getpid())
        except Exception:
            pass


atexit.register(_cleanup_multiprocess_state)


def observe_timespan_latency(span: str, duration_ns: int | float) -> None:
    FIREREDCHAT_TIMESPAN_LATENCY.labels(span=span).observe(float(duration_ns) / 1e9)


def trace_started() -> None:
    FIREREDCHAT_E2E_TRACES_STARTED.inc()


def trace_finished(status: str) -> None:
    FIREREDCHAT_E2E_TRACES_FINISHED.labels(status=status).inc()


def trace_first_audio() -> None:
    FIREREDCHAT_E2E_TRACES_FIRST_AUDIO.inc()


def set_session_stall(session_id: str | None, active: bool) -> None:
    if not session_id:
        return

    with _STALL_LOCK:
        if active:
            _ACTIVE_STALLED_SESSIONS.add(session_id)
        else:
            _ACTIVE_STALLED_SESSIONS.discard(session_id)
        _sync_stall_gauge()


def clear_session_stalls() -> None:
    with _STALL_LOCK:
        _ACTIVE_STALLED_SESSIONS.clear()
        _sync_stall_gauge()


def _reset_state_for_tests() -> None:
    clear_session_stalls()
