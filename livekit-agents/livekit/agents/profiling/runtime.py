from __future__ import annotations

import contextlib
import socket
from dataclasses import dataclass, field
from os import getpid
from pathlib import Path
from typing import Any, Iterator

from .clock import make_clock_anchor, now_mono_ns, now_wall_time_ns
from .context import TraceContext, current_trace_context
from .jsonl import JsonlWriter
from .prometheus import observe_timespan_latency


def _serialize_context(trace_ctx: TraceContext | None) -> dict[str, object]:
    return trace_ctx.to_fields() if trace_ctx is not None else {}


@dataclass(slots=True)
class _OpenSpan:
    runtime: "ProfilingRuntime"
    span_name: str
    trace_ctx: TraceContext | None
    attrs: dict[str, Any] = field(default_factory=dict)
    start_mono_ns: int = field(default_factory=now_mono_ns)
    start_wall_time_ns: int = field(default_factory=now_wall_time_ns)

    def close(self, *, attrs: dict[str, Any] | None = None) -> None:
        merged = dict(self.attrs)
        if attrs:
            merged.update(attrs)
        self.runtime.emit_logical_span(
            self.span_name,
            start_mono_ns=self.start_mono_ns,
            end_mono_ns=now_mono_ns(),
            start_wall_time_ns=self.start_wall_time_ns,
            end_wall_time_ns=now_wall_time_ns(),
            trace_ctx=self.trace_ctx,
            attrs=merged,
        )


class ProfilingRuntime:
    def __init__(
        self,
        *,
        component: str,
        log_root: Path,
        run_id: str,
        worker_id: str | None = None,
        job_id: str | None = None,
        session_id: str | None = None,
        extra_file_tag: str | None = None,
    ) -> None:
        self.component = component
        self.run_id = run_id
        self.worker_id = worker_id
        self.job_id = job_id
        self.session_id = session_id
        filename = f"{run_id}_{getpid()}"
        if extra_file_tag:
            filename = f"{filename}_{extra_file_tag}"
        self._writer = JsonlWriter(log_root / component / f"{filename}.jsonl")
        self._host = socket.gethostname()
        self._pid = getpid()

    @property
    def path(self) -> Path:
        return self._writer.path

    def base_trace_context(self) -> TraceContext:
        return TraceContext(
            run_id=self.run_id,
            worker_id=self.worker_id,
            job_id=self.job_id,
            session_id=self.session_id,
        )

    def write_clock_anchor(self) -> None:
        anchor = make_clock_anchor(component=self.component, run_id=self.run_id)
        anchor.update(
            {
                "worker_id": self.worker_id,
                "job_id": self.job_id,
                "session_id": self.session_id,
            }
        )
        self._writer.write(anchor)

    def emit_event(
        self,
        event_name: str,
        *,
        trace_ctx: TraceContext | None = None,
        mono_ns: int | None = None,
        wall_time_ns: int | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        trace_ctx = trace_ctx or current_trace_context()
        payload = {
            "schema_version": 1,
            "record_type": "event",
            "component": self.component,
            "host": self._host,
            "pid": self._pid,
            "event": event_name,
            "mono_ns": mono_ns if mono_ns is not None else now_mono_ns(),
            "wall_time_ns": wall_time_ns if wall_time_ns is not None else now_wall_time_ns(),
            **_serialize_context(trace_ctx),
            "attrs": attrs or {},
        }
        self._writer.write(payload)

    def emit_logical_span(
        self,
        span_name: str,
        *,
        start_mono_ns: int,
        end_mono_ns: int,
        start_wall_time_ns: int,
        end_wall_time_ns: int,
        trace_ctx: TraceContext | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        duration_ns = max(0, end_mono_ns - start_mono_ns)
        payload = {
            "schema_version": 1,
            "record_type": "logical_span",
            "component": self.component,
            "host": self._host,
            "pid": self._pid,
            "span": span_name,
            "start_mono_ns": start_mono_ns,
            "end_mono_ns": end_mono_ns,
            "start_wall_time_ns": start_wall_time_ns,
            "end_wall_time_ns": end_wall_time_ns,
            "duration_ns": duration_ns,
            **_serialize_context(trace_ctx or current_trace_context()),
            "attrs": attrs or {},
        }
        self._writer.write(payload)
        observe_timespan_latency(span_name, duration_ns)

    def emit_state_span(
        self,
        state_name: str,
        *,
        state: str,
        start_mono_ns: int,
        end_mono_ns: int,
        trace_ctx: TraceContext | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        duration_ns = max(0, end_mono_ns - start_mono_ns)
        payload = {
            "schema_version": 1,
            "record_type": "state_span",
            "component": self.component,
            "host": self._host,
            "pid": self._pid,
            "span": state_name,
            "state": state,
            "start_mono_ns": start_mono_ns,
            "end_mono_ns": end_mono_ns,
            "duration_ns": duration_ns,
            **_serialize_context(trace_ctx or current_trace_context()),
            "attrs": attrs or {},
        }
        self._writer.write(payload)

    @contextlib.contextmanager
    def open_logical_span(
        self,
        span_name: str,
        *,
        trace_ctx: TraceContext | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> Iterator[_OpenSpan]:
        span = _OpenSpan(
            runtime=self,
            span_name=span_name,
            trace_ctx=trace_ctx or current_trace_context(),
            attrs=attrs or {},
        )
        try:
            yield span
        finally:
            span.close()

    def flush(self) -> None:
        # JsonlWriter flushes on every write; kept for API symmetry.
        return None

    def close(self) -> None:
        self._writer.close()
