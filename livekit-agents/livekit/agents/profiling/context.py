from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, replace
from typing import Iterator


@dataclass(slots=True, frozen=True)
class TraceContext:
    run_id: str
    session_id: str | None = None
    worker_id: str | None = None
    job_id: str | None = None
    turn_id: str | None = None
    trace_id: str | None = None
    response_id: str | None = None
    attempt_id: str | None = None
    generation_step_id: str | None = None
    request_id: str | None = None
    task_id: str | None = None
    chunk_id: str | None = None
    chunk_idx: int | None = None

    def with_updates(self, **kwargs: object) -> "TraceContext":
        return replace(self, **kwargs)

    def to_fields(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "trace_id": self.trace_id,
            "response_id": self.response_id,
            "attempt_id": self.attempt_id,
            "generation_step_id": self.generation_step_id,
            "request_id": self.request_id,
            "task_id": self.task_id,
            "chunk_id": self.chunk_id,
            "chunk_idx": self.chunk_idx,
        }


_TRACE_CONTEXT_VAR: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "fireredchat_trace_context",
    default=None,
)


def current_trace_context() -> TraceContext | None:
    return _TRACE_CONTEXT_VAR.get()


@contextlib.contextmanager
def use_trace_context(ctx: TraceContext | None) -> Iterator[TraceContext | None]:
    token = _TRACE_CONTEXT_VAR.set(ctx)
    try:
        yield ctx
    finally:
        _TRACE_CONTEXT_VAR.reset(token)
