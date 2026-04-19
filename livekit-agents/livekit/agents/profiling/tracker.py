from __future__ import annotations

from dataclasses import dataclass, field

from .clock import now_mono_ns, now_wall_time_ns
from .context import TraceContext
from .ids import ProfilingIdGenerator
from .prometheus import trace_finished, trace_started
from .runtime import ProfilingRuntime


@dataclass(slots=True)
class TurnTracker:
    runtime: ProfilingRuntime
    base_context: TraceContext
    current_context: TraceContext | None = None
    turn_started_mono_ns: int | None = None
    turn_started_wall_time_ns: int | None = None
    first_user_audio_mono_ns: int | None = None
    first_user_audio_wall_time_ns: int | None = None
    last_user_audio_mono_ns: int | None = None
    last_user_audio_wall_time_ns: int | None = None
    last_vad_end_mono_ns: int | None = None
    last_vad_end_wall_time_ns: int | None = None
    _trace_start_by_id: dict[str, tuple[int, int]] = field(default_factory=dict)
    _user_last_audio_by_trace_id: dict[str, tuple[int, int]] = field(default_factory=dict)

    def ensure_turn_started(self) -> TraceContext:
        if self.current_context is not None:
            return self.current_context

        start_mono_ns = now_mono_ns()
        start_wall_time_ns = now_wall_time_ns()
        ctx = self.base_context.with_updates(
            turn_id=ProfilingIdGenerator.turn_id(),
            trace_id=ProfilingIdGenerator.trace_id(),
        )
        self.current_context = ctx
        self.turn_started_mono_ns = start_mono_ns
        self.turn_started_wall_time_ns = start_wall_time_ns
        self.first_user_audio_mono_ns = None
        self.first_user_audio_wall_time_ns = None
        self.last_user_audio_mono_ns = None
        self.last_user_audio_wall_time_ns = None
        self.last_vad_end_mono_ns = None
        self.last_vad_end_wall_time_ns = None
        self._trace_start_by_id[ctx.trace_id or ""] = (start_mono_ns, start_wall_time_ns)
        self.runtime.emit_event(
            "turn_started",
            trace_ctx=ctx,
            mono_ns=start_mono_ns,
            wall_time_ns=start_wall_time_ns,
        )
        self.runtime.emit_event(
            "trace_started",
            trace_ctx=ctx,
            mono_ns=start_mono_ns,
            wall_time_ns=start_wall_time_ns,
        )
        trace_started()
        return ctx

    def on_raw_user_audio_frame(self) -> TraceContext:
        ctx = self.ensure_turn_started()
        now_mono = now_mono_ns()
        now_wall = now_wall_time_ns()
        if self.first_user_audio_mono_ns is None:
            self.first_user_audio_mono_ns = now_mono
            self.first_user_audio_wall_time_ns = now_wall
            self.runtime.emit_event(
                "user_first_audio_frame",
                trace_ctx=ctx,
                mono_ns=now_mono,
                wall_time_ns=now_wall,
            )
        self.last_user_audio_mono_ns = now_mono
        self.last_user_audio_wall_time_ns = now_wall
        return ctx

    def on_vad_start(self, *, attrs: dict[str, object] | None = None) -> TraceContext:
        ctx = self.ensure_turn_started()
        self.runtime.emit_event("vad_start_of_speech", trace_ctx=ctx, attrs=attrs)
        return ctx

    def on_vad_end(self, *, attrs: dict[str, object] | None = None) -> TraceContext:
        ctx = self.ensure_turn_started()
        end_mono_ns = now_mono_ns()
        end_wall_time_ns = now_wall_time_ns()
        self.last_vad_end_mono_ns = end_mono_ns
        self.last_vad_end_wall_time_ns = end_wall_time_ns
        self.runtime.emit_event(
            "vad_end_of_speech",
            trace_ctx=ctx,
            mono_ns=end_mono_ns,
            wall_time_ns=end_wall_time_ns,
            attrs=attrs,
        )
        if (
            self.last_user_audio_mono_ns is not None
            and self.last_user_audio_wall_time_ns is not None
        ):
            self.runtime.emit_logical_span(
                "vad_endpointing",
                start_mono_ns=self.last_user_audio_mono_ns,
                end_mono_ns=end_mono_ns,
                start_wall_time_ns=self.last_user_audio_wall_time_ns,
                end_wall_time_ns=end_wall_time_ns,
                trace_ctx=ctx,
                attrs=attrs,
            )
        return ctx

    def on_end_of_turn_committed(self, *, attrs: dict[str, object] | None = None) -> TraceContext:
        ctx = self.ensure_turn_started()
        commit_mono_ns = now_mono_ns()
        commit_wall_time_ns = now_wall_time_ns()
        if self.last_user_audio_mono_ns is not None:
            if ctx.trace_id is not None and self.last_user_audio_wall_time_ns is not None:
                self._user_last_audio_by_trace_id[ctx.trace_id] = (
                    self.last_user_audio_mono_ns,
                    self.last_user_audio_wall_time_ns,
                )
            self.runtime.emit_event(
                "user_last_audio_frame",
                trace_ctx=ctx,
                mono_ns=self.last_user_audio_mono_ns,
                wall_time_ns=self.last_user_audio_wall_time_ns,
                attrs=attrs,
            )
        self.runtime.emit_event(
            "end_of_turn_committed",
            trace_ctx=ctx,
            mono_ns=commit_mono_ns,
            wall_time_ns=commit_wall_time_ns,
            attrs=attrs,
        )
        if (
            self.last_vad_end_mono_ns is not None
            and self.last_vad_end_wall_time_ns is not None
        ):
            self.runtime.emit_logical_span(
                "endpoint_commit",
                start_mono_ns=self.last_vad_end_mono_ns,
                end_mono_ns=commit_mono_ns,
                start_wall_time_ns=self.last_vad_end_wall_time_ns,
                end_wall_time_ns=commit_wall_time_ns,
                trace_ctx=ctx,
                attrs=attrs,
            )
        return ctx

    def start_response(self, *, response_id: str | None = None) -> TraceContext:
        turn_ctx = self.ensure_turn_started()
        return turn_ctx.with_updates(
            response_id=response_id or ProfilingIdGenerator.response_id(),
            attempt_id=ProfilingIdGenerator.attempt_id(),
        )

    def get_trace_start(self, trace_id: str | None) -> tuple[int, int] | None:
        if trace_id is None:
            return None
        return self._trace_start_by_id.get(trace_id)

    def get_user_last_audio(self, trace_id: str | None) -> tuple[int, int] | None:
        if trace_id is None:
            return None
        return self._user_last_audio_by_trace_id.get(trace_id)

    def finish_trace(self, *, status: str, trace_ctx: TraceContext | None = None) -> None:
        ctx = trace_ctx or self.current_context
        if ctx is None:
            return
        self.runtime.emit_event("trace_finished", trace_ctx=ctx, attrs={"status": status})
        trace_finished(status)
        if ctx.trace_id is not None:
            self._trace_start_by_id.pop(ctx.trace_id, None)
            self._user_last_audio_by_trace_id.pop(ctx.trace_id, None)
        if self.current_context and ctx.trace_id == self.current_context.trace_id:
            self.current_context = None
            self.turn_started_mono_ns = None
            self.turn_started_wall_time_ns = None
            self.first_user_audio_mono_ns = None
            self.first_user_audio_wall_time_ns = None
            self.last_user_audio_mono_ns = None
            self.last_user_audio_wall_time_ns = None
            self.last_vad_end_mono_ns = None
            self.last_vad_end_wall_time_ns = None


@dataclass(slots=True)
class GenerationStepTracker:
    runtime: ProfilingRuntime
    _step_idx_by_response: dict[str, int] = field(default_factory=dict)

    def next_context(self, response_ctx: TraceContext) -> TraceContext:
        response_key = ":".join(
            [
                response_ctx.response_id or "response",
                response_ctx.attempt_id or "attempt",
            ]
        )
        step_idx = self._step_idx_by_response.get(response_key, 0) + 1
        self._step_idx_by_response[response_key] = step_idx
        ctx = response_ctx.with_updates(
            generation_step_id=f"generation_step_{step_idx:04d}",
        )
        self.runtime.emit_event(
            "generation_step_started",
            trace_ctx=ctx,
            attrs={"step_idx": step_idx},
        )
        return ctx
