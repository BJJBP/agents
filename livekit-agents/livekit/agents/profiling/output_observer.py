from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .chunk import ProfilingAudioChunk
from .clock import now_mono_ns, now_wall_time_ns
from .prometheus import set_session_stall, trace_first_audio
from .runtime import ProfilingRuntime
from .tracker import TurnTracker

_STALL_DEBOUNCE_NS = 100_000_000


@dataclass(slots=True)
class _ResponseState:
    response_start_mono_ns: int | None = None
    response_start_wall_time_ns: int | None = None
    next_playable_ns: int = 0
    last_chunk_playable_ns: int | None = None
    last_chunk_playable_wall_time_ns: int | None = None
    first_audio_playable_ns: int | None = None
    next_playable_wall_time_ns: int = 0
    session_id: str | None = None
    stall_active: bool = False
    stall_start_ns: int | None = None
    stall_start_wall_time_ns: int | None = None
    stall_timer: asyncio.TimerHandle | None = None
    terminal: bool = False
    current_trace_key: str = ""


class UnifiedOutputObserver:
    def __init__(
        self,
        runtime: ProfilingRuntime,
        *,
        turn_tracker: TurnTracker | None = None,
    ) -> None:
        self._runtime = runtime
        self._turn_tracker = turn_tracker
        self._responses: dict[str, _ResponseState] = {}

    def _key_for_chunk(self, chunk: ProfilingAudioChunk) -> str:
        response_id = chunk.trace_ctx.response_id or "response"
        attempt_id = chunk.trace_ctx.attempt_id or "attempt"
        return f"{response_id}:{attempt_id}"

    def _get_state(self, chunk: ProfilingAudioChunk) -> _ResponseState:
        key = self._key_for_chunk(chunk)
        state = self._responses.get(key)
        if state is None:
            state = _ResponseState(current_trace_key=key)
            self._responses[key] = state
        return state

    def on_response_started(
        self,
        *,
        trace_ctx,
        mono_ns: int | None = None,
        wall_time_ns: int | None = None,
    ) -> None:
        if trace_ctx.response_id is None:
            return
        key = f"{trace_ctx.response_id}:{trace_ctx.attempt_id or 'attempt'}"
        state = self._responses.get(key)
        if state is None:
            state = _ResponseState(current_trace_key=key)
            self._responses[key] = state
        state.session_id = trace_ctx.session_id
        state.response_start_mono_ns = mono_ns if mono_ns is not None else now_mono_ns()
        state.response_start_wall_time_ns = (
            wall_time_ns if wall_time_ns is not None else now_wall_time_ns()
        )

    def push_chunk(self, chunk: ProfilingAudioChunk) -> None:
        state = self._get_state(chunk)
        accepted_ns = now_mono_ns()
        accepted_wall_time_ns = now_wall_time_ns()
        self._runtime.emit_event(
            "audio_sink_accepted",
            trace_ctx=chunk.trace_ctx,
            mono_ns=accepted_ns,
            wall_time_ns=accepted_wall_time_ns,
            attrs={"chunk_id": chunk.chunk_id, "chunk_idx": chunk.chunk_idx},
        )

        if state.stall_active:
            self._runtime.emit_logical_span(
                "audio_stall",
                start_mono_ns=state.stall_start_ns or accepted_ns,
                end_mono_ns=accepted_ns,
                start_wall_time_ns=state.stall_start_wall_time_ns or accepted_wall_time_ns,
                end_wall_time_ns=accepted_wall_time_ns,
                trace_ctx=chunk.trace_ctx,
                attrs={},
            )
            state.stall_active = False
            state.stall_start_ns = None
            state.stall_start_wall_time_ns = None
            set_session_stall(state.session_id or chunk.trace_ctx.session_id, False)

        start_ns = max(accepted_ns, state.next_playable_ns or accepted_ns)
        start_wall_time_ns = accepted_wall_time_ns + max(0, start_ns - accepted_ns)
        if state.last_chunk_playable_ns is not None:
            self._runtime.emit_logical_span(
                "tba",
                start_mono_ns=state.last_chunk_playable_ns,
                end_mono_ns=start_ns,
                start_wall_time_ns=state.last_chunk_playable_wall_time_ns or start_wall_time_ns,
                end_wall_time_ns=start_wall_time_ns,
                trace_ctx=chunk.trace_ctx,
                attrs={"previous_chunk_playable_ns": state.last_chunk_playable_ns},
            )
        self._runtime.emit_logical_span(
            "audio_chunk_egress",
            start_mono_ns=chunk.audio_ready_ns,
            end_mono_ns=start_ns,
            start_wall_time_ns=accepted_wall_time_ns,
            end_wall_time_ns=start_wall_time_ns,
            trace_ctx=chunk.trace_ctx,
            attrs={"chunk_id": chunk.chunk_id, "chunk_idx": chunk.chunk_idx},
        )
        state.next_playable_ns = start_ns + chunk.duration_ns
        state.next_playable_wall_time_ns = start_wall_time_ns + chunk.duration_ns
        state.last_chunk_playable_ns = start_ns
        state.last_chunk_playable_wall_time_ns = start_wall_time_ns

        self._runtime.emit_event(
            "audio_chunk_playable",
            trace_ctx=chunk.trace_ctx,
            mono_ns=start_ns,
            wall_time_ns=start_wall_time_ns,
            attrs={"chunk_id": chunk.chunk_id, "chunk_idx": chunk.chunk_idx},
        )
        if state.first_audio_playable_ns is None:
            state.first_audio_playable_ns = start_ns
            if (
                state.response_start_mono_ns is not None
                and state.response_start_wall_time_ns is not None
            ):
                self._runtime.emit_logical_span(
                    "first_audio_egress",
                    start_mono_ns=state.response_start_mono_ns,
                    end_mono_ns=start_ns,
                    start_wall_time_ns=state.response_start_wall_time_ns,
                    end_wall_time_ns=start_wall_time_ns,
                    trace_ctx=chunk.trace_ctx,
                )
            if self._turn_tracker is not None:
                user_last_audio = self._turn_tracker.get_user_last_audio(chunk.trace_ctx.trace_id)
                if user_last_audio is not None:
                    self._runtime.emit_logical_span(
                        "e2e_ttfa",
                        start_mono_ns=user_last_audio[0],
                        end_mono_ns=start_ns,
                        start_wall_time_ns=user_last_audio[1],
                        end_wall_time_ns=start_wall_time_ns,
                        trace_ctx=chunk.trace_ctx,
                    )
            self._runtime.emit_event(
                "first_agent_audio_playable",
                trace_ctx=chunk.trace_ctx,
                mono_ns=start_ns,
                wall_time_ns=start_wall_time_ns,
            )
            trace_first_audio()

        self._arm_stall_timer(state, chunk)

    def _arm_stall_timer(self, state: _ResponseState, chunk: ProfilingAudioChunk) -> None:
        if state.stall_timer is not None:
            state.stall_timer.cancel()

        delay = max((state.next_playable_ns + _STALL_DEBOUNCE_NS - now_mono_ns()) / 1e9, 0.0)
        loop = asyncio.get_running_loop()

        def _open_stall() -> None:
            if state.terminal or state.stall_active:
                return
            state.stall_active = True
            state.stall_start_ns = state.next_playable_ns
            state.stall_start_wall_time_ns = (
                state.next_playable_wall_time_ns or now_wall_time_ns()
            )
            set_session_stall(state.session_id or chunk.trace_ctx.session_id, True)
            self._runtime.emit_event(
                "playback_stall_started",
                trace_ctx=chunk.trace_ctx,
                mono_ns=state.stall_start_ns,
                wall_time_ns=state.stall_start_wall_time_ns,
            )

        state.stall_timer = loop.call_later(delay, _open_stall)

    def on_response_terminal(
        self,
        *,
        trace_ctx,
        interrupted: bool,
    ) -> None:
        if trace_ctx.response_id is None:
            return
        key = f"{trace_ctx.response_id}:{trace_ctx.attempt_id or 'attempt'}"
        state = self._responses.pop(key, None)
        if state is None:
            return

        state.terminal = True
        if state.stall_timer is not None:
            state.stall_timer.cancel()
        if state.stall_active:
            set_session_stall(state.session_id or trace_ctx.session_id, False)
            self._runtime.emit_event(
                "playback_stall_ended",
                trace_ctx=trace_ctx,
                mono_ns=now_mono_ns(),
                wall_time_ns=now_wall_time_ns(),
                attrs={"reason": "response_terminal"},
            )
        else:
            set_session_stall(state.session_id or trace_ctx.session_id, False)
        self._runtime.emit_event(
            "old_agent_audio_stopped",
            trace_ctx=trace_ctx,
            mono_ns=now_mono_ns(),
            wall_time_ns=now_wall_time_ns(),
            attrs={"interrupted": interrupted},
        )

    def close(self) -> None:
        for state in self._responses.values():
            state.terminal = True
            if state.stall_timer is not None:
                state.stall_timer.cancel()
            set_session_stall(state.session_id, False)
        self._responses.clear()
