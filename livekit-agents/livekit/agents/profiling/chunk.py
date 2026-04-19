from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from livekit import rtc

from .context import TraceContext

if TYPE_CHECKING:
    from ..voice.io import TimedString


@dataclass(slots=True)
class ProfilingAudioChunk:
    chunk_id: str
    chunk_idx: int
    trace_ctx: TraceContext
    frames: list[rtc.AudioFrame]
    audio_ready_ns: int
    timed_texts: list[TimedString] = field(default_factory=list)
    task_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return sum(frame.duration for frame in self.frames)

    @property
    def duration_ns(self) -> int:
        return int(self.duration * 1e9)
