from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import TYPE_CHECKING

from livekit import rtc

from ..types import USERDATA_TIMED_TRANSCRIPT
from .chunk import ProfilingAudioChunk
from .clock import now_mono_ns
from .context import TraceContext
from .ids import ProfilingIdGenerator

if TYPE_CHECKING:
    from ..voice.io import TimedString

TTS_FRAME_METADATA = "fireredchat.tts.frame_meta"


class TTSChunkAdapter:
    def __init__(self, *, trace_ctx: TraceContext) -> None:
        self._trace_ctx = trace_ctx

    async def adapt(self, tts_output: AsyncIterable[rtc.AudioFrame]) -> AsyncIterator[ProfilingAudioChunk]:
        frames: list[rtc.AudioFrame] = []
        timed_texts: list["TimedString"] = []
        chunk_idx = 0

        async for frame in tts_output:
            frames.append(frame)
            texts = frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
            if texts:
                timed_texts.extend(texts)
                chunk_idx += 1
                yield ProfilingAudioChunk(
                    chunk_id=ProfilingIdGenerator.chunk_id(),
                    chunk_idx=chunk_idx,
                    trace_ctx=self._trace_ctx.with_updates(chunk_idx=chunk_idx),
                    frames=list(frames),
                    audio_ready_ns=now_mono_ns(),
                    timed_texts=list(timed_texts),
                    task_id=frame.userdata.get(TTS_FRAME_METADATA, {}).get("task_id"),
                    attrs=dict(frame.userdata.get(TTS_FRAME_METADATA, {})),
                )
                frames.clear()
                timed_texts.clear()

        if frames:
            chunk_idx += 1
            yield ProfilingAudioChunk(
                chunk_id=ProfilingIdGenerator.chunk_id(),
                chunk_idx=chunk_idx,
                trace_ctx=self._trace_ctx.with_updates(chunk_idx=chunk_idx),
                frames=list(frames),
                audio_ready_ns=now_mono_ns(),
                timed_texts=list(timed_texts),
            )
