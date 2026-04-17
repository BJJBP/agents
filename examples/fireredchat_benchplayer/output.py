from __future__ import annotations

import asyncio
import time

from livekit import rtc
from livekit.agents import utils
from livekit.agents.log import logger
from livekit.agents.voice import io


class BenchAudioOutputSink(io.AudioOutput):
    def __init__(self, *, sample_rate: int = 24000) -> None:
        super().__init__(
            label="BenchAudioOutputSink",
            next_in_chain=None,
            sample_rate=sample_rate,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self._segment_task: asyncio.Task[None] | None = None
        self._segment_event = asyncio.Event()
        self._interrupted = False
        self._flushed = False
        self._closed = False

        self._capture_start = 0.0
        self._pushed_duration = 0.0
        self._paused_at: float | None = None
        self._paused_duration = 0.0

        self._last_playback_position = 0.0
        self._total_playback_position = 0.0

    @property
    def playback_position(self) -> float:
        return self._total_playback_position

    @property
    def last_playback_position(self) -> float:
        return self._last_playback_position

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

        if self._segment_task and not self._segment_task.done():
            logger.error("capture_frame called while previous flush is in progress")
            await self._segment_task

        if not self._pushed_duration:
            self._capture_start = time.monotonic()
            self._interrupted = False
            self._flushed = False
            self._paused_duration = 0.0
            self._paused_at = None
            self._segment_event.clear()

        self._pushed_duration += frame.duration

    def flush(self) -> None:
        super().flush()
        self._flushed = True
        if not self._pushed_duration:
            return

        if self._segment_task and not self._segment_task.done():
            logger.error("flush called while previous flush is in progress")
            self._segment_task.cancel()

        self._segment_task = asyncio.create_task(self._wait_for_playout())

    def clear_buffer(self) -> None:
        if not self._pushed_duration:
            return

        if not self._flushed:
            super().flush()
            self._flushed = True
            self._finish_segment(interrupted=True)
            return

        self._interrupted = True
        self._segment_event.set()

    def pause(self) -> None:
        super().pause()
        if self._paused_at is None:
            self._paused_at = time.monotonic()
            self._segment_event.set()

    def resume(self) -> None:
        super().resume()
        if self._paused_at is not None:
            self._paused_duration += time.monotonic() - self._paused_at
            self._paused_at = None
            self._segment_event.set()

    async def aclose(self) -> None:
        self._closed = True
        self._interrupted = True
        self._segment_event.set()
        if self._segment_task is not None:
            await utils.aio.cancel_and_wait(self._segment_task)

    async def _wait_for_playout(self) -> None:
        while True:
            if self._interrupted:
                self._finish_segment(interrupted=True)
                return

            if self._paused_at is not None:
                self._segment_event.clear()
                await self._segment_event.wait()
                continue

            deadline = self._capture_start + self._pushed_duration + self._paused_duration
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._finish_segment(interrupted=False)
                return

            self._segment_event.clear()
            try:
                await asyncio.wait_for(self._segment_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                self._finish_segment(interrupted=False)
                return

    def _finish_segment(self, *, interrupted: bool) -> None:
        played_duration = self._pushed_duration
        if interrupted:
            played_duration = time.monotonic() - self._capture_start - self._paused_duration
            if self._paused_at is not None:
                played_duration -= time.monotonic() - self._paused_at
            played_duration = min(max(played_duration, 0.0), self._pushed_duration)

        self._last_playback_position = played_duration
        self._total_playback_position += played_duration
        self._segment_event.clear()
        self._interrupted = False
        self._flushed = False
        self._capture_start = 0.0
        self._pushed_duration = 0.0
        self._paused_at = None
        self._paused_duration = 0.0
        self.on_playback_finished(playback_position=played_duration, interrupted=interrupted)
