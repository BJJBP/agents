from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice import io


class BenchAudioInput(io.AudioInput):
    def __init__(
        self,
        wav_path: Path,
        *,
        sample_rate: int,
        num_channels: int,
        sample_width_bytes: int,
        frame_samples: int,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__(label="BenchAudioInput")
        self._wav_path = wav_path
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._sample_width_bytes = sample_width_bytes
        self._frame_samples = frame_samples
        self._loop = loop or asyncio.get_event_loop()
        self._frame_ch = utils.aio.Chan[rtc.AudioFrame](loop=self._loop)
        self._closed = False

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._frame_ch.__anext__()

    async def play(self) -> None:
        samples_sent = 0
        playback_start = time.monotonic()

        try:
            with wave.open(str(self._wav_path), "rb") as wav_file:
                if wav_file.getframerate() != self._sample_rate:
                    raise ValueError(
                        f"unexpected sample rate for {self._wav_path}: {wav_file.getframerate()}"
                    )
                if wav_file.getnchannels() != self._num_channels:
                    raise ValueError(
                        f"unexpected channel count for {self._wav_path}: {wav_file.getnchannels()}"
                    )
                if wav_file.getsampwidth() != self._sample_width_bytes:
                    raise ValueError(
                        f"unexpected sample width for {self._wav_path}: {wav_file.getsampwidth()}"
                    )

                while not self._closed:
                    pcm = wav_file.readframes(self._frame_samples)
                    if not pcm:
                        break

                    samples_per_channel = len(pcm) // (
                        self._sample_width_bytes * self._num_channels
                    )
                    if samples_per_channel <= 0:
                        continue

                    target_at = playback_start + (samples_sent / self._sample_rate)
                    sleep_for = target_at - time.monotonic()
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)

                    await self._frame_ch.send(
                        rtc.AudioFrame(
                            data=pcm,
                            sample_rate=self._sample_rate,
                            num_channels=self._num_channels,
                            samples_per_channel=samples_per_channel,
                        )
                    )
                    samples_sent += samples_per_channel
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._frame_ch.close()

    async def aclose(self) -> None:
        self.close()

    def on_detached(self) -> None:
        self.close()
        super().on_detached()
