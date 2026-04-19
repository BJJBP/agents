from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import Any

from .. import tokenize, utils
from ..profiling.clock import now_mono_ns, now_wall_time_ns
from ..profiling.context import current_trace_context, use_trace_context
from ..profiling.ids import ProfilingIdGenerator
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from .stream_pacer import SentenceStreamPacer
from .tts import (
    TTS,
    AudioEmitter,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
)

# already a retry mechanism in TTS.synthesize, don't retry in stream adapter
DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS = APIConnectOptions(
    max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout
)


class StreamAdapter(TTS):
    def __init__(
        self,
        *,
        tts: TTS,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        text_pacing: SentenceStreamPacer | bool = False,
    ) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=True, aligned_transcript=True),
            sample_rate=tts.sample_rate,
            num_channels=tts.num_channels,
        )
        self._wrapped_tts = tts
        self._sentence_tokenizer = sentence_tokenizer or tokenize.blingfire.SentenceTokenizer(
            retain_format=True
        )
        self._stream_pacer: SentenceStreamPacer | None = None
        if text_pacing is True:
            self._stream_pacer = SentenceStreamPacer()
        elif isinstance(text_pacing, SentenceStreamPacer):
            self._stream_pacer = text_pacing

        @self._wrapped_tts.on("metrics_collected")
        def _forward_metrics(*args: Any, **kwargs: Any) -> None:
            # TODO(theomonnom): The segment_id needs to be populated!
            self.emit("metrics_collected", *args, **kwargs)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return self._wrapped_tts.synthesize(text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> StreamAdapterWrapper:
        return StreamAdapterWrapper(tts=self, conn_options=conn_options)

    def prewarm(self) -> None:
        self._wrapped_tts.prewarm()


class StreamAdapterWrapper(SynthesizeStream):
    def __init__(self, *, tts: StreamAdapter, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=DEFAULT_STREAM_ADAPTER_API_CONNECT_OPTIONS)
        self._tts: StreamAdapter = tts
        self._wrapped_tts_conn_options = conn_options

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[SynthesizedAudio]) -> None:
        pass  # do nothing

    async def _run(self, output_emitter: AudioEmitter) -> None:
        sent_stream = self._tts._sentence_tokenizer.stream()
        if self._tts._stream_pacer:
            sent_stream = self._tts._stream_pacer.wrap(
                sent_stream=sent_stream,
                audio_emitter=output_emitter,
            )

        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=True,
        )

        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        async def _forward_input() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue

                sent_stream.push_text(data)

            sent_stream.end_input()

        async def _synthesize() -> None:
            from ..voice.io import TimedString

            duration = 0.0
            chunk_idx = 0
            scheduler_client = getattr(
                self._tts._wrapped_tts,
                "_fireredchat_scheduler_client",
                getattr(self._tts, "_fireredchat_scheduler_client", None),
            )
            runtime = getattr(
                self._tts._wrapped_tts,
                "_fireredchat_runtime",
                getattr(self._tts, "_fireredchat_runtime", None),
            )
            async for ev in sent_stream:
                output_emitter.push_timed_transcript(
                    TimedString(text=ev.token, start_time=duration)
                )

                if not (text := ev.token.strip()):
                    continue
                chunk_idx += 1
                chunk_ctx = None
                chunk_enqueued_mono_ns = now_mono_ns()
                chunk_enqueued_wall_time_ns = now_wall_time_ns()
                if trace_ctx := current_trace_context():
                    tts_request_id = ":".join(
                        [
                            trace_ctx.trace_id or "trace",
                            trace_ctx.response_id or "response",
                            trace_ctx.attempt_id or "attempt",
                            trace_ctx.generation_step_id or "generation_step",
                            f"chunk_{chunk_idx:04d}",
                            ProfilingIdGenerator.request_id(),
                        ]
                    )
                    chunk_ctx = trace_ctx.with_updates(
                        task_id=ProfilingIdGenerator.task_id(),
                        chunk_idx=chunk_idx,
                        request_id=tts_request_id,
                    )
                    if runtime is not None:
                        runtime.emit_event(
                            "tts_chunk_enqueued",
                            trace_ctx=chunk_ctx,
                            mono_ns=chunk_enqueued_mono_ns,
                            wall_time_ns=chunk_enqueued_wall_time_ns,
                            attrs={"chunk_idx": chunk_idx, "text": text},
                        )

                scheduler_task_submitted = False
                scheduler_task_authorized = False
                if (
                    scheduler_client is not None
                    and getattr(scheduler_client, "enabled", False)
                    and chunk_ctx is not None
                ):
                    try:
                        await scheduler_client.submit(
                            {
                                "task_id": chunk_ctx.task_id,
                                "trace_id": chunk_ctx.trace_id,
                                "response_id": chunk_ctx.response_id,
                                "attempt_id": chunk_ctx.attempt_id,
                                "generation_step_id": chunk_ctx.generation_step_id,
                                "chunk_idx": chunk_idx,
                            }
                        )
                        scheduler_task_submitted = True
                        await scheduler_client.wait(chunk_ctx.task_id)
                        scheduler_task_authorized = True
                        if runtime is not None:
                            runtime.emit_event(
                                "tts_chunk_authorized",
                                trace_ctx=chunk_ctx,
                                mono_ns=now_mono_ns(),
                                wall_time_ns=now_wall_time_ns(),
                                attrs={"chunk_idx": chunk_idx, "task_id": chunk_ctx.task_id},
                            )
                    except asyncio.CancelledError:
                        if scheduler_task_submitted and chunk_ctx.task_id is not None:
                            await scheduler_client.cancel(chunk_ctx.task_id)
                        raise
                    except Exception:
                        if scheduler_task_submitted and chunk_ctx.task_id is not None:
                            await scheduler_client.cancel(chunk_ctx.task_id)
                        raise

                task_status = "ok"
                try:
                    with use_trace_context(chunk_ctx):
                        async with self._tts._wrapped_tts.synthesize(
                            text, conn_options=self._wrapped_tts_conn_options
                        ) as tts_stream:
                            async for audio in tts_stream:
                                output_emitter.push(audio.frame.data.tobytes())
                                duration += audio.frame.duration
                            output_emitter.flush()
                except asyncio.CancelledError:
                    task_status = "cancelled"
                    raise
                except Exception:
                    task_status = "error"
                    raise
                finally:
                    if runtime is not None and chunk_ctx is not None:
                        runtime.emit_logical_span(
                            "tts_chunk_wall",
                            start_mono_ns=chunk_enqueued_mono_ns,
                            end_mono_ns=now_mono_ns(),
                            start_wall_time_ns=chunk_enqueued_wall_time_ns,
                            end_wall_time_ns=now_wall_time_ns(),
                            trace_ctx=chunk_ctx,
                            attrs={
                                "chunk_idx": chunk_idx,
                                "task_id": chunk_ctx.task_id,
                                "scheduler_authorized": scheduler_task_authorized,
                            },
                        )
                    if (
                        scheduler_client is not None
                        and getattr(scheduler_client, "enabled", False)
                        and chunk_ctx is not None
                        and chunk_ctx.task_id is not None
                    ):
                        await scheduler_client.mark_terminal(chunk_ctx.task_id, task_status)

        tasks = [
            asyncio.create_task(_forward_input()),
            asyncio.create_task(_synthesize()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await sent_stream.aclose()
            await utils.aio.cancel_and_wait(*tasks)
