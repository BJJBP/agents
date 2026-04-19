from __future__ import annotations

import asyncio
import multiprocessing as mp
from datetime import datetime
from typing import Any
from uuid import uuid4

from livekit.agents.profiling import StandaloneMetricsServer
from livekit.agents.inference_runner import _InferenceRunner
from livekit.agents.ipc.inference_proc_executor import InferenceProcExecutor

from fireredchat_app import (
    attach_session_text_logging,
    build_agent,
    build_profiling_config,
    build_profiling_userdata,
    build_session,
    close_profiling_userdata,
    load_vad,
    write_session_transcript,
)

from .config import BenchConfig
from .dataset import BenchCase, scan_cases
from .input import BenchAudioInput
from .logs import (
    get_bench_logger,
    log_run_finished,
    log_run_started,
    log_session_error,
    log_session_finished,
    log_session_started,
)
from .output import BenchAudioOutputSink


def _build_run_id() -> str:
    return datetime.utcnow().strftime("bench_%Y%m%d_%H%M%S")


def _build_session_id(case_index: int) -> str:
    return f"bench_session_{case_index:06d}_{uuid4().hex[:8]}"


async def run_bench(config: BenchConfig) -> None:
    cases = scan_cases(config)
    logger = get_bench_logger(config.runtime_log_dir)
    run_id = _build_run_id()
    inference_executor = await _create_inference_executor()
    metrics_server: StandaloneMetricsServer | None = None
    worker_count = min(config.concurrent_sessions, len(cases))

    try:
        if config.profiling_enabled:
            metrics_server = StandaloneMetricsServer(
                host="127.0.0.1",
                port=config.bench_metrics_port,
            )
            await metrics_server.start()

        log_run_started(
            logger,
            run_id=run_id,
            root_dir=config.root_dir,
            case_count=len(cases),
            concurrent_sessions=config.concurrent_sessions,
        )

        case_queue: asyncio.Queue[tuple[int, BenchCase] | None] = asyncio.Queue()
        for index, case in enumerate(cases, start=1):
            case_queue.put_nowait((index, case))
        for _ in range(worker_count):
            case_queue.put_nowait(None)

        worker_tasks = [
            asyncio.create_task(
                _run_case_worker(
                    config=config,
                    run_id=run_id,
                    case_queue=case_queue,
                    inference_executor=inference_executor,
                    logger=logger,
                ),
                name=f"bench-case-worker-{worker_index + 1}",
            )
            for worker_index in range(worker_count)
        ]
        worker_results = await asyncio.gather(*worker_tasks)

        finished_cases = sum(result[0] for result in worker_results)
        failed_cases = sum(result[1] for result in worker_results)
        log_run_finished(
            logger,
            run_id=run_id,
            total_cases=len(cases),
            finished_cases=finished_cases,
            failed_cases=failed_cases,
        )
    finally:
        if metrics_server is not None:
            await metrics_server.aclose()
        await inference_executor.aclose()


async def _run_case_worker(
    *,
    config: BenchConfig,
    run_id: str,
    case_queue: asyncio.Queue[tuple[int, BenchCase] | None],
    inference_executor: InferenceProcExecutor,
    logger: Any,
) -> tuple[int, int]:
    finished_cases = 0
    failed_cases = 0

    while True:
        case_item = await case_queue.get()
        if case_item is None:
            return finished_cases, failed_cases

        case_index, case = case_item
        status = await _run_case(
            config=config,
            run_id=run_id,
            case=case,
            case_index=case_index,
            inference_executor=inference_executor,
            logger=logger,
        )
        finished_cases += 1
        if status != "ok":
            failed_cases += 1


async def _run_case(
    *,
    config: BenchConfig,
    run_id: str,
    case: BenchCase,
    case_index: int,
    inference_executor: InferenceProcExecutor,
    logger: Any,
) -> str:
    session_id = _build_session_id(case_index)
    session: Any | None = None
    audio_input: BenchAudioInput | None = None
    audio_output: BenchAudioOutputSink | None = None

    status = "error"
    try:
        # pVAD keeps mutable inference buffers on the model instance, so each case needs
        # its own VAD object to avoid cross-session state bleed in concurrent bench runs.
        vad = await asyncio.to_thread(load_vad)
        profiling_userdata = build_profiling_userdata(
            build_profiling_config(
                run_id=run_id,
                session_id=session_id,
                enabled=config.profiling_enabled,
                log_root=config.profiling_log_root,
                scheduler_endpoint=config.scheduler_endpoint,
            )
        )
        session = build_session(
            vad,
            userdata={
                **profiling_userdata,
                "run_id": run_id,
                "session_id": session_id,
                "wav_path": str(case.wav_path),
            },
            turn_inference_executor=inference_executor,
        )
        attach_session_text_logging(session, mode="bench")
        agent = build_agent()
        audio_input = BenchAudioInput(
            case.wav_path,
            sample_rate=config.sample_rate,
            num_channels=config.num_channels,
            sample_width_bytes=config.sample_width_bytes,
            frame_samples=config.frame_samples,
        )
        audio_output = BenchAudioOutputSink(sample_rate=config.sample_rate)

        session.input.audio = audio_input
        session.output.audio = audio_output

        await asyncio.wait_for(
            _execute_case(
                session=session,
                agent=agent,
                audio_input=audio_input,
                audio_output=audio_output,
                run_id=run_id,
                session_id=session_id,
                case=case,
                logger=logger,
                post_roll_ms=config.post_roll_ms,
            ),
            timeout=config.case_timeout_s,
        )
        status = "ok"
    except asyncio.TimeoutError:
        status = "timeout"
        log_session_error(
            logger,
            run_id=run_id,
            session_id=session_id,
            wav_path=case.wav_path,
            relative_path=case.relative_path,
            error=f"case timed out after {config.case_timeout_s:.1f}s",
        )
    except Exception as exc:
        log_session_error(
            logger,
            run_id=run_id,
            session_id=session_id,
            wav_path=case.wav_path,
            relative_path=case.relative_path,
            error=f"{type(exc).__name__}: {exc}",
        )
        status = "error"
    finally:
        if session is not None:
            try:
                write_session_transcript(session, session_id, root_dir=config.transcript_root)
            except Exception as exc:
                log_session_error(
                    logger,
                    run_id=run_id,
                    session_id=session_id,
                    wav_path=case.wav_path,
                    relative_path=case.relative_path,
                    error=f"failed to write transcript: {type(exc).__name__}: {exc}",
                )

        if session is not None or audio_input is not None or audio_output is not None:
            try:
                await _close_case(audio_input=audio_input, audio_output=audio_output, session=session)
                if session is not None:
                    await close_profiling_userdata(session.userdata)
            except Exception as exc:
                log_session_error(
                    logger,
                    run_id=run_id,
                    session_id=session_id,
                    wav_path=case.wav_path,
                    relative_path=case.relative_path,
                    error=f"failed to close case: {type(exc).__name__}: {exc}",
                )
                status = "error"

        log_session_finished(
            logger,
            run_id=run_id,
            session_id=session_id,
            wav_path=case.wav_path,
            relative_path=case.relative_path,
            duration_s=case.duration_s,
            status=status,
            playback_position=audio_output.playback_position if audio_output is not None else 0.0,
        )

    return status


async def _execute_case(
    *,
    session: Any,
    agent: Any,
    audio_input: BenchAudioInput,
    audio_output: BenchAudioOutputSink,
    run_id: str,
    session_id: str,
    case: BenchCase,
    logger: Any,
    post_roll_ms: int,
) -> None:
    await session.start(agent=agent)

    log_session_started(
        logger,
        run_id=run_id,
        session_id=session_id,
        wav_path=case.wav_path,
        relative_path=case.relative_path,
        duration_s=case.duration_s,
    )

    await audio_input.play()
    session.input.set_audio_enabled(False)
    session.commit_user_turn(transcript_timeout=10.0)
    await session.drain()
    await audio_output.wait_for_playout()
    if post_roll_ms > 0:
        await asyncio.sleep(post_roll_ms / 1000)


async def _close_case(
    *,
    audio_input: BenchAudioInput | None,
    audio_output: BenchAudioOutputSink | None,
    session: Any | None,
) -> None:
    try:
        if audio_input is not None:
            await audio_input.aclose()
    finally:
        try:
            if audio_output is not None:
                await audio_output.aclose()
        finally:
            if session is not None:
                await session.aclose()


async def _create_inference_executor() -> InferenceProcExecutor:
    loop = asyncio.get_running_loop()
    executor = InferenceProcExecutor(
        runners=_InferenceRunner.registered_runners,
        initialize_timeout=45.0,
        close_timeout=5.0,
        memory_warn_mb=2000,
        memory_limit_mb=0,
        ping_interval=5.0,
        ping_timeout=60.0,
        high_ping_threshold=2.5,
        mp_ctx=mp.get_context("spawn"),
        loop=loop,
        http_proxy=None,
    )
    await executor.start()
    await executor.initialize()
    return executor
