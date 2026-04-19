from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from livekit.agents.profiling import (
    ProfilingIdGenerator,
    ProfilingRuntime,
    SharedTTSAdmissionService,
    StandaloneMetricsServer,
)


async def _run() -> None:
    run_id = os.getenv("FIREREDCHAT_RUN_ID") or ProfilingIdGenerator.run_id()
    log_root = os.getenv("FIREREDCHAT_PROFILING_LOG_ROOT", "/NAS/projects/FireRedChat/logs/profiling")
    scheduler_host = os.getenv("FIREREDCHAT_TTS_SCHEDULER_HOST", "127.0.0.1")
    scheduler_port = int(os.getenv("FIREREDCHAT_TTS_SCHEDULER_PORT", "8765"))
    metrics_port = int(os.getenv("FIREREDCHAT_TTS_SCHEDULER_METRICS_PORT", "9103"))
    # max_running_tasks = int(os.getenv("FIREREDCHAT_TTS_SCHEDULER_MAX_RUNNING", "1"))
    max_running_tasks = 4

    runtime = ProfilingRuntime(
        component="tts_scheduler",
        log_root=Path(log_root).expanduser(),
        run_id=run_id,
    )
    service = SharedTTSAdmissionService(
        runtime=runtime,
        host=scheduler_host,
        port=scheduler_port,
        max_running_tasks=max_running_tasks,
    )
    metrics_server = StandaloneMetricsServer(host="127.0.0.1", port=metrics_port)
    await service.start()
    await metrics_server.start()

    stop_event = asyncio.Event()

    def _stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    try:
        await stop_event.wait()
    finally:
        await metrics_server.aclose()
        await service.aclose()
        runtime.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
