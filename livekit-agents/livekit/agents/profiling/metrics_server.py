from __future__ import annotations

import asyncio

from aiohttp import web
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess


class StandaloneMetricsServer:
    def __init__(self, *, host: str, port: int, multiprocess_dir: str | None = None) -> None:
        self._host = host
        self._port = port
        self._multiprocess_dir = multiprocess_dir
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    async def start(self) -> None:
        app = web.Application()
        app.add_routes([web.get("/metrics", self._metrics)])
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

    async def aclose(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _metrics(self, _request: web.Request) -> web.Response:
        loop = asyncio.get_running_loop()

        def _collect() -> bytes:
            if self._multiprocess_dir:
                registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(registry)
                return generate_latest(registry)
            return generate_latest()

        data = await loop.run_in_executor(None, _collect)
        return web.Response(
            body=data,
            headers={"Content-Type": CONTENT_TYPE_LATEST, "Content-Length": str(len(data))},
        )
