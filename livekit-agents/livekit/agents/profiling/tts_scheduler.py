from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from aiohttp import web

from .clock import now_mono_ns, now_wall_time_ns
from .prometheus import FIREREDCHAT_TTS_SCHED_RUNNING, FIREREDCHAT_TTS_SCHED_WAITING
from .runtime import ProfilingRuntime


@dataclass(slots=True)
class _QueuedTask:
    task_id: str
    payload: dict[str, Any]
    enqueued_mono_ns: int = field(default_factory=now_mono_ns)
    authorized: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False


class SharedTTSAdmissionService:
    def __init__(
        self,
        *,
        runtime: ProfilingRuntime,
        host: str,
        port: int,
        max_running_tasks: int = 1,
    ) -> None:
        self._runtime = runtime
        self._host = host
        self._port = port
        self._max_running_tasks = max_running_tasks
        self._queue: deque[_QueuedTask] = deque()
        self._tasks: dict[str, _QueuedTask] = {}
        self._running: set[str] = set()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._runtime.write_clock_anchor()
        app = web.Application()
        app.add_routes(
            [
                web.post("/submit", self._submit),
                web.post("/wait", self._wait),
                web.post("/cancel", self._cancel),
                web.post("/cancel_response", self._cancel_response),
                web.post("/terminal", self._terminal),
            ]
        )
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

    async def aclose(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _submit(self, request: web.Request) -> web.Response:
        payload = await request.json()
        task = _QueuedTask(task_id=payload["task_id"], payload=payload)
        async with self._lock:
            self._tasks[task.task_id] = task
            self._queue.append(task)
            FIREREDCHAT_TTS_SCHED_WAITING.inc()
            self._runtime.emit_event("tts_task_enqueued", attrs=payload)
            self._pump_queue_locked()
        return web.json_response({"ok": True})

    async def _wait(self, request: web.Request) -> web.Response:
        payload = await request.json()
        task = self._tasks[payload["task_id"]]
        await task.authorized.wait()
        return web.json_response(
            {
                "ok": True,
                "authorized_mono_ns": now_mono_ns(),
                "authorized_wall_time_ns": now_wall_time_ns(),
            }
        )

    async def _cancel(self, request: web.Request) -> web.Response:
        payload = await request.json()
        task_id = payload["task_id"]
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                task.cancelled = True
                if task_id in self._running:
                    self._running.discard(task_id)
                    FIREREDCHAT_TTS_SCHED_RUNNING.dec()
                self._runtime.emit_event("tts_task_cancelled", attrs={"task_id": task_id})
                self._pump_queue_locked()
        return web.json_response({"ok": True})

    async def _cancel_response(self, request: web.Request) -> web.Response:
        payload = await request.json()
        response_id = payload.get("response_id")
        attempt_id = payload.get("attempt_id")
        async with self._lock:
            cancelled_task_ids: list[str] = []
            for task_id, task in list(self._tasks.items()):
                if task.payload.get("response_id") != response_id:
                    continue
                if attempt_id is not None and task.payload.get("attempt_id") != attempt_id:
                    continue
                task.cancelled = True
                cancelled_task_ids.append(task_id)
                if task_id in self._running:
                    self._running.discard(task_id)
                    FIREREDCHAT_TTS_SCHED_RUNNING.dec()
            if cancelled_task_ids:
                self._runtime.emit_event(
                    "tts_response_cancelled",
                    attrs={
                        "response_id": response_id,
                        "attempt_id": attempt_id,
                        "task_ids": cancelled_task_ids,
                    },
                )
            self._pump_queue_locked()
        return web.json_response({"ok": True})

    async def _terminal(self, request: web.Request) -> web.Response:
        payload = await request.json()
        task_id = payload["task_id"]
        async with self._lock:
            if task_id in self._running:
                self._running.discard(task_id)
                FIREREDCHAT_TTS_SCHED_RUNNING.dec()
            self._tasks.pop(task_id, None)
            self._runtime.emit_event("tts_task_terminal", attrs=payload)
            self._pump_queue_locked()
        return web.json_response({"ok": True})

    def _pump_queue_locked(self) -> None:
        while self._queue and len(self._running) < self._max_running_tasks:
            task = self._queue.popleft()
            FIREREDCHAT_TTS_SCHED_WAITING.dec()
            if task.cancelled:
                self._tasks.pop(task.task_id, None)
                continue
            self._running.add(task.task_id)
            FIREREDCHAT_TTS_SCHED_RUNNING.inc()
            task.authorized.set()
            self._runtime.emit_event("tts_task_authorized", attrs=task.payload)


class SharedTTSAdmissionClient:
    def __init__(self, endpoint: str | None) -> None:
        self._endpoint = endpoint.rstrip("/") if endpoint else None
        self._session: aiohttp.ClientSession | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._endpoint)

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def submit(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        async with self._ensure_session().post(f"{self._endpoint}/submit", json=payload) as resp:
            resp.raise_for_status()

    async def wait(self, task_id: str) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": True}
        async with self._ensure_session().post(
            f"{self._endpoint}/wait",
            json={"task_id": task_id},
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def cancel(self, task_id: str) -> None:
        if not self.enabled:
            return
        async with self._ensure_session().post(
            f"{self._endpoint}/cancel",
            json={"task_id": task_id},
        ) as resp:
            resp.raise_for_status()

    async def mark_terminal(self, task_id: str, status: str) -> None:
        if not self.enabled:
            return
        async with self._ensure_session().post(
            f"{self._endpoint}/terminal",
            json={"task_id": task_id, "status": status},
        ) as resp:
            resp.raise_for_status()

    async def cancel_response(self, response_id: str | None, attempt_id: str | None) -> None:
        if not self.enabled or response_id is None:
            return
        async with self._ensure_session().post(
            f"{self._endpoint}/cancel_response",
            json={"response_id": response_id, "attempt_id": attempt_id},
        ) as resp:
            resp.raise_for_status()

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
