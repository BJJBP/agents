from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(slots=True)
class ProfilingIdGenerator:
    @staticmethod
    def run_id() -> str:
        return _new_id("run")

    @staticmethod
    def session_id() -> str:
        return _new_id("session")

    @staticmethod
    def turn_id() -> str:
        return _new_id("turn")

    @staticmethod
    def trace_id() -> str:
        return _new_id("trace")

    @staticmethod
    def response_id() -> str:
        return _new_id("response")

    @staticmethod
    def attempt_id() -> str:
        return _new_id("attempt")

    @staticmethod
    def generation_step_id() -> str:
        return _new_id("generation_step")

    @staticmethod
    def request_id() -> str:
        return _new_id("request")

    @staticmethod
    def task_id() -> str:
        return _new_id("task")

    @staticmethod
    def chunk_id() -> str:
        return _new_id("chunk")
