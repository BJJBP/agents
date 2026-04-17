from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import DEFAULT_RUNTIME_LOG_DIR

LOGGER_NAME = "fireredchat.benchplayer"


def get_bench_logger(log_dir: Path = DEFAULT_RUNTIME_LOG_DIR) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, "_benchplayer_configured", False):
        return logger

    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "benchplayer.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    setattr(logger, "_benchplayer_configured", True)
    return logger


def _emit(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def log_run_started(
    logger: logging.Logger,
    *,
    run_id: str,
    root_dir: Path,
    case_count: int,
    concurrent_sessions: int,
) -> None:
    _emit(
        logger,
        "run_started",
        run_id=run_id,
        root_dir=str(root_dir),
        case_count=case_count,
        concurrent_sessions=concurrent_sessions,
    )


def log_run_finished(
    logger: logging.Logger,
    *,
    run_id: str,
    total_cases: int,
    finished_cases: int,
    failed_cases: int,
) -> None:
    _emit(
        logger,
        "run_finished",
        run_id=run_id,
        total_cases=total_cases,
        finished_cases=finished_cases,
        failed_cases=failed_cases,
    )


def log_session_started(
    logger: logging.Logger,
    *,
    run_id: str,
    session_id: str,
    wav_path: Path,
    relative_path: Path,
    duration_s: float,
) -> None:
    _emit(
        logger,
        "session_started",
        run_id=run_id,
        session_id=session_id,
        wav_path=str(wav_path),
        relative_path=str(relative_path),
        duration_s=round(duration_s, 3),
        status="started",
        playback_position=0.0,
    )


def log_session_finished(
    logger: logging.Logger,
    *,
    run_id: str,
    session_id: str,
    wav_path: Path,
    relative_path: Path,
    duration_s: float,
    status: str,
    playback_position: float,
) -> None:
    _emit(
        logger,
        "session_finished",
        run_id=run_id,
        session_id=session_id,
        wav_path=str(wav_path),
        relative_path=str(relative_path),
        duration_s=round(duration_s, 3),
        status=status,
        playback_position=round(playback_position, 3),
    )


def log_session_error(
    logger: logging.Logger,
    *,
    run_id: str,
    session_id: str,
    wav_path: Path,
    relative_path: Path,
    error: str,
) -> None:
    _emit(
        logger,
        "session_error",
        run_id=run_id,
        session_id=session_id,
        wav_path=str(wav_path),
        relative_path=str(relative_path),
        error=error,
    )
