from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BenchConfig

_TOKEN_SPLIT_RE = re.compile(r"(\d+)")


@dataclass(slots=True, frozen=True)
class BenchCase:
    wav_path: Path
    relative_path: Path
    duration_s: float


def _natural_tokens(text: str) -> tuple[tuple[int, Any], ...]:
    tokens: list[tuple[int, Any]] = []
    for token in _TOKEN_SPLIT_RE.split(text):
        if not token:
            continue
        if token.isdigit():
            tokens.append((0, int(token)))
        else:
            tokens.append((1, token.lower()))
    return tuple(tokens)


def natural_sort_key(path: Path) -> tuple[tuple[tuple[int, Any], ...], ...]:
    return tuple(_natural_tokens(part) for part in path.parts)


def _read_wav_header(wav_path: Path) -> tuple[int, int, int, int, str]:
    with wave.open(str(wav_path), "rb") as wav_file:
        return (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getnframes(),
            wav_file.getcomptype(),
        )


def scan_cases(config: BenchConfig) -> list[BenchCase]:
    if not config.root_dir.exists():
        raise FileNotFoundError(f"bench root does not exist: {config.root_dir}")
    if not config.root_dir.is_dir():
        raise NotADirectoryError(f"bench root is not a directory: {config.root_dir}")

    wav_paths = sorted(
        (
            path
            for path in config.root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        ),
        key=lambda path: natural_sort_key(path.relative_to(config.root_dir)),
    )

    if not wav_paths:
        raise ValueError(f"no .wav files found under bench root: {config.root_dir}")

    cases: list[BenchCase] = []
    for wav_path in wav_paths:
        sample_rate, num_channels, sample_width, num_frames, comp_type = _read_wav_header(wav_path)
        if comp_type != "NONE":
            raise ValueError(f"unsupported wav compression for {wav_path}: {comp_type}")
        if sample_rate != config.sample_rate:
            raise ValueError(
                f"unsupported sample rate for {wav_path}: {sample_rate}, expected {config.sample_rate}"
            )
        if num_channels != config.num_channels:
            raise ValueError(
                f"unsupported channel count for {wav_path}: {num_channels}, expected {config.num_channels}"
            )
        if sample_width != config.sample_width_bytes:
            raise ValueError(
                f"unsupported sample width for {wav_path}: {sample_width}, expected {config.sample_width_bytes}"
            )

        duration_s = num_frames / sample_rate if sample_rate else 0.0
        cases.append(
            BenchCase(
                wav_path=wav_path.resolve(),
                relative_path=wav_path.relative_to(config.root_dir),
                duration_s=duration_s,
            )
        )

    return cases
