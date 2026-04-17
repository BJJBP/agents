from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_BENCH_ROOT = Path(
    "/mnt/datasets/FD-Bench/FD-Bench-Audio-Input/Chattts/chattts-single-round-combine-easy"
)
DEFAULT_RUNTIME_LOG_DIR = Path("/NAS/projects/FireRedChat/logs/runtime")
DEFAULT_TRANSCRIPT_ROOT = Path("/NAS/projects/FireRedChat/logs")


@dataclass(slots=True, frozen=True)
class BenchConfig:
    root_dir: Path
    case_timeout_s: float = 180.0
    post_roll_ms: int = 500
    concurrent_sessions: int = 1
    sample_rate: int = 24000
    num_channels: int = 1
    sample_width_bytes: int = 2
    frame_ms: int = 10
    runtime_log_dir: Path = DEFAULT_RUNTIME_LOG_DIR
    transcript_root: Path = DEFAULT_TRANSCRIPT_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_dir", self.root_dir.expanduser().resolve())
        object.__setattr__(
            self, "runtime_log_dir", self.runtime_log_dir.expanduser().resolve()
        )
        object.__setattr__(
            self, "transcript_root", self.transcript_root.expanduser().resolve()
        )

        if self.case_timeout_s <= 0:
            raise ValueError("bench case timeout must be positive")
        if self.post_roll_ms < 0:
            raise ValueError("bench post roll must be >= 0")
        if self.concurrent_sessions <= 0:
            raise ValueError("bench concurrent sessions must be positive")
        if self.sample_rate <= 0 or self.num_channels <= 0 or self.sample_width_bytes <= 0:
            raise ValueError("bench audio config must be positive")
        if self.frame_ms <= 0:
            raise ValueError("bench frame size must be positive")

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def runtime_log_file(self) -> Path:
        return self.runtime_log_dir / "benchplayer.log"
