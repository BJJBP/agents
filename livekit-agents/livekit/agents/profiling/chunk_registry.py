from __future__ import annotations

from dataclasses import dataclass, field

from .chunk import ProfilingAudioChunk


@dataclass(slots=True)
class ChunkRegistry:
    _chunks: dict[str, ProfilingAudioChunk] = field(default_factory=dict)

    def register(self, chunk: ProfilingAudioChunk) -> None:
        self._chunks[chunk.chunk_id] = chunk

    def get(self, chunk_id: str) -> ProfilingAudioChunk | None:
        return self._chunks.get(chunk_id)

    def pop(self, chunk_id: str) -> ProfilingAudioChunk | None:
        return self._chunks.pop(chunk_id, None)
