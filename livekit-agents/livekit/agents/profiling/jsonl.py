from __future__ import annotations

import json
from pathlib import Path
from threading import Lock


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        self._lock = Lock()
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    def write(self, payload: dict[str, object]) -> None:
        if self._closed:
            return

        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._file.write(line)
            self._file.write("\n")
            self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._file.flush()
            self._file.close()
            self._closed = True
