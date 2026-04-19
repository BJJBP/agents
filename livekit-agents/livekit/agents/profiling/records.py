from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProfilingRecord:
    record_type: str
    component: str
    fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "record_type": self.record_type,
            "component": self.component,
            **self.fields,
        }
