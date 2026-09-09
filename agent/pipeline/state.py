"""Atomic context updates for pipeline step execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    """Mutable run state whose step outputs are committed as one validated unit."""

    context: dict[str, Any] = field(default_factory=dict)

    def values_for(self, keys: list[str]) -> dict[str, Any]:
        missing = [key for key in keys if key not in self.context]
        if missing:
            raise ValueError(f"pipeline step is missing context inputs: {missing}")
        return {key: self.context[key] for key in keys}

    def commit(self, output_keys: list[str], values: Any) -> None:
        if len(output_keys) == 1:
            pending = {output_keys[0]: values}
        else:
            if not isinstance(values, tuple | list) or len(values) != len(output_keys):
                raise ValueError(
                    f"step returned {len(values) if isinstance(values, tuple | list) else 1} "
                    f"values for {len(output_keys)} outputs"
                )
            pending = dict(zip(output_keys, values, strict=True))
        collisions = set(pending) & set(self.context)
        if collisions:
            raise ValueError(f"pipeline output would overwrite context keys: {sorted(collisions)}")
        self.context.update(pending)
