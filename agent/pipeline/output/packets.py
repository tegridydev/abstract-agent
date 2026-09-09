"""Compact, injection resistant rendering of dynamic prompt inputs."""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from typing import Any

DEFAULT_DYNAMIC_BUDGET = 14_000
PER_VALUE_BUDGET = 8_000


@dataclass(frozen=True)
class PromptPacket:
    values: dict[str, str]
    prompt_hash: str
    dynamic_characters: int
    truncated_fields: tuple[str, ...]


def _serialize(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return str(value)


def escape_prompt_data(value: Any) -> str:
    """Escape markup delimiters so retrieved text cannot close the data boundary."""
    return html.escape(_serialize(value), quote=False)


def build_prompt_packet(
    input_keys: list[str],
    context: dict[str, Any],
    *,
    dynamic_budget: int = DEFAULT_DYNAMIC_BUDGET,
) -> PromptPacket:
    remaining = max(512, dynamic_budget)
    rendered: dict[str, str] = {}
    truncated: list[str] = []
    for key in input_keys:
        value = context[key]
        text = escape_prompt_data(value)
        allowance = min(PER_VALUE_BUDGET, remaining)
        if len(text) > allowance:
            if isinstance(value, (dict, list)):
                text = escape_prompt_data(
                    {
                        "omitted": True,
                        "value_type": type(value).__name__,
                        "item_count": len(value),
                    }
                )
            else:
                text = text[: max(0, allowance - 20)].rstrip() + "…[truncated]"
            truncated.append(key)
        rendered[key] = text
        remaining = max(0, remaining - len(text))
    canonical = json.dumps(rendered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return PromptPacket(
        values=rendered,
        prompt_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        dynamic_characters=sum(map(len, rendered.values())),
        truncated_fields=tuple(truncated),
    )


__all__ = ["PromptPacket", "build_prompt_packet", "escape_prompt_data"]
