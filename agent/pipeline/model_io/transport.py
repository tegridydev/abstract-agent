"""Ollama streaming transport isolated from contract and domain validation."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


def _value(value: Any, key: str, default: Any = None) -> Any:
    if hasattr(value, key):
        return getattr(value, key)
    if isinstance(value, dict):
        return value.get(key, default)
    return default


@dataclass(frozen=True)
class StreamResult:
    content: str
    thinking: str
    metrics: dict[str, Any]
    first_token_seconds: float | None


def consume_chat_stream(
    chunks: Iterable[Any],
    *,
    attempt: int,
    progress_callback: Callable[[int, int], None] | None = None,
    started_at: float | None = None,
) -> StreamResult:
    """Accumulate public response fields and bounded timing from one streamed request."""
    request_started = started_at if started_at is not None else time.monotonic()
    first_token_seconds: float | None = None
    content = ""
    thinking = ""
    metrics: dict[str, Any] = {}
    last_progress = 0.0
    for chunk in chunks:
        if first_token_seconds is None:
            first_token_seconds = max(0.0, time.monotonic() - request_started)
        message = _value(chunk, "message", {})
        part = _value(message, "content", "")
        if isinstance(part, str):
            content += part
        thought = _value(message, "thinking", "")
        if isinstance(thought, str):
            thinking += thought
        for key in (
            "model",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        ):
            metric = _value(chunk, key)
            if metric is not None:
                metrics[key] = metric
        now = time.monotonic()
        done = bool(_value(chunk, "done", False))
        if progress_callback and (done or now - last_progress >= 0.5):
            progress_callback(attempt, len(content) + len(thinking))
            last_progress = now
    return StreamResult(content, thinking, metrics, first_token_seconds)


__all__ = ["StreamResult", "consume_chat_stream"]
