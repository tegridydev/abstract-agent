"""Dependency inversion ports used by the pipeline core."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol


class CompletionPort(Protocol):
    def ask(self, prompt: str, **kwargs: Any) -> tuple[Any, dict[str, Any]]: ...

    def draft(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]: ...

    def close(self) -> None: ...


class EmbeddingPort(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def close(self) -> None: ...


class ToolResolver(Protocol):
    def __call__(self, name: str) -> Callable[..., Any]: ...


class EventSink(Protocol):
    def emit(self, event: Any) -> None: ...


__all__ = ["CompletionPort", "EmbeddingPort", "EventSink", "ToolResolver"]
