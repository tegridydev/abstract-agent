"""Typed pipeline failures with explicit recovery boundaries."""

from __future__ import annotations

from typing import Any


class PipelineError(RuntimeError):
    """Base class for failures surfaced by pipeline execution."""


class PipelineInvariantError(PipelineError):
    """Trusted application data violated an internal invariant."""


class InsufficientEvidenceError(PipelineError):
    """The source backed corpus cannot satisfy the configured evidence floor."""


class ModelRuntimeError(PipelineError):
    """Base failure for a completion model interaction."""

    def __init__(self, message: str, *, diagnostic: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic or {}


class ModelOutputError(PipelineError):
    """A responsive model did not produce a valid application contract."""

    def __init__(self, message: str, *, diagnostic: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic or {}


class ModelTransportError(ModelRuntimeError):
    """The model endpoint was unavailable, timed out, or rejected the request."""


# Historical transport name remains available while callers adopt ``ModelTransportError``.
OllamaRuntimeError = ModelTransportError


__all__ = [
    "InsufficientEvidenceError",
    "ModelOutputError",
    "ModelRuntimeError",
    "ModelTransportError",
    "OllamaRuntimeError",
    "PipelineError",
    "PipelineInvariantError",
]
