"""Compatibility facade for pipeline context lifecycle helpers."""

try:
    from .pipeline.context import prepare_pipeline_context
except ImportError:  # pragma: no cover - direct script execution
    from pipeline.context import prepare_pipeline_context

__all__ = ["prepare_pipeline_context"]
