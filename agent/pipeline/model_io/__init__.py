"""Model transport, capabilities, and contract telemetry."""

from .capabilities import build_capability_reports
from .schemas import (
    identity_scoped_format_schema,
    ollama_format_schema,
    slotted_relevance_format_schema,
)
from .telemetry import is_structural_compilation_failure
from .transport import StreamResult, consume_chat_stream

__all__ = [
    "StreamResult",
    "build_capability_reports",
    "consume_chat_stream",
    "identity_scoped_format_schema",
    "is_structural_compilation_failure",
    "ollama_format_schema",
    "slotted_relevance_format_schema",
]
