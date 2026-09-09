"""Small, dependency free Ollama embedding adapter."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from typing import Any

from ollama import Client

from ..ollama_models import ollama_http_timeout, validate_model_name


class EmbeddingUnavailableError(RuntimeError):
    """Raised when embeddings cannot safely be used for the current operation."""


def _value(value: Any, key: str, default: Any = None) -> Any:
    if hasattr(value, key):
        return getattr(value, key)
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def normalize_vector(vector: Sequence[float]) -> list[float]:
    converted = [float(item) for item in vector]
    if not converted or not all(math.isfinite(item) for item in converted):
        raise EmbeddingUnavailableError("embedding vector is empty or non finite")
    magnitude = math.sqrt(sum(item * item for item in converted))
    if magnitude <= 0:
        raise EmbeddingUnavailableError("embedding vector has zero magnitude")
    return [item / magnitude for item in converted]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))


def pack_vector(vector: Sequence[float]) -> bytes:
    normalized = normalize_vector(vector)
    return struct.pack(f"<{len(normalized)}f", *normalized)


def unpack_vector(value: bytes, dimensions: int) -> list[float]:
    if dimensions < 1 or len(value) != dimensions * 4:
        raise ValueError("stored embedding dimensions do not match its byte length")
    return list(struct.unpack(f"<{dimensions}f", value))


class OllamaEmbeddingService:
    """Batch embeddings with strict dimensional and numeric validation."""

    def __init__(
        self,
        model: str,
        host: str,
        timeout: float = 60,
        *,
        digest: str = "",
        client: Any | None = None,
    ) -> None:
        self.model = validate_model_name(model)
        self.digest = digest
        self.client = client or Client(host=host.rstrip("/"), timeout=ollama_http_timeout(timeout))
        self.dimensions: int | None = None

    @property
    def space_id(self) -> str:
        return f"{self.model}@{self.digest or 'unknown'}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = [str(text).strip() for text in texts]
        if not values or any(not text for text in values):
            raise EmbeddingUnavailableError("embedding input must contain non empty text")
        try:
            response = self.client.embed(model=self.model, input=values, truncate=False)
            raw_vectors = _value(response, "embeddings", []) or []
            vectors = [normalize_vector(vector) for vector in raw_vectors]
        except EmbeddingUnavailableError:
            raise
        except Exception as error:
            raise EmbeddingUnavailableError(
                f"embedding model '{self.model}' failed: {type(error).__name__}: {error}"
            ) from error
        if len(vectors) != len(values):
            raise EmbeddingUnavailableError(
                f"embedding model returned {len(vectors)} vectors for {len(values)} inputs"
            )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise EmbeddingUnavailableError("embedding model returned inconsistent dimensions")
        dimension = dimensions.pop()
        if self.dimensions is not None and dimension != self.dimensions:
            raise EmbeddingUnavailableError("embedding dimensions changed during the run")
        self.dimensions = dimension
        return vectors

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
