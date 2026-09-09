"""Local knowledge storage and deterministic evidence extraction."""

from .embeddings import EmbeddingUnavailableError, OllamaEmbeddingService
from .evidence import EVIDENCE_CONTRACT_VERSION, build_evidence_ledger, canonical_source
from .store import KnowledgeStore

__all__ = [
    "EVIDENCE_CONTRACT_VERSION",
    "EmbeddingUnavailableError",
    "KnowledgeStore",
    "OllamaEmbeddingService",
    "build_evidence_ledger",
    "canonical_source",
]
