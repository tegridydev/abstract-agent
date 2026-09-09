"""Deterministic, source anchored evidence card construction."""

from __future__ import annotations

import hashlib
import html
import math
import re
from typing import Any

from ..contracts.research import EvidenceCard, EvidenceClaim, EvidenceLedger, EvidenceProvenance
from .embeddings import EmbeddingUnavailableError, OllamaEmbeddingService, cosine

EVIDENCE_CONTRACT_VERSION = "1"
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def canonical_source(value: str) -> str:
    """Return stable plain text used for both hashing and exact source offsets."""
    unescaped = html.unescape(_TAG.sub(" ", str(value or "")))
    return " ".join(unescaped.split()).strip()


def source_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_spans(text: str, maximum: int = 360) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for match in _BOUNDARY.finditer(text):
        end = match.start()
        if end > cursor:
            spans.extend(_split_span(text, cursor, end, maximum))
        cursor = match.end()
    if cursor < len(text):
        spans.extend(_split_span(text, cursor, len(text), maximum))
    return [span for span in spans if len(span[2]) >= 12] or _split_span(
        text, 0, len(text), maximum
    )


def _split_span(text: str, start: int, end: int, maximum: int) -> list[tuple[int, int, str]]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    result = []
    while end - start > maximum:
        cut = text.rfind(" ", start, start + maximum + 1)
        if cut <= start:
            cut = start + maximum
        result.append((start, cut, text[start:cut]))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    if end > start:
        result.append((start, end, text[start:end]))
    return result


def _tokens(value: str) -> set[str]:
    return {match.group(0).casefold() for match in _WORD.finditer(value)}


def _lexical_score(query: set[str], passage: str) -> float:
    passage_tokens = _tokens(passage)
    if not query or not passage_tokens:
        return 0.0
    overlap = len(query & passage_tokens)
    return overlap / math.sqrt(len(query) * len(passage_tokens))


def _rank_spans(
    query: str,
    spans: list[tuple[int, int, str]],
    embedding_service: OllamaEmbeddingService | None,
) -> tuple[list[int], str]:
    lexical = [_lexical_score(_tokens(query), passage) for _, _, passage in spans]
    semantic: list[float] | None = None
    mode = "lexical_extractive"
    if embedding_service is not None:
        try:
            vectors = embedding_service.embed([query, *(passage for _, _, passage in spans)])
            semantic = [cosine(vectors[0], vector) for vector in vectors[1:]]
            mode = "embedding_extractive"
        except EmbeddingUnavailableError:
            semantic = None
    scores = [
        (0.75 * semantic[index] + 0.25 * lexical[index]) if semantic is not None else lexical[index]
        for index in range(len(spans))
    ]
    first = max(range(len(spans)), key=lambda index: (scores[index], -index))
    selected = [first]
    if len(spans) > 1:
        first_tokens = _tokens(spans[first][2])

        def mmr(index: int) -> tuple[float, int]:
            candidate = _tokens(spans[index][2])
            union = first_tokens | candidate
            similarity = len(first_tokens & candidate) / len(union) if union else 0.0
            return 0.8 * scores[index] - 0.2 * similarity, -index

        second = max((index for index in range(len(spans)) if index != first), key=mmr)
        if scores[second] > 0 or scores[first] == 0:
            selected.append(second)
    return selected, mode


def _relevance(paper: dict[str, Any], topic: str) -> str:
    judgement = paper.get("relevance_judgement")
    if isinstance(judgement, dict):
        rationale = str(judgement.get("rationale") or "").strip()
        if rationale:
            return rationale[:240]
    return f"Retrieved as evidence relevant to {topic.strip() or 'the research topic'}."[:240]


def build_evidence_ledger(
    top_papers: list[dict[str, Any]],
    topic: str,
    research_scope: dict[str, Any] | None = None,
    *,
    embedding_service: OllamaEmbeddingService | None = None,
    knowledge_store: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one valid evidence card per paper without asking a completion model."""
    if not top_papers:
        raise ValueError("cannot build an evidence ledger without papers")
    scope_label = str((research_scope or {}).get("label") or "").strip()
    query = " ".join(part for part in (topic.strip(), scope_label) if part)
    cards: list[EvidenceCard] = []
    reused = 0
    modes: dict[str, int] = {}
    for paper in top_papers:
        paper_id = str(paper.get("paper_id") or "").strip()
        if not paper_id:
            raise ValueError("every admitted paper requires a canonical paper_id")
        source = canonical_source(str(paper.get("summary") or ""))
        source_field = "summary"
        if not source:
            source = canonical_source(str(paper.get("title") or "Untitled source"))
            source_field = "title"
        digest = source_hash(source)
        stored = None
        if knowledge_store is not None:
            stored = knowledge_store.get_reusable_evidence(
                paper_id, digest, EVIDENCE_CONTRACT_VERSION
            )
        if stored is not None:
            provenance = {
                **stored.get("provenance", {}),
                "reused": True,
            }
            card = EvidenceCard.model_validate({**stored, "provenance": provenance})
            for finding in card.findings:
                if source[finding.source_start : finding.source_end] != finding.source_passage:
                    raise ValueError(f"stored evidence offsets no longer match paper '{paper_id}'")
            cards.append(card)
            reused += 1
            modes[card.provenance.extraction_mode] = (
                modes.get(card.provenance.extraction_mode, 0) + 1
            )
            continue
        spans = _bounded_spans(source)
        selected, mode = _rank_spans(query, spans, embedding_service)
        findings = [
            EvidenceClaim(
                claim=spans[index][2],
                source_passage=spans[index][2],
                source_start=spans[index][0],
                source_end=spans[index][1],
            )
            for index in selected[:2]
        ]
        card = EvidenceCard(
            paper_id=paper_id,
            findings=findings,
            method="Not reported",
            limitations=[],
            relevance=_relevance(paper, topic),
            provenance=EvidenceProvenance(
                extraction_mode=mode,
                source_content_hash=digest,
                source_field=source_field,
                embedding_model=(
                    embedding_service.model if mode == "embedding_extractive" else None
                ),
                evidence_contract_version=EVIDENCE_CONTRACT_VERSION,
            ),
        )
        cards.append(card)
        modes[mode] = modes.get(mode, 0) + 1
    ledger = EvidenceLedger(cards=cards)
    if knowledge_store is not None:
        knowledge_store.ingest_papers(top_papers)
        knowledge_store.ingest_evidence(ledger.model_dump())
    return ledger.model_dump(), {
        "paper_count": len(cards),
        "reused_count": reused,
        "modes": modes,
        "embedding_model": embedding_service.model if embedding_service else None,
        "contract_version": EVIDENCE_CONTRACT_VERSION,
    }
