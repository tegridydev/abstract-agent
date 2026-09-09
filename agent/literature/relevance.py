"""Deterministic paper relevance and source bound rationale construction."""

from __future__ import annotations

import math
import re
from typing import Any

from ..knowledge.embeddings import EmbeddingUnavailableError, cosine

_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_BACKGROUND_MARKERS = ("survey", "review", "state of the art", "systematic mapping")
_STOP_WORDS = {
    "about",
    "analysis",
    "approach",
    "based",
    "for",
    "from",
    "method",
    "methods",
    "model",
    "models",
    "research",
    "study",
    "that",
    "the",
    "this",
    "using",
    "with",
}


def _term_key(value: str) -> str:
    token = value.casefold()
    if token.startswith("compress"):
        return "compress"
    if token.startswith("quantiz"):
        return "quantize"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _terms(value: Any) -> set[str]:
    return {
        _term_key(match.group(0))
        for match in _WORD.finditer(str(value or ""))
        if len(match.group(0)) > 2 and match.group(0).casefold() not in _STOP_WORDS
    }


def _bounded_excerpt(text: str, anchors: set[str], maximum: int = 150) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return "No abstract passage was available."
    sentences = [item.strip() for item in _SENTENCE.split(normalized) if item.strip()]
    passage = max(sentences or [normalized], key=lambda item: len(_terms(item) & anchors))
    if len(passage) <= maximum:
        return passage
    shortened = passage[: maximum - 3]
    boundary = shortened.rfind(" ")
    if boundary > maximum // 2:
        shortened = shortened[:boundary]
    return shortened.rstrip() + "..."


def _concept_gate(content_terms: set[str], topic_terms: set[str], concepts: list[set[str]]) -> bool:
    usable = [concept for concept in concepts if concept]
    minimum = 1 if len(topic_terms) <= 2 else 2
    topic_pass = len(content_terms & topic_terms) >= minimum
    if not topic_pass:
        return False
    # Model-planned concepts can strengthen a match but can never override the user-owned topic.
    return (
        not usable
        or any(concept <= content_terms for concept in usable)
        or topic_terms <= content_terms
    )


def assess_relevance(
    candidate_papers: list[dict[str, Any]],
    topic: str,
    *,
    research_plan: dict[str, Any] | None = None,
    revised_candidate: dict[str, Any] | None = None,
    alternate_candidate: dict[str, Any] | None = None,
    max_results: int = 12,
    minimum_score: float = 0.40,
    embedding_service: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Admit papers without allowing a completion model to bind identities or evidence roles."""
    if max_results < 1:
        raise ValueError("max_results must be positive")
    topic_terms = _terms(topic)
    concepts = [
        _terms(concept)
        for concept in (research_plan or {}).get("required_concepts", [])
        if str(concept).strip()
    ]
    focus_parts = [topic]
    candidate_focuses: dict[str, set[str]] = {}
    candidate_focus_texts: dict[str, str] = {}
    for candidate in (revised_candidate, alternate_candidate):
        if isinstance(candidate, dict):
            candidate_parts = [
                str(candidate.get(key) or "") for key in ("title", "hypothesis", "mechanism")
            ]
            focus_parts.extend(candidate_parts)
            candidate_id = str(candidate.get("candidate_id") or "")
            if candidate_id:
                candidate_focus_text = " ".join([topic, *candidate_parts])
                candidate_focus_texts[candidate_id] = candidate_focus_text
                candidate_focuses[candidate_id] = _terms(candidate_focus_text)
    focus = " ".join(part for part in focus_parts if part).strip()
    focus_terms = _terms(focus) or topic_terms
    paper_texts = [
        " ".join(
            part
            for part in (str(paper.get("title") or ""), str(paper.get("summary") or ""))
            if part
        )
        for paper in candidate_papers
    ]
    semantic: list[float] | None = None
    candidate_semantic: dict[str, list[float]] = {}
    if embedding_service is not None and paper_texts:
        try:
            candidate_ids = list(candidate_focus_texts)
            focus_texts = [focus, *(candidate_focus_texts[item] for item in candidate_ids)]
            vectors = embedding_service.embed([*focus_texts, *paper_texts])
            paper_vectors = vectors[len(focus_texts) :]
            semantic = [max(0.0, min(1.0, cosine(vectors[0], item))) for item in paper_vectors]
            candidate_semantic = {
                candidate_id: [
                    max(0.0, min(1.0, cosine(vectors[offset], item))) for item in paper_vectors
                ]
                for offset, candidate_id in enumerate(candidate_ids, start=1)
            }
        except (EmbeddingUnavailableError, ValueError, TypeError):
            semantic = None
            candidate_semantic = {}

    decisions: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, (paper, paper_text) in enumerate(zip(candidate_papers, paper_texts, strict=True)):
        paper_id = str(paper.get("paper_id") or "").strip()
        if not paper_id:
            raise ValueError("relevance candidates require canonical paper IDs")
        title = str(paper.get("title") or "")
        title_terms = _terms(title)
        content_terms = _terms(paper_text)
        matched_candidate_ids = [
            str(candidate_id)
            for candidate_id in paper.get("matched_candidate_ids", [])
            if str(candidate_id) in candidate_focuses
        ]
        paper_focus_terms = (
            set().union(*(candidate_focuses[item] for item in matched_candidate_ids))
            if matched_candidate_ids
            else focus_terms
        )
        overlap = paper_focus_terms & content_terms
        coverage = len(overlap) / len(paper_focus_terms) if paper_focus_terms else 0.0
        title_coverage = (
            len(paper_focus_terms & title_terms) / len(paper_focus_terms)
            if paper_focus_terms
            else 0.0
        )
        lexical = min(1.0, 0.70 * coverage + 0.30 * title_coverage)
        retrieval = max(
            0.0,
            min(
                1.0,
                float(paper.get("relevance_score") or paper.get("composite_score") or 0.0),
            ),
        )
        semantic_score = semantic[index] if semantic is not None else None
        if matched_candidate_ids and candidate_semantic:
            semantic_score = max(
                candidate_semantic[candidate_id][index]
                for candidate_id in matched_candidate_ids
                if candidate_id in candidate_semantic
            )
        combined = (
            0.50 * lexical + 0.35 * semantic_score + 0.15 * retrieval
            if semantic_score is not None
            else 0.75 * lexical + 0.25 * retrieval
        )
        gate = _concept_gate(content_terms, topic_terms, concepts)
        if not gate:
            combined = 0.0
        background = any(marker in title.casefold() for marker in _BACKGROUND_MARKERS)
        if combined < minimum_score:
            role = "irrelevant"
        elif background:
            role = "background"
        elif lexical >= 0.60 or (title_coverage > 0 and combined >= 0.60):
            role = "direct_evidence"
        elif combined >= 0.50:
            role = "transferable_method"
        else:
            role = "background"
        matched = sorted(overlap)[:8]
        excerpt = _bounded_excerpt(str(paper.get("summary") or title), set(matched))
        rationale = (
            "No required topic concept was found in this paper."
            if not gate
            else (f"Matched {', '.join(matched) or 'the topic'}; source: {excerpt}")
        )[:240]
        accepted_paper = role != "irrelevant"
        decision = {
            "paper_id": paper_id,
            "score": max(1, min(5, math.ceil(combined * 5))),
            "role": role,
            "rationale": rationale,
            "accepted": accepted_paper,
            "decision_source": "deterministic_hybrid",
            "alignment_method": "embedding" if semantic_score is not None else "lexical",
            "score_components": {
                "lexical": round(lexical, 3),
                "semantic": round(semantic_score, 3) if semantic_score is not None else None,
                "retrieval": round(retrieval, 3),
                "combined": round(combined, 3),
            },
            "matched_terms": matched,
            "source_passage": excerpt,
        }
        decisions.append(decision)
        if accepted_paper:
            item = dict(paper)
            item["relevance_judgement"] = decision
            item["selection_score"] = round(combined, 3)
            accepted.append(item)
        else:
            rejected.append(decision)
    accepted.sort(
        key=lambda paper: (-float(paper.get("selection_score", 0)), str(paper["paper_id"]))
    )
    if not candidate_focuses:
        selected = accepted[:max_results]
    else:
        selected = []
        selected_ids: set[str] = set()
        per_candidate_limit = min(6, max_results)
        for candidate_id in candidate_focuses:
            candidate_papers_for_id = [
                paper
                for paper in accepted
                if candidate_id in paper.get("matched_candidate_ids", [])
            ][:per_candidate_limit]
            for paper in candidate_papers_for_id:
                if paper["paper_id"] not in selected_ids and len(selected) < max_results:
                    selected.append(paper)
                    selected_ids.add(paper["paper_id"])
        for paper in accepted:
            if paper["paper_id"] not in selected_ids and len(selected) < max_results:
                selected.append(paper)
                selected_ids.add(paper["paper_id"])
    return selected, decisions, rejected


__all__ = ["assess_relevance"]
