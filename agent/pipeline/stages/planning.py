"""Deterministic planning fallback policy."""

from __future__ import annotations

import re

from ...contracts.research import ResearchPlan


def recover_research_plan(topic: str) -> dict:
    normalized = " ".join(topic.split())[:120] or "research topic"
    terms = re.findall(r"[A-Za-z0-9_-]+", normalized)
    concept = " ".join(terms[:4]) or "research topic"
    return ResearchPlan.model_validate(
        {
            "topic_interpretation": normalized,
            "assumptions": ["The planning model was unavailable; broad search facets were used."],
            "required_concepts": [concept],
            "mechanism_query": f"{normalized} mechanism analysis"[:160],
            "approaches_query": f"{normalized} methods approaches"[:160],
            "prior_art_query": f"{normalized} prior art evidence"[:160],
            "exclusions": [],
        }
    ).model_dump()


def normalize_research_plan(raw: object, topic: str) -> tuple[dict, list[dict[str, str]]]:
    """Overlay bounded model enrichment on an application owned search baseline."""
    baseline = recover_research_plan(topic)
    source = raw if isinstance(raw, dict) else {}
    events: list[dict[str, str]] = []
    merged = dict(baseline)
    for field in (
        "topic_interpretation",
        "assumptions",
        "required_concepts",
        "mechanism_query",
        "approaches_query",
        "prior_art_query",
        "exclusions",
    ):
        if source.get(field):
            merged[field] = source[field]

    topic_terms = {
        item.casefold() for item in re.findall(r"[A-Za-z0-9_-]+", topic) if len(item) > 2
    }
    concepts = [
        " ".join(str(item).split())[:120]
        for item in merged.get("required_concepts", [])
        if str(item).strip()
    ]
    concept_terms = {
        token.casefold() for concept in concepts for token in re.findall(r"[A-Za-z0-9_-]+", concept)
    }
    if topic_terms and not topic_terms & concept_terms:
        concepts.insert(0, baseline["required_concepts"][0])
        events.append(
            {
                "path": "required_concepts",
                "operation": "baseline_merge",
                "reason": "restored a distinctive topic concept",
            }
        )
    merged["required_concepts"] = list(dict.fromkeys(concepts))[:4]
    return ResearchPlan.model_validate(merged).model_dump(), events


__all__ = ["normalize_research_plan", "recover_research_plan"]
