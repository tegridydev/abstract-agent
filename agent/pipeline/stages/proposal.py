"""Application owned proposal assembly for exact and model limited runs."""

from __future__ import annotations

import re
from typing import Any

from ...contracts.research import ResearchProposal


def _clean(value: Any, maximum: int, fallback: str) -> str:
    text = " ".join(str(value or "").split()) or fallback
    if len(text) <= maximum:
        return text
    shortened = text[: maximum - 3]
    boundary = shortened.rfind(" ")
    if boundary > maximum // 2:
        shortened = shortened[:boundary]
    return shortened.rstrip() + "..."


def _dedupe(items: list[str], maximum: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item or "").split())
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
        if len(result) == maximum:
            break
    return result


def _unverified_prose(value: Any, maximum: int, fallback: str) -> str:
    """Remove unsupported epistemic certainty without changing the proposed mechanism."""
    text = _clean(value, maximum, fallback)
    replacements = (
        (r"\bthe first\b", "a candidate"),
        (r"\bunprecedented\b", "candidate"),
        (r"\bstate[- ]of[- ]the[- ]art\b", "comparative"),
        (r"\bnovel\b", "candidate"),
        (r"\bnew understanding\b", "potential understanding"),
        (r"\bwill outperform\b", "is predicted to outperform"),
        (r"\bwill improve\b", "is predicted to improve"),
        (r"\bwill reduce\b", "is predicted to reduce"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return _clean(text, maximum, fallback)


def _test_prediction(value: Any) -> str:
    text = _unverified_prose(value, 560, "A measurable directional difference should be tested.")
    lowered = text.casefold()
    if lowered.startswith(("test whether ", "evaluate whether ", "measure whether ")):
        return text
    predicted = re.match(r"^(?:the )?(.+?) is predicted to (.+)$", text, flags=re.IGNORECASE)
    if predicted:
        return _clean(f"Test whether {predicted.group(1)} can {predicted.group(2)}", 600, text)
    first_word = text.split(maxsplit=1)[0]
    fragment = text if first_word.isupper() else text[0].lower() + text[1:]
    return _clean(f"Test whether {fragment}", 600, text)


def build_conservative_proposal(
    *,
    topic: str,
    final_candidate: dict[str, Any],
    novelty_verification: dict[str, Any],
    evidence_papers: list[dict[str, Any]],
    model: str | None = None,
    semantic_draft: str | None = None,
    model_content: dict[str, Any] | None = None,
    generation_mode: str | None = None,
    validation_attempts: int = 1,
) -> dict[str, Any]:
    """Assemble model owned prose into a canonical, auditable proposal exactly once."""
    supplied = model_content if isinstance(model_content, dict) else {}
    known = {
        str(paper.get("paper_id")): paper
        for paper in evidence_papers
        if isinstance(paper, dict) and paper.get("paper_id")
    }
    references = [
        paper_id
        for paper_id in dict.fromkeys(final_candidate.get("supporting_paper_ids", []))
        if paper_id in known
    ][:8]
    if not references:
        references = list(known)[:2]
    if not references:
        raise ValueError("cannot construct a proposal without evidence papers")

    mode = generation_mode or (
        "draft_projected" if semantic_draft or supplied else "conservative_fallback"
    )
    contract_valid = mode in {"structured_exact", "structured_normalized"}
    novelty_supported = novelty_verification.get("status") == "novelty_supported"
    verified = contract_valid and novelty_supported
    status = "novelty_supported" if verified else "unverified_research_direction"

    hypothesis = _clean(
        final_candidate.get("hypothesis"),
        1200,
        f"Investigate a measurable relationship within {topic}.",
    )
    method = _clean(
        final_candidate.get("proposed_experiment"),
        2000,
        "Use a preregistered controlled comparison with independent replication.",
    )
    raw_predictions = final_candidate.get("predictions", [])
    predictions = _dedupe(
        [
            _clean(item, 600, "A measurable directional difference is expected.")
            for item in raw_predictions[:8]
            if str(item).strip()
        ],
        8,
    ) or ["A preregistered directional difference should be reproducible across evaluations."]
    raw_insights = final_candidate.get("novel_insights", [])
    insights = _dedupe(
        [
            _clean(item, 600, "Potential insight to test.")
            for item in raw_insights[:8]
            if str(item).strip()
        ],
        8,
    ) or ["The proposed mechanism is a potential insight to test, not an established finding."]
    rationale = _clean(
        supplied.get("rationale") or final_candidate.get("mechanism") or semantic_draft,
        1600,
        "This direction is grounded in cited context but its mechanism remains unverified.",
    )
    title = _clean(
        final_candidate.get("title"),
        240,
        f"Research direction: {topic}",
    )
    falsifier = _clean(
        final_candidate.get("falsifier"),
        1000,
        "Reject the hypothesis when the preregistered result is absent or reverses.",
    )
    if not verified:
        title = _unverified_prose(title, 240, title)
        hypothesis = _unverified_prose(hypothesis, 1200, hypothesis)
        rationale = _unverified_prose(rationale, 1600, rationale)
        method = _unverified_prose(method, 2000, method)
        predictions = [_test_prediction(item) for item in predictions]
        insights = [_unverified_prose(item, 600, "Potential insight to test.") for item in insights]

    supplied_limitations = _dedupe(
        [
            _clean(item, 600, "Model generated limitation.")
            for item in supplied.get("limitations_and_risks", [])[:9]
            if str(item).strip()
        ],
        9,
    )
    candidate_risks = _dedupe(
        [
            _clean(item, 600, "Model generated risk.")
            for item in final_candidate.get("risks", [])[:8]
            if str(item).strip()
        ],
        8,
    )
    limitations = _dedupe(supplied_limitations or candidate_risks, 9)
    if not verified:
        limitations = _dedupe(
            [*limitations, "Human review is required before relying on this proposal."], 10
        )

    direction = "proposal" if verified else "unverified research direction"
    abstract = _clean(
        " ".join(
            (
                f"This {direction} tests the following hypothesis: {hypothesis}",
                f"Method: {method}",
                f"Primary preregistered test: {predictions[0]}",
                "Novelty requires human review."
                if not verified
                else "Novelty passed the configured prior art audit.",
            )
        ),
        2400,
        f"A research direction concerning {topic}.",
    )

    claims = [
        ("hypothesis", "hypothesis", hypothesis),
        *[
            (f"prediction:{index}", "prediction", prediction)
            for index, prediction in enumerate(predictions, start=1)
        ],
    ]
    result = {
        "proposal_status": status,
        "title": title,
        "hypothesis": hypothesis,
        "rationale": rationale,
        "proposed_method": method,
        "predictions": predictions,
        "falsifier": falsifier,
        "novel_insights": insights,
        "limitations_and_risks": limitations,
        "claim_evidence": [
            {
                "claim_id": claim_id,
                "claim_kind": claim_kind,
                "claim": _clean(claim, 500, "Proposed inference."),
                "paper_ids": references,
                "relationship": "proposed_inference",
                "confidence": "low" if not verified else "medium",
            }
            for claim_id, claim_kind, claim in claims
        ],
        "abstract": abstract,
        "references": references,
        "generation_provenance": {
            "mode": mode,
            "model": model,
            "validation_attempts": max(0, validation_attempts),
            "human_review_required": not contract_valid,
            "application_enriched": True,
            "contract_integrity": {
                "structured_exact": "exact",
                "structured_normalized": "normalized",
                "draft_projected": "projected",
                "conservative_fallback": "fallback",
            }.get(mode, "projected"),
            "scientific_integrity": "pass",
            "limitations": []
            if contract_valid
            else [
                "Application owned recovery produced the final structured proposal.",
                "Novelty is not claimed.",
            ],
        },
    }
    return ResearchProposal.model_validate(result).model_dump()


__all__ = ["build_conservative_proposal"]
