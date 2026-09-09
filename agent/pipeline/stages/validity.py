"""Application owned scientific semantics, evidence alignment, and final eligibility gates."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ...contracts.research import HypothesisCandidate, ScientificValidityAudit
from ...knowledge.embeddings import EmbeddingUnavailableError, cosine

_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*", re.IGNORECASE)
_NUMBER = re.compile(r"(?:^|\s)(?:\d+(?:\.\d+)?|\.\d+)(?:%|\s|$)")
_RELATION = re.compile(
    r"\b(?:will\s+)?(?:differ|higher|lower|increase|decrease|noninferior|associated)\b",
    re.IGNORECASE,
)
_PLACEHOLDER = re.compile(
    r"^(?:(?:a|an|the)\s+)?(?:proposed|new|novel|unspecified|generic|tbd|unknown)?\s*"
    r"(?:method|approach|technique|intervention|exposure|mechanism)?$",
    re.IGNORECASE,
)
_STOP = {
    "and",
    "for",
    "from",
    "into",
    "method",
    "model",
    "proposed",
    "research",
    "that",
    "the",
    "this",
    "using",
    "with",
}


def _terms(value: Any) -> set[str]:
    return {
        match.group(0).casefold()
        for match in _WORD.finditer(str(value or ""))
        if len(match.group(0)) > 2 and match.group(0).casefold() not in _STOP
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _passage(card: dict[str, Any]) -> str:
    for finding in card.get("findings", []):
        if isinstance(finding, dict):
            text = str(finding.get("source_passage") or finding.get("claim") or "").strip()
            if text:
                return text[:360]
    return str(card.get("relevance") or "Source backed evidence card")[:360]


def _finding(
    candidate_id: str,
    findings: list[dict[str, Any]],
    category: str,
    severity: str,
    field_paths: list[str],
    issue: str,
    revision: str,
) -> None:
    identity = "\0".join(
        (
            candidate_id,
            "deterministic",
            category,
            "|".join(field_paths),
            issue.casefold(),
            revision.casefold(),
        )
    )
    findings.append(
        {
            "finding_id": f"finding:{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
            "source": "deterministic",
            "category": category,
            "severity": severity,
            "field_paths": field_paths,
            "issue": issue,
            "required_revision": revision,
        }
    )


def _is_placeholder(value: Any) -> bool:
    text = " ".join(str(value or "").split())
    return len(text) < 3 or bool(_PLACEHOLDER.fullmatch(text))


def _segments(candidate: dict[str, Any]) -> list[str]:
    spec = candidate.get("scientific_spec")
    if isinstance(spec, dict):
        exposure = spec.get("exposure_or_intervention", {})
        outcome = spec.get("primary_outcome", {})
        mechanism = spec.get("mechanism", {})
        return [
            " ".join(
                str(item or "")
                for item in (
                    spec.get("system"),
                    exposure.get("name") if isinstance(exposure, dict) else exposure,
                    exposure.get("operational_definition") if isinstance(exposure, dict) else "",
                    outcome.get("name") if isinstance(outcome, dict) else outcome,
                    outcome.get("metric") if isinstance(outcome, dict) else "",
                )
            ),
            " ".join(str(item or "") for item in mechanism.values())
            if isinstance(mechanism, dict)
            else str(mechanism or ""),
        ]
    return [
        str(candidate.get("hypothesis") or ""),
        " ".join(
            str(candidate.get(key) or "")
            for key in ("mechanism", "novelty_delta", "proposed_experiment")
        ),
    ]


def _semantic_findings(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_id = str(candidate.get("candidate_id") or "candidate")
    findings: list[dict[str, Any]] = []
    spec = candidate.get("scientific_spec")
    if not isinstance(spec, dict):
        _finding(
            candidate_id,
            findings,
            "internal_consistency",
            "major",
            ["scientific_spec"],
            "The candidate has no v5 atomic scientific specification.",
            "Regenerate the candidate using the v5 intervention, outcome, mechanism, "
            "and experiment fields.",
        )
        public_intervention = str(candidate.get("mechanism") or "")
        if (
            _is_placeholder(public_intervention)
            or "not reliably specified" in public_intervention.casefold()
        ):
            _finding(
                candidate_id,
                findings,
                "intervention",
                "fatal",
                ["mechanism", "scientific_spec.exposure_or_intervention"],
                "The intervention or mechanism is undefined.",
                "Name the intervention and provide a reproducible operational definition.",
            )
    else:
        exposure = spec.get("exposure_or_intervention", {})
        comparator = spec.get("comparator", {})
        outcome = spec.get("primary_outcome", {})
        mechanism = spec.get("mechanism", {})
        exposure_name = str(exposure.get("name") or "")
        comparator_name = str(comparator.get("name") or "")
        if _is_placeholder(exposure_name) or _is_placeholder(
            exposure.get("operational_definition")
        ):
            _finding(
                candidate_id,
                findings,
                "intervention",
                "fatal",
                ["scientific_spec.exposure_or_intervention"],
                "The intervention/exposure is a placeholder or lacks a reproducible definition.",
                "Name the operation and state exactly how it will be applied or measured.",
            )
        if _is_placeholder(comparator_name) or _is_placeholder(
            comparator.get("operational_definition")
        ):
            _finding(
                candidate_id,
                findings,
                "intervention",
                "fatal",
                ["scientific_spec.comparator"],
                "The comparator is a placeholder or lacks a reproducible definition.",
                "Name and operationally define the comparator.",
            )
        if exposure_name.casefold() == comparator_name.casefold():
            _finding(
                candidate_id,
                findings,
                "internal_consistency",
                "fatal",
                ["scientific_spec.exposure_or_intervention", "scientific_spec.comparator"],
                "The intervention and comparator are identical.",
                "Specify distinct experimental conditions.",
            )
        for key in ("name", "metric"):
            value = str(outcome.get(key) or "")
            if _is_placeholder(value) or _RELATION.search(value):
                _finding(
                    candidate_id,
                    findings,
                    "outcome",
                    "fatal",
                    [f"scientific_spec.primary_outcome.{key}"],
                    f"The primary outcome {key} is missing, generic, or contains comparison prose.",
                    "Provide one atomic outcome and metric; keep direction in "
                    "expected_relation only.",
                )
        relation = str(outcome.get("expected_relation") or "")
        decision_rule = str(outcome.get("decision_rule") or "")
        if relation == "differs" and not (
            _NUMBER.search(decision_rule)
            or any(
                marker in decision_rule.casefold()
                for marker in ("minimum", "effect size", "confidence", "alpha", "equivalence")
            )
        ):
            _finding(
                candidate_id,
                findings,
                "falsifiability",
                "fatal",
                ["scientific_spec.primary_outcome.decision_rule"],
                "A non directional 'differs' claim has no minimum meaningful effect or "
                "statistical decision rule.",
                "Preregister a two sided minimum effect and uncertainty or significance criterion.",
            )
        mechanism_values = list(mechanism.values()) if isinstance(mechanism, dict) else []
        if len(mechanism_values) < 3 or any(_is_placeholder(item) for item in mechanism_values):
            _finding(
                candidate_id,
                findings,
                "mechanism",
                "fatal",
                ["scientific_spec.mechanism"],
                "The proposed mechanism lacks a concrete cause, mediator, or measurable "
                "consequence.",
                "Specify the complete causal chain and its observable consequence.",
            )

    rendered = " ".join(
        str(candidate.get(key) or "")
        for key in ("title", "hypothesis", "mechanism", "proposed_experiment")
    ).casefold()
    predictions = [
        " ".join(str(item).casefold().split()) for item in candidate.get("predictions", [])
    ]
    if len(predictions) != len(set(predictions)) or any(
        re.search(r"\bwill differ\b.*\bwill differ\b", item) for item in predictions
    ):
        _finding(
            candidate_id,
            findings,
            "internal_consistency",
            "fatal",
            ["predictions"],
            "Predictions are duplicated or contain a relation rendered more than once.",
            "Render each atomic outcome exactly once from the structured relation field.",
        )
    if (
        "lossless" in rendered
        and any(marker in rendered for marker in ("psnr", "ssim"))
        and not all(marker in rendered for marker in ("lossy", "rate-distortion"))
    ):
        _finding(
            candidate_id,
            findings,
            "outcome",
            "fatal",
            ["scientific_spec.primary_outcome", "proposed_experiment"],
            "Lossless reconstruction is being evaluated with lossy fidelity metrics "
            "without an explicit rate distortion design.",
            "Use exact reconstruction/integrity for lossless claims or define a "
            "rate matched lossy comparison.",
        )
    if (
        any(marker in rendered for marker in ("image", "psnr", "ssim"))
        and any(marker in rendered for marker in ("sampling frequency", "channel bandwidth"))
        and not any(marker in rendered for marker in ("stream", "transmission", "ecg", "signal"))
    ):
        _finding(
            candidate_id,
            findings,
            "internal_consistency",
            "fatal",
            ["scientific_spec.experiment.controls", "proposed_experiment"],
            "The experiment applies signal/transmission controls to an image only study.",
            "Replace those controls with modality relevant image and codec controls.",
        )
    return findings


def _ground_candidate(
    candidate: dict[str, Any],
    evidence_ledger: dict[str, Any],
    embedding_service: Any | None,
) -> tuple[list[dict[str, Any]], str, str, int, int]:
    cards = {
        str(card.get("paper_id")): card
        for card in evidence_ledger.get("cards", [])
        if isinstance(card, dict) and card.get("paper_id")
    }
    supporting = [item for item in candidate.get("supporting_paper_ids", []) if item in cards]
    segments = [item for item in _segments(candidate) if item.strip()]
    passages = [_passage(cards[paper_id]) for paper_id in supporting]
    lexical = [
        max((_jaccard(_terms(segment), _terms(passage)) for segment in segments), default=0.0)
        for passage in passages
    ]
    semantic: list[float] | None = None
    if embedding_service is not None and segments and passages:
        try:
            vectors = embedding_service.embed([*segments, *passages])
            segment_vectors = vectors[: len(segments)]
            semantic = [
                max(
                    max(0.0, min(1.0, cosine(segment, passage_vector)))
                    for segment in segment_vectors
                )
                for passage_vector in vectors[len(segments) :]
            ]
        except (EmbeddingUnavailableError, ValueError, TypeError):
            semantic = None
    method = "embedding" if semantic is not None else "lexical"
    scores = (
        [0.70 * sem + 0.30 * lex for sem, lex in zip(semantic, lexical, strict=True)]
        if semantic is not None
        else lexical
    )
    contextual_threshold = 0.50 if method == "embedding" else 0.12
    mechanism_threshold = 0.60 if method == "embedding" else 0.25
    contextual = sum(score >= contextual_threshold for score in scores)
    mechanism = sum(score >= mechanism_threshold for score in scores)
    tier = (
        "mechanism_aligned"
        if mechanism >= 1
        else "contextual"
        if contextual >= 2
        else "weak"
        if any(score >= (0.30 if method == "embedding" else 0.05) for score in scores)
        else "insufficient"
    )
    old_links = {str(link.get("paper_id")): link for link in candidate.get("grounding_links", [])}
    links = []
    for paper_id, passage, score in zip(supporting, passages, scores, strict=True):
        previous = old_links.get(paper_id, {})
        links.append(
            {
                "paper_id": paper_id,
                "evidence_role": previous.get("evidence_role", "transferable_method"),
                "alignment_method": method,
                "alignment_score": round(max(0.0, min(1.0, score)), 3),
                "source_passage": passage,
            }
        )
    return links, tier, method, contextual, mechanism


def audit_hypothesis_candidates(
    hypothesis_candidates: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
    topic: str,
    *,
    embedding_service: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit model drafts without letting model declared confidence control eligibility."""
    del topic  # reserved for future domain rule routing
    audited_candidates = []
    audits = []
    for raw_candidate in hypothesis_candidates:
        candidate = dict(raw_candidate)
        findings = _semantic_findings(candidate)
        links, tier, method, contextual, mechanism = _ground_candidate(
            candidate, evidence_ledger, embedding_service
        )
        candidate["grounding_links"] = links
        fatal = any(item["severity"] == "fatal" for item in findings)
        major = any(item["severity"] == "major" for item in findings)
        status = "fail" if fatal else "revisable" if major else "pass"
        provenance = dict(candidate.get("generation_provenance", {}))
        provenance["scientific_integrity"] = status
        provenance["human_review_required"] = bool(fatal or major) or provenance.get(
            "human_review_required", False
        )
        candidate["generation_provenance"] = provenance
        candidate = HypothesisCandidate.model_validate(candidate).model_dump()
        audit = ScientificValidityAudit.model_validate(
            {
                "candidate_id": candidate["candidate_id"],
                "status": status,
                "findings": findings,
                "grounding_tier": tier,
                "grounding_method": method,
                "contextual_link_count": contextual,
                "mechanism_link_count": mechanism,
            }
        ).model_dump()
        audited_candidates.append(candidate)
        audits.append(audit)
    return audited_candidates, audits


def gate_revised_candidate(
    revised_candidate: dict[str, Any],
    candidate_scorecard: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
    topic: str,
    selected_candidate: dict[str, Any] | None = None,
    *,
    embedding_service: Any | None = None,
    attempt_number: int = 1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Stop invalid candidates before prior art retrieval while preserving an evidence result."""
    candidates, audits = audit_hypothesis_candidates(
        [revised_candidate], evidence_ledger, topic, embedding_service=embedding_service
    )
    candidate = candidates[0]
    audit = audits[0]
    selected_score = next(
        (
            item
            for item in candidate_scorecard
            if item.get("candidate_id") == candidate.get("candidate_id")
        ),
        {},
    )
    unresolved = list(selected_score.get("unresolved_major_revisions", []))
    required_finding_ids = set(selected_score.get("unresolved_finding_ids", []))
    blocking_by_id = {
        str(item.get("finding_id")): item
        for item in _blocking_findings(selected_score)
        if item.get("finding_id")
    }
    spec = candidate.get("scientific_spec", {})
    addressed_finding_ids = set(
        spec.get("addressed_finding_ids", []) if isinstance(spec, dict) else []
    )
    verified_finding_ids = set(addressed_finding_ids)
    changed_scientific_fields: list[str] = []
    if selected_candidate is not None and required_finding_ids:
        original_spec = selected_candidate.get("scientific_spec", {})
        comparable_keys = sorted((set(original_spec) | set(spec)) - {"addressed_finding_ids"})
        changed_scientific_fields = [
            f"scientific_spec.{key}"
            for key in comparable_keys
            if original_spec.get(key) != spec.get(key)
        ]
        verified_finding_ids = set()
        for finding_id in addressed_finding_ids & required_finding_ids:
            finding = blocking_by_id.get(finding_id, {})
            field_paths = [str(item) for item in finding.get("field_paths", [])]
            affected_changed = (
                any(
                    path == "scientific_spec"
                    or any(
                        changed == path
                        or changed.startswith(path + ".")
                        or path.startswith(changed + ".")
                        for changed in changed_scientific_fields
                    )
                    for path in field_paths
                )
                if field_paths
                else bool(changed_scientific_fields)
            )
            if affected_changed:
                verified_finding_ids.add(finding_id)
    missing_finding_ids = sorted(required_finding_ids - verified_finding_ids)
    falsely_addressed_finding_ids = sorted(
        (addressed_finding_ids & required_finding_ids) - verified_finding_ids
    )
    grounding_eligible = audit["grounding_tier"] in {"contextual", "mechanism_aligned"}
    eligible = audit["status"] == "pass" and grounding_eligible and not missing_finding_ids
    failure_reasons = [
        *(["scientific_validation"] if audit["status"] != "pass" else []),
        *(["insufficient_grounding"] if not grounding_eligible else []),
        *(["unaddressed_findings"] if missing_finding_ids else []),
    ]
    outcome = {
        "status": "candidate_valid" if eligible else "insufficient_valid_candidates",
        "proposal_eligible": eligible,
        "candidate_id": candidate.get("candidate_id"),
        "scientific_integrity": audit["status"],
        "grounding_tier": audit["grounding_tier"],
        "findings": audit["findings"],
        "attempt_count": attempt_number,
        "required_finding_ids": sorted(required_finding_ids),
        "addressed_finding_ids": sorted(addressed_finding_ids),
        "verified_addressed_finding_ids": sorted(verified_finding_ids),
        "falsely_addressed_finding_ids": falsely_addressed_finding_ids,
        "changed_scientific_fields": changed_scientific_fields,
        "unresolved_major_revisions": unresolved if missing_finding_ids else [],
        "missing_addressed_finding_ids": missing_finding_ids,
        "remaining_findings": audit["findings"],
        "failure_reasons": failure_reasons,
        "repair_attempted": attempt_number > 1,
        "repair_accepted": eligible and attempt_number > 1,
    }
    return candidate, audit, outcome


def initialize_revision_candidate(
    selected_candidate: dict[str, Any],
) -> dict[str, Any]:
    """Start the bounded revision gate from the exact selected candidate.

    Revision is performed by the focused repair unit only when the deterministic
    audit identifies a concrete blocking finding. This avoids asking a model to
    regenerate an already valid, citation bearing aggregate contract.
    """
    return dict(selected_candidate)


def _blocking_findings(score: dict[str, Any]) -> list[dict[str, Any]]:
    supplied = score.get("blocking_findings", [])
    if isinstance(supplied, list) and supplied:
        return [dict(item) for item in supplied if isinstance(item, dict)][:8]
    return [
        {
            "finding_id": finding_id,
            "source": "legacy",
            "category": "clarity",
            "severity": "major",
            "issue": revision,
            "required_revision": revision,
        }
        for finding_id, revision in zip(
            score.get("unresolved_finding_ids", []),
            score.get("unresolved_major_revisions", []),
            strict=False,
        )
    ][:8]


def _repair_blocking_findings(
    selected_score: dict[str, Any],
    initial_revision_validity_audit: dict[str, Any],
    initial_revision_outcome: dict[str, Any],
) -> list[dict[str, Any]]:
    blocking = _blocking_findings(selected_score)
    known_ids = {str(item.get("finding_id")) for item in blocking if item.get("finding_id")}
    for finding in initial_revision_validity_audit.get("findings", []):
        if not isinstance(finding, dict) or finding.get("severity") not in {"major", "fatal"}:
            continue
        finding_id = str(finding.get("finding_id") or "")
        if finding_id and finding_id not in known_ids:
            blocking.append(dict(finding))
            known_ids.add(finding_id)
    if "insufficient_grounding" in initial_revision_outcome.get("failure_reasons", []):
        grounding = {
            "finding_id": "gate:grounding",
            "source": "deterministic",
            "category": "evidence",
            "severity": "fatal",
            "issue": "The candidate is not contextually grounded in its selected evidence.",
            "required_revision": (
                "Choose evidence slots whose source passages directly support the named "
                "intervention, comparator, mechanism, and measurable outcome."
            ),
        }
        if grounding["finding_id"] not in known_ids:
            blocking.append(grounding)
    return blocking[:8]


def prepare_revision_repair_packets(
    initially_validated_revised_candidate: dict[str, Any],
    initial_revision_validity_audit: dict[str, Any],
    initial_revision_outcome: dict[str, Any],
    candidate_scorecard: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
    research_scope: dict[str, Any],
    top_papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build at most one closed, evidence backed repair packet after the first gate."""
    if initial_revision_outcome.get("proposal_eligible"):
        return []
    candidate_id = str(initially_validated_revised_candidate.get("candidate_id") or "")
    selected_score = next(
        (item for item in candidate_scorecard if item.get("candidate_id") == candidate_id), {}
    )
    blocking = _repair_blocking_findings(
        selected_score, initial_revision_validity_audit, initial_revision_outcome
    )

    titles = {
        str(paper.get("paper_id")): str(paper.get("title") or "")
        for paper in top_papers
        if isinstance(paper, dict) and paper.get("paper_id")
    }
    roles = {
        str(paper.get("paper_id")): paper.get("relevance_judgement", {}).get("role")
        for paper in top_papers
        if isinstance(paper, dict) and paper.get("paper_id")
    }
    allowed = (
        set(research_scope.get("supporting_paper_ids", []))
        if research_scope.get("mode") == "domain"
        else None
    )
    cards = [
        (index, card)
        for index, card in enumerate(evidence_ledger.get("cards", []), start=1)
        if isinstance(card, dict)
        and card.get("paper_id")
        and roles.get(str(card.get("paper_id"))) in {"direct_evidence", "transferable_method"}
        and (allowed is None or str(card.get("paper_id")) in allowed)
    ]
    query_terms = _terms(
        " ".join(
            (
                str(initially_validated_revised_candidate.get("scientific_spec") or ""),
                " ".join(str(item.get("required_revision") or "") for item in blocking),
            )
        )
    )
    preferred = list(
        dict.fromkeys(
            str(item)
            for item in initially_validated_revised_candidate.get("supporting_paper_ids", [])
        )
    )
    ranked = sorted(
        cards,
        key=lambda indexed_card: (
            -len(
                query_terms
                & _terms(
                    titles.get(str(indexed_card[1].get("paper_id")), "")
                    + " "
                    + _passage(indexed_card[1])
                )
            ),
            indexed_card[0],
        ),
    )
    ordered_ids = list(
        dict.fromkeys([*preferred, *(str(card.get("paper_id")) for _, card in ranked)])
    )[:6]
    indexed_by_id = {str(card.get("paper_id")): (index, card) for index, card in cards}
    evidence_cards = []
    for paper_id in ordered_ids:
        if paper_id not in indexed_by_id:
            continue
        index, card = indexed_by_id[paper_id]
        finding = next((item for item in card.get("findings", []) if isinstance(item, dict)), {})
        evidence_cards.append(
            {
                "slot": f"E{index}",
                "title": titles.get(paper_id, "")[:180],
                "claim": str(finding.get("claim") or "")[:220],
                "source_passage": str(finding.get("source_passage") or "")[:300],
                "method": str(card.get("method") or "")[:160],
            }
        )
    return [
        {
            "candidate": {
                "candidate_id": candidate_id,
                "scientific_spec": initially_validated_revised_candidate.get("scientific_spec", {}),
                "configured_axis": initially_validated_revised_candidate.get(
                    "generation_provenance", {}
                ).get("configured_axis"),
                "effective_axis": initially_validated_revised_candidate.get(
                    "generation_provenance", {}
                ).get("effective_axis"),
            },
            "blocking_findings": blocking,
            "evidence_cards": evidence_cards,
        }
    ]


def _audit_rank(audit: dict[str, Any], outcome: dict[str, Any]) -> tuple[int, int, int]:
    return (
        {"pass": 2, "revisable": 1, "fail": 0}.get(str(audit.get("status")), 0),
        {
            "mechanism_aligned": 3,
            "contextual": 2,
            "weak": 1,
            "insufficient": 0,
        }.get(str(audit.get("grounding_tier")), 0),
        -len(outcome.get("missing_addressed_finding_ids", [])),
    )


def finalize_revision_candidates(
    initially_validated_revised_candidate: dict[str, Any],
    initial_revision_validity_audit: dict[str, Any],
    initial_revision_outcome: dict[str, Any],
    revision_repair_candidates: list[dict[str, Any]],
    candidate_scorecard: list[dict[str, Any]],
    evidence_ledger: dict[str, Any],
    topic: str,
    selected_candidate: dict[str, Any] | None = None,
    *,
    embedding_service: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Use one exact repair at most, then return a final graceful gate outcome."""
    if initial_revision_outcome.get("proposal_eligible"):
        return (
            initially_validated_revised_candidate,
            initial_revision_validity_audit,
            initial_revision_outcome,
        )
    repair_rejection = "discarded_non_exact_repair"
    unwrapped_repairs = [
        item.get("value") if isinstance(item, dict) and item.get("status") == "succeeded" else item
        for item in revision_repair_candidates
    ]
    repair = next(
        (
            item
            for item in unwrapped_repairs
            if isinstance(item, dict)
            and item.get("generation_provenance", {}).get("mode")
            in {"structured_exact", "structured_normalized"}
        ),
        None,
    )
    if repair is not None:
        initial_provenance = initially_validated_revised_candidate.get("generation_provenance", {})
        repair_provenance = repair.get("generation_provenance", {})
        identity_matches = repair.get("candidate_id") == initially_validated_revised_candidate.get(
            "candidate_id"
        )
        axes_match = all(
            not initial_provenance.get(key)
            or repair_provenance.get(key) == initial_provenance.get(key)
            for key in ("configured_axis", "effective_axis")
        )
        if not identity_matches or not axes_match:
            repair_rejection = "discarded_identity_or_axis_mismatch"
            repair = None
    if repair is None:
        outcome = {
            **initial_revision_outcome,
            "attempt_count": 2,
            "repair_attempted": True,
            "repair_accepted": False,
            "repair_disposition": repair_rejection,
            "failure_reasons": list(
                dict.fromkeys(
                    [*initial_revision_outcome.get("failure_reasons", []), "repair_invalid"]
                )
            ),
        }
        return initially_validated_revised_candidate, initial_revision_validity_audit, outcome

    repair_scorecard = [dict(item) for item in candidate_scorecard]
    repair_score = next(
        (
            item
            for item in repair_scorecard
            if item.get("candidate_id") == repair.get("candidate_id")
        ),
        None,
    )
    if repair_score is None:
        repair_score = {"candidate_id": repair.get("candidate_id")}
        repair_scorecard.append(repair_score)
    repair_findings = _repair_blocking_findings(
        repair_score, initial_revision_validity_audit, initial_revision_outcome
    )
    repair_score["blocking_findings"] = repair_findings
    repair_score["unresolved_finding_ids"] = [str(item["finding_id"]) for item in repair_findings]
    repair_score["unresolved_major_revisions"] = [
        str(item.get("required_revision") or "") for item in repair_findings
    ]
    repaired, repaired_audit, repaired_outcome = gate_revised_candidate(
        repair,
        repair_scorecard,
        evidence_ledger,
        topic,
        selected_candidate,
        embedding_service=embedding_service,
        attempt_number=2,
    )
    if repaired_outcome.get("proposal_eligible"):
        provenance = repaired.setdefault("generation_provenance", {})
        provenance["human_review_required"] = True
        provenance["limitations"] = list(
            dict.fromkeys(
                [
                    *provenance.get("limitations", []),
                    "A focused repair passed deterministic scientific and grounding checks but "
                    "was not independently reviewed again.",
                ]
            )
        )[:8]
        repaired = HypothesisCandidate.model_validate(repaired).model_dump()
        repaired_outcome["repair_disposition"] = "accepted"
        return repaired, repaired_audit, repaired_outcome

    if _audit_rank(repaired_audit, repaired_outcome) > _audit_rank(
        initial_revision_validity_audit, initial_revision_outcome
    ):
        repaired_outcome["repair_disposition"] = "retained_better_ineligible_repair"
        return repaired, repaired_audit, repaired_outcome
    outcome = {
        **initial_revision_outcome,
        "attempt_count": 2,
        "repair_attempted": True,
        "repair_accepted": False,
        "repair_disposition": "retained_initial_candidate",
    }
    return initially_validated_revised_candidate, initial_revision_validity_audit, outcome


__all__ = [
    "audit_hypothesis_candidates",
    "finalize_revision_candidates",
    "gate_revised_candidate",
    "initialize_revision_candidate",
    "prepare_revision_repair_packets",
]
