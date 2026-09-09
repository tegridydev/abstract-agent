"""Candidate critique fallback policy."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any, Literal

from ...contracts.research import CandidateCritiques


def normalize_critique_item(
    raw_item: Any,
    *,
    expected_candidate_ids: set[str],
    critic_source: Literal["evidence_critic", "methodology_critic", "legacy"] = "legacy",
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Retain bounded critique content; missing scores become non ranking fallback scores."""
    if not isinstance(raw_item, dict):
        return None, [{"operation": "drop", "reason": "critique is not an object"}]
    candidate_id = str(raw_item.get("candidate_id") or "")
    if not candidate_id or candidate_id not in expected_candidate_ids:
        return None, [
            {
                "operation": "drop",
                "reason": f"unknown critique candidate_id '{candidate_id or 'missing'}'",
            }
        ]
    operations: list[dict[str, Any]] = []
    scores: dict[str, int] = {}
    missing_score = False
    for key in ("grounding", "novelty", "falsifiability", "feasibility", "clarity"):
        value = raw_item.get(key)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = 1
            missing_score = True
        bounded = min(5, max(1, parsed))
        if value != bounded:
            operations.append(
                {
                    "path": key,
                    "operation": "derive" if missing_score else "limit",
                    "reason": "critique score was bounded to the 1-5 contract",
                }
            )
        scores[key] = bounded

    used_fallback_content = False

    allowed_categories = {
        "evidence",
        "novelty",
        "internal_consistency",
        "intervention",
        "outcome",
        "falsifiability",
        "mechanism",
        "confounding",
        "feasibility",
        "clarity",
    }
    findings = []
    raw_findings = raw_item.get("findings", [])
    for finding in raw_findings if isinstance(raw_findings, list) else []:
        if not isinstance(finding, dict):
            continue
        category = str(finding.get("category") or "")
        severity = str(finding.get("severity") or "")
        issue = " ".join(str(finding.get("issue") or "").split())[:360]
        revision = " ".join(str(finding.get("required_revision") or "").split())[:360]
        if (
            category in allowed_categories
            and severity in {"minor", "major", "fatal"}
            and issue
            and revision
        ):
            identity = "\0".join(
                (candidate_id, critic_source, category, issue.casefold(), revision.casefold())
            )
            findings.append(
                {
                    "finding_id": f"finding:{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
                    "source": critic_source,
                    "category": category,
                    "severity": severity,
                    "issue": issue,
                    "required_revision": revision,
                }
            )
        if len(findings) == 8:
            break

    raw_issues = raw_item.get("issues", [])
    raw_revisions = raw_item.get("required_revisions", [])
    issue_items = (
        [" ".join(str(item).split())[:360] for item in raw_issues if str(item).strip()]
        if isinstance(raw_issues, list)
        else []
    )
    revision_items = (
        [" ".join(str(item).split())[:360] for item in raw_revisions if str(item).strip()]
        if isinstance(raw_revisions, list)
        else []
    )
    if not findings and issue_items:
        for index, issue in enumerate(issue_items[:8]):
            lowered = issue.casefold()
            category = (
                "intervention"
                if any(
                    item in lowered for item in ("intervention", "undefined method", "technique")
                )
                else "outcome"
                if any(item in lowered for item in ("outcome", "metric", "psnr", "ssim"))
                else "falsifiability"
                if any(item in lowered for item in ("falsif", "will differ", "trivial"))
                else "mechanism"
                if "mechanism" in lowered
                else "internal_consistency"
                if any(item in lowered for item in ("inconsisten", "restatement", "duplicate"))
                else "evidence"
                if any(item in lowered for item in ("evidence", "ground", "citation", "source"))
                else "novelty"
                if "novel" in lowered
                else "feasibility"
            )
            fatal = any(
                marker in lowered
                for marker in (
                    "undefined",
                    "non-falsifiable",
                    "no hypothesis",
                    "internal inconsistency",
                    "not defined",
                )
            )
            required_revision = (
                revision_items[index]
                if index < len(revision_items)
                else "Resolve this issue before candidate selection."
            )
            identity = "\0".join(
                (
                    candidate_id,
                    critic_source,
                    category,
                    issue.casefold(),
                    required_revision.casefold(),
                )
            )
            findings.append(
                {
                    "finding_id": f"finding:{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
                    "source": critic_source,
                    "category": category,
                    "severity": "fatal" if fatal else "major",
                    "issue": issue,
                    "required_revision": required_revision,
                }
            )
        operations.append(
            {
                "path": "findings",
                "operation": "derive",
                "reason": "converted legacy issue arrays into selection blocking findings",
            }
        )

    def statements(key: str, fallback: str) -> list[str]:
        nonlocal used_fallback_content
        value = raw_item.get(key, [])
        items = (
            list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))[:6]
            if isinstance(value, list)
            else []
        )
        if not items:
            used_fallback_content = True
            items = [fallback]
            operations.append(
                {
                    "path": key,
                    "operation": "fallback",
                    "reason": "inserted a conservative review requirement",
                }
            )
        return items

    if findings:
        issues = list(
            dict.fromkeys(
                [item["issue"] for item in findings]
                + (
                    [str(item).strip() for item in raw_issues]
                    if isinstance(raw_issues, list)
                    else []
                )
            )
        )[:6]
        revisions = list(
            dict.fromkeys(
                [item["required_revision"] for item in findings]
                + (
                    [str(item).strip() for item in raw_revisions]
                    if isinstance(raw_revisions, list)
                    else []
                )
            )
        )[:6]
    elif raw_item.get("verdict") == "pass":
        issues = []
        revisions = []
    else:
        issues = statements("issues", "The model critique requires human review.")
        revisions = statements(
            "required_revisions", "Obtain a complete independent critique before selection."
        )
    item = {
        "candidate_id": candidate_id,
        **scores,
        "issues": issues,
        "required_revisions": revisions,
        "verdict": raw_item.get("verdict")
        if raw_item.get("verdict") in {"pass", "revise", "exclude"}
        else (
            "exclude"
            if any(item.get("severity") == "fatal" for item in findings)
            else "revise"
            if findings
            else "pass"
        ),
        "findings": findings,
        "fatal_flags": list(
            dict.fromkeys(
                [
                    str(flag)
                    for flag in raw_item.get("fatal_flags", [])
                    if str(flag)
                    in {
                        "internal_inconsistency",
                        "undefined_intervention",
                        "undefined_outcome",
                        "non_falsifiable",
                        "unsupported_mechanism",
                    }
                ]
                + [
                    {
                        "internal_consistency": "internal_inconsistency",
                        "intervention": "undefined_intervention",
                        "outcome": "undefined_outcome",
                        "falsifiability": "non_falsifiable",
                        "mechanism": "unsupported_mechanism",
                    }[str(finding["category"])]
                    for finding in findings
                    if finding.get("severity") == "fatal"
                    and finding.get("category")
                    in {
                        "internal_consistency",
                        "intervention",
                        "outcome",
                        "falsifiability",
                        "mechanism",
                    }
                ]
            )
        )[:5],
        "score_source": (
            "fallback"
            if missing_score or used_fallback_content
            else raw_item.get("score_source", "model")
        ),
    }
    return item, operations


def recover_critiques(candidate_ids: Iterable[str]) -> dict:
    return CandidateCritiques.model_validate(
        {
            "critiques": [
                {
                    "candidate_id": candidate_id,
                    "grounding": 1,
                    "novelty": 1,
                    "falsifiability": 1,
                    "feasibility": 1,
                    "clarity": 1,
                    "issues": ["The critic model did not return a valid assessment."],
                    "required_revisions": [
                        "Require human review before relying on this candidate."
                    ],
                    "score_source": "fallback",
                }
                for candidate_id in sorted(set(candidate_ids))
            ]
        }
    ).model_dump()


__all__ = ["normalize_critique_item", "recover_critiques"]
