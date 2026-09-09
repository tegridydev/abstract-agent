from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent import schemas


def llm(**overrides: Any) -> dict[str, Any]:
    value = {
        "name": "Generate result",
        "type": "ollama_call",
        "model": "model:1",
        "inputs": ["topic"],
        "outputs": ["result"],
        "prompt_template": "Generate for {topic}",
    }
    value.update(overrides)
    return value


def tool(name: str = "Run tool", **overrides: Any) -> dict[str, Any]:
    value = {
        "name": name,
        "type": "tool_call",
        "function": "example.tool",
        "inputs": ["topic"],
        "outputs": ["result"],
    }
    value.update(overrides)
    return value


def candidate(**overrides: Any) -> dict[str, Any]:
    value = {
        "candidate_id": "c1",
        "title": "Title",
        "hypothesis": "Hypothesis",
        "mechanism": "Mechanism",
        "novelty_delta": "Delta",
        "supporting_paper_ids": ["p1"],
        "predictions": ["Prediction"],
        "falsifier": "Falsifier",
        "proposed_experiment": "Experiment",
        "novel_insights": ["Insight"],
        "applicability": "broad",
    }
    value.update(overrides)
    return value


def test_host_validator_none_scheme_credentials_invalid_port_and_trailing_slash() -> None:
    assert schemas._validate_ollama_host(None) is None
    assert schemas._validate_ollama_host("http://localhost:11434/") == ("http://localhost:11434")
    for host in (
        "relative",
        "ftp://host",
        f"http://{SYNTHETIC_CREDENTIALS}@host",
        "http://host:bad",
    ):
        with pytest.raises(ValueError):
            schemas._validate_ollama_host(host)


def test_model_assignment_resolution_precedence() -> None:
    assignments = schemas.ModelAssignments(all_model="all", creative="creative")
    assert assignments.resolve("creative", "configured") == "creative"
    assert assignments.resolve("writing", "configured") == "all"
    assert assignments.resolve("general", "configured") == "all"


def test_step_context_keys_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="context keys must be unique"):
        schemas.LlmStep.model_validate(llm(inputs=["topic", "topic"]))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"outputs": ["one", "two"]}, "exactly one output"),
        ({"prompt_template": "Generate {unknown}"}, "placeholders"),
        ({"response_schema": "unknown"}, "unknown response schema"),
        ({"max_output_tokens": 512, "repair_output_tokens": 128}, "cannot be below"),
        ({"allow_partial_recovery": True}, "requires a response schema"),
        ({"variants": ["one", "one"]}, "variants must be unique"),
        ({"map_input": "missing"}, "map_input must be declared"),
        (
            {"map_input": "topic", "variants": ["one"]},
            "cannot combine variants and map_input",
        ),
    ],
)
def test_llm_step_semantic_validation(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        schemas.LlmStep.model_validate(llm(**overrides))


@pytest.mark.parametrize(
    ("bindings", "params", "message"),
    [
        ({"not-valid": "topic"}, {}, "invalid function parameter"),
        ({"query": "missing"}, {}, "undeclared context keys"),
        ({"query": "topic"}, {"query": "fixed"}, "collide with params"),
        ({"topic": "other"}, {}, "duplicate implicit arguments"),
    ],
)
def test_tool_binding_validation(
    bindings: dict[str, str], params: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        schemas.ToolStep.model_validate(
            tool(inputs=["topic", "other"], input_bindings=bindings, params=params)
        )


def test_pipeline_data_flow_rejects_duplicate_missing_overwrite_and_final() -> None:
    cases = [
        ([tool(), tool()], "duplicate step name"),
        ([tool(inputs=["missing"])], "unavailable inputs"),
        ([tool(outputs=["topic"])], "overwrites context keys"),
    ]
    for steps, message in cases:
        with pytest.raises(ValidationError, match=message):
            schemas.PipelineConfig.model_validate({"context_init": ["topic"], "steps": steps})
    with pytest.raises(ValidationError, match="final outputs are unavailable"):
        schemas.PipelineConfig.model_validate(
            {"context_init": ["topic"], "steps": [tool()], "final_outputs": ["missing"]}
        )


def test_hypothesis_candidate_applicability_rules() -> None:
    with pytest.raises(ValidationError, match="require target_domain"):
        schemas.HypothesisCandidate.model_validate(candidate(applicability="domain_specific"))
    with pytest.raises(ValidationError, match="only domain specific"):
        schemas.HypothesisCandidate.model_validate(candidate(target_domain="Imaging"))
    assert (
        schemas.HypothesisCandidate.model_validate(
            candidate(applicability="domain_specific", target_domain="Imaging")
        ).target_domain
        == "Imaging"
    )


def test_novelty_verification_relationship_rules() -> None:
    base = {
        "candidate_id": "c1",
        "status": "uncertain",
        "reasoning": "Reasoning",
        "closest_prior_art_paper_ids": ["p1"],
    }
    with pytest.raises(ValidationError, match="included in closest"):
        schemas.NoveltyVerification.model_validate(
            {**base, "materially_matching_paper_ids": ["p2"]}
        )
    with pytest.raises(ValidationError, match="cannot include materially matching"):
        schemas.NoveltyVerification.model_validate(
            {
                **base,
                "status": "novelty_supported",
                "materially_matching_paper_ids": ["p1"],
            }
        )


def test_claim_and_research_reference_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="paper_ids must be unique"):
        schemas.ClaimEvidenceLink.model_validate(
            {
                "claim": "Claim",
                "paper_ids": ["p1", "p1"],
                "relationship": "supports",
                "confidence": "high",
            }
        )
    proposal = {
        "proposal_status": "novelty_supported",
        "title": "Title",
        "hypothesis": "Hypothesis",
        "rationale": "Rationale",
        "proposed_method": "Method",
        "predictions": ["Prediction"],
        "falsifier": "Falsifier",
        "novel_insights": ["Insight"],
        "claim_evidence": [
            {
                "claim": "Claim",
                "paper_ids": ["p1"],
                "relationship": "supports",
                "confidence": "high",
            }
        ],
        "abstract": "Abstract",
        "references": ["p1", "p1"],
    }
    with pytest.raises(ValidationError, match="references must be unique"):
        schemas.ResearchProposal.model_validate(proposal)


@pytest.mark.parametrize(
    "field", ["failed_query_count", "cached_query_count", "warning_query_count"]
)
def test_source_status_query_subcounts_cannot_exceed_total(field: str) -> None:
    with pytest.raises(ValidationError, match="cannot exceed query_count"):
        schemas.SourceStatus.model_validate(
            {
                "provider": "p",
                "status": "partial",
                "result_count": 0,
                "duration_seconds": 0,
                "query_count": 1,
                field: 2,
            }
        )


SYNTHETIC_CREDENTIALS = "user:pass"
