from __future__ import annotations

import pytest

from agent.agent import PipelineManager
from agent.contracts.pipeline import PipelineConfig
from agent.pipeline.errors import ModelOutputError
from agent.pipeline.stages.hypotheses import (
    project_hypothesis_candidate,
    resolve_hypothesis_axis,
)


def _paper(paper_id: str) -> dict:
    return {
        "paper_id": paper_id,
        "relevance_judgement": {"role": "direct_evidence"},
    }


def _card(paper_id: str) -> dict:
    return {
        "paper_id": paper_id,
        "findings": [
            {
                "claim": f"Finding for {paper_id}",
                "source_passage": f"Measured evidence passage for {paper_id}.",
            }
        ],
        "method": "Controlled comparison.",
        "limitations": [],
        "relevance": "Direct evidence.",
    }


def test_selected_scope_nested_alternative_and_unscoped_papers_substitute_boundary_axis() -> None:
    selected = {f"paper:{index}" for index in range(10)}
    core = selected | {"paper:unscoped-a", "paper:unscoped-b"}
    resolution = resolve_hypothesis_axis(
        "cross-domain-application",
        {
            "mode": "domain",
            "label": "LLM Weight Compression",
            "supporting_paper_ids": sorted(selected),
        },
        [
            {"scope_id": "weights", "paper_ids": sorted(selected)},
            {"scope_id": "awq", "paper_ids": ["paper:1", "paper:2"]},
        ],
        core,
    )

    assert resolution.configured_axis == "cross-domain-application"
    assert resolution.effective_axis == "boundary-condition-or-failure-mode"
    assert resolution.secondary_scope_ids == ()
    assert "outside the selected scope" in str(resolution.substitution_reason)


def test_selected_scope_with_independent_alternative_keeps_cross_domain_axis() -> None:
    resolution = resolve_hypothesis_axis(
        "cross-domain-application",
        {"mode": "domain", "label": "Weights", "supporting_paper_ids": ["paper:a"]},
        [
            {"scope_id": "weights", "paper_ids": ["paper:a"]},
            {"scope_id": "caches", "paper_ids": ["paper:b"]},
        ],
        {"paper:a", "paper:b"},
    )

    assert resolution.effective_axis == "cross-domain-application"
    assert resolution.primary_scope_ids == ("paper:a",)
    assert resolution.secondary_scope_ids == ("paper:b",)


def test_broad_scope_requires_mutually_exclusive_evidence() -> None:
    nested = resolve_hypothesis_axis(
        "cross-domain-application",
        {"mode": "broad", "label": "Broad topic"},
        [
            {"scope_id": "all", "paper_ids": ["paper:a", "paper:b"]},
            {"scope_id": "nested", "paper_ids": ["paper:b"]},
        ],
        {"paper:a", "paper:b"},
    )
    overlapping = resolve_hypothesis_axis(
        "cross-domain-application",
        {"mode": "broad", "label": "Broad topic"},
        [
            {"scope_id": "left", "paper_ids": ["paper:a", "paper:shared"]},
            {"scope_id": "right", "paper_ids": ["paper:b", "paper:shared"]},
        ],
        {"paper:a", "paper:b", "paper:shared"},
    )

    assert nested.effective_axis == "boundary-condition-or-failure-mode"
    assert overlapping.effective_axis == "cross-domain-application"
    assert overlapping.primary_scope_ids == ("paper:a",)
    assert overlapping.secondary_scope_ids == ("paper:b",)


def test_projection_uses_selected_scope_after_cross_domain_substitution() -> None:
    paper_ids = ["paper:a", "paper:b", "paper:unscoped"]
    candidate = project_hypothesis_candidate(
        {},
        topic="llm model weights compression",
        candidate_id="cross-domain-application",
        research_scope={
            "mode": "domain",
            "label": "LLM Weight Compression",
            "supporting_paper_ids": ["paper:a", "paper:b"],
        },
        evidence_scope_options=[
            {"scope_id": "weights", "paper_ids": ["paper:a", "paper:b"]},
            {"scope_id": "nested", "paper_ids": ["paper:b"]},
        ],
        evidence_ledger={"cards": [_card(paper_id) for paper_id in paper_ids]},
        evidence_papers=[_paper(paper_id) for paper_id in paper_ids],
    )

    assert candidate["candidate_id"] == "boundary-condition-or-failure-mode"
    assert candidate["applicability"] == "domain_specific"
    assert candidate["target_domain"] == "LLM Weight Compression"
    assert set(candidate["supporting_paper_ids"]) <= {"paper:a", "paper:b"}
    provenance = candidate["generation_provenance"]
    assert provenance["configured_axis"] == "cross-domain-application"
    assert provenance["effective_axis"] == "boundary-condition-or-failure-mode"
    assert provenance["axis_substitution_reason"]


def test_projection_uses_one_exclusive_paper_from_each_genuine_domain() -> None:
    candidate = project_hypothesis_candidate(
        {},
        topic="llm compression",
        candidate_id="cross-domain-application",
        research_scope={
            "mode": "domain",
            "label": "Weights",
            "supporting_paper_ids": ["paper:a"],
        },
        evidence_scope_options=[
            {"scope_id": "weights", "paper_ids": ["paper:a"]},
            {"scope_id": "caches", "paper_ids": ["paper:b"]},
        ],
        evidence_ledger={"cards": [_card("paper:a"), _card("paper:b")]},
        evidence_papers=[_paper("paper:a"), _paper("paper:b")],
    )

    assert candidate["candidate_id"] == "cross-domain-application"
    assert candidate["applicability"] == "cross_domain"
    assert set(candidate["supporting_paper_ids"]) == {"paper:a", "paper:b"}


def _pipeline_config() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "version": "5.0",
            "context_init": [
                "topic",
                "research_scope",
                "evidence_scope_options",
                "evidence_ledger",
                "gap_map",
                "top_papers",
            ],
            "steps": [
                {
                    "name": "Generate Independent Hypotheses",
                    "type": "ollama_call",
                    "model": "test:model",
                    "inputs": ["topic", "research_scope", "evidence_ledger", "gap_map"],
                    "outputs": ["hypothesis_candidates"],
                    "prompt_template": "axis={innovation_axis}",
                    "variants": [
                        "mechanism",
                        "method-or-measurement",
                        "cross-domain-application",
                    ],
                    "variant_key": "innovation_axis",
                    "response_schema": "hypothesis_candidate",
                    "policy": "hypotheses.generate",
                    "prompt_version": "7",
                    "validation_attempts": 1,
                }
            ],
            "final_outputs": ["hypothesis_candidates"],
        }
    )


def _pipeline_context() -> dict:
    return {
        "topic": "llm model weights compression",
        "research_scope": {
            "mode": "domain",
            "label": "LLM Weight Compression",
            "supporting_paper_ids": ["paper:a", "paper:b"],
        },
        "evidence_scope_options": {
            "options": [
                {"scope_id": "weights", "paper_ids": ["paper:a", "paper:b"]},
                {"scope_id": "nested", "paper_ids": ["paper:b"]},
            ]
        },
        "evidence_ledger": {"cards": [_card("paper:a"), _card("paper:b")]},
        "gap_map": {"established": [], "contradictions": [], "unresolved_questions": []},
        "top_papers": [_paper("paper:a"), _paper("paper:b")],
    }


def _exact_candidate(candidate_id: str, *, invalid_cross_domain: bool = False) -> dict:
    return {
        "candidate_id": candidate_id,
        "title": f"Candidate {candidate_id}",
        "hypothesis": "A preregistered intervention changes the measured outcome.",
        "mechanism": "The intervention changes a measured mediator.",
        "novelty_delta": "The difference from prior art remains unverified.",
        "supporting_paper_ids": ["paper:a", "paper:b"],
        "conflicting_paper_ids": [],
        "predictions": ["The measured outcome changes by the preregistered threshold."],
        "falsifier": "The effect is absent or reverses.",
        "proposed_experiment": "Compare matched intervention and control units.",
        "risks": ["Dataset sensitivity."],
        "novel_insights": ["A measurable boundary may be identified."],
        "applicability": "cross_domain" if invalid_cross_domain else "domain_specific",
        "target_domain": (None if invalid_cross_domain else "LLM Weight Compression"),
        "grounding_links": [],
        "generation_provenance": {
            "mode": "structured_exact",
            "human_review_required": False,
        },
    }


class _AxisAgent:
    def __init__(self, *, fail_boundary_once: bool = False):
        self.calls: list[str] = []
        self.fail_boundary_once = fail_boundary_once

    def ask(self, prompt: str, **kwargs):
        candidate_id = str(kwargs["expected_candidate_id"])
        self.calls.append(candidate_id)
        invalid = self.fail_boundary_once and candidate_id == "boundary-condition-or-failure-mode"
        self.fail_boundary_once = False if invalid else self.fail_boundary_once
        return _exact_candidate(candidate_id, invalid_cross_domain=invalid), {
            "request_count": 1,
            "response_characters": 500,
            "generation_characters": 500,
            "compilation_outcome": "exact",
            "output_route": "content_schema",
            "format_mode": "schema",
        }

    def close(self):
        return None


def test_pipeline_substitutes_boundary_axis_before_model_generation() -> None:
    agent = _AxisAgent()
    manager = PipelineManager(_pipeline_config(), agent_factory=lambda *_args: agent)

    result = manager.run_pipeline(_pipeline_context())

    assert agent.calls == [
        "mechanism",
        "method-or-measurement",
        "boundary-condition-or-failure-mode",
    ]
    candidate = result["hypothesis_candidates"][2]
    provenance = candidate["generation_provenance"]
    assert provenance["configured_axis"] == "cross-domain-application"
    assert provenance["effective_axis"] == "boundary-condition-or-failure-mode"
    assert provenance["axis_substitution_reason"]
    assert any("replaced cross-domain-application" in item for item in result["_warnings"])


def test_post_generation_scope_error_checkpoints_and_resume_skips_completed_variants() -> None:
    agent = _AxisAgent(fail_boundary_once=True)
    manager = PipelineManager(_pipeline_config(), agent_factory=lambda *_args: agent)

    with pytest.raises(ModelOutputError, match="post generation validation"):
        manager.run_pipeline(_pipeline_context())

    pending = manager.last_context["_pending_step_outputs"]["Generate Independent Hypotheses"]
    assert pending["completed_count"] == 2
    assert pending["failed_variant"] == "cross-domain-application"
    assert pending["failed_effective_variant"] == "boundary-condition-or-failure-mode"
    assert manager.last_context["_failure_diagnostic"]["category"] == (
        "post_generation_validation_error"
    )

    result = manager.run_pipeline(
        manager.last_context,
        start_step="Generate Independent Hypotheses",
    )

    assert agent.calls == [
        "mechanism",
        "method-or-measurement",
        "boundary-condition-or-failure-mode",
        "boundary-condition-or-failure-mode",
    ]
    assert len(result["hypothesis_candidates"]) == 3
    assert "Generate Independent Hypotheses" not in result["_pending_step_outputs"]


def test_revision_preserves_substituted_effective_axis_and_provenance() -> None:
    config = PipelineConfig.model_validate(
        {
            "version": "5.0",
            "context_init": [
                "topic",
                "selected_candidate",
                "research_scope",
                "evidence_scope_options",
                "evidence_ledger",
                "top_papers",
            ],
            "steps": [
                {
                    "name": "Revise Winning Candidate",
                    "type": "ollama_call",
                    "model": "test:model",
                    "inputs": ["topic", "selected_candidate", "evidence_ledger"],
                    "outputs": ["revised_candidate"],
                    "prompt_template": "revise={selected_candidate}",
                    "response_schema": "hypothesis_candidate",
                    "policy": "hypotheses.revise",
                    "prompt_version": "7",
                    "validation_attempts": 1,
                }
            ],
            "final_outputs": ["revised_candidate"],
        }
    )
    context = _pipeline_context()
    context["selected_candidate"] = _exact_candidate("boundary-condition-or-failure-mode")
    context["selected_candidate"]["generation_provenance"].update(
        {
            "configured_axis": "cross-domain-application",
            "effective_axis": "boundary-condition-or-failure-mode",
            "axis_substitution_reason": "No independent alternative evidence scope.",
        }
    )
    agent = _AxisAgent()
    manager = PipelineManager(config, agent_factory=lambda *_args: agent)
    manager._validated_model_metadata = {"test:model": {"allocated_context_length": 16_384}}

    result = manager.run_pipeline(context)

    assert agent.calls == ["boundary-condition-or-failure-mode"]
    provenance = result["revised_candidate"]["generation_provenance"]
    assert provenance["configured_axis"] == "cross-domain-application"
    assert provenance["effective_axis"] == "boundary-condition-or-failure-mode"
    assert provenance["axis_substitution_reason"]
