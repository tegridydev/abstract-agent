from __future__ import annotations

from agent.agent import PipelineManager
from agent.agent_helpers import choose_verified_candidate, validate_research_proposal
from agent.literature.relevance import assess_relevance
from agent.pipeline.stages import build_conservative_proposal, project_hypothesis_candidate
from agent.schemas import PipelineConfig


def paper(paper_id: str, title: str, summary: str, role: str | None = None) -> dict:
    value = {
        "paper_id": paper_id,
        "provider": "test",
        "provider_id": paper_id,
        "source": "test",
        "title": title,
        "summary": summary,
        "relevance_score": 0.9,
        "composite_score": 0.8,
    }
    if role:
        value["relevance_judgement"] = {"role": role}
    return value


def evidence() -> tuple[list[dict], dict]:
    papers = [
        paper(
            "paper:awq",
            "Activation-aware Weight Quantization for LLM Compression",
            "Activation-aware quantization compresses large language models while preserving "
            "important weights.",
            "direct_evidence",
        ),
        paper(
            "paper:svd",
            "Singular Value Methods for Language Model Compression",
            "Low-rank singular value truncation reduces language model parameter storage.",
            "transferable_method",
        ),
        paper(
            "paper:cache",
            "Efficient KV Cache Compression",
            "Cache compression reduces memory during large language model inference.",
            "direct_evidence",
        ),
    ]
    ledger = {
        "cards": [
            {
                "paper_id": item["paper_id"],
                "findings": [
                    {
                        "claim": item["summary"],
                        "source_passage": item["summary"],
                    }
                ],
                "relevance": item["summary"],
            }
            for item in papers
        ]
    }
    return papers, ledger


def test_deterministic_relevance_cannot_shift_rationales_between_papers() -> None:
    candidates = [
        paper(
            "paper:awq",
            "Activation-aware Weight Quantization for LLM Compression",
            "AWQ quantizes large language model weights for efficient inference.",
        ),
        paper(
            "paper:image",
            "Lossless Microscopy Image Serialization",
            "A lossless image codec serializes microscopy pixels and metadata.",
        ),
    ]

    accepted, decisions, rejected = assess_relevance(
        candidates,
        "llm compression",
        research_plan={"required_concepts": ["llm compression"]},
    )

    assert [item["paper_id"] for item in accepted] == ["paper:awq"]
    assert decisions[0]["decision_source"] == "deterministic_hybrid"
    assert "AWQ quantizes" in decisions[0]["source_passage"]
    assert rejected[0]["paper_id"] == "paper:image"
    assert "AWQ" not in rejected[0]["rationale"]


def test_prior_art_relevance_retains_a_quota_for_each_candidate() -> None:
    revised = {
        "candidate_id": "winner",
        "title": "Saliency-aware LLM compression",
        "hypothesis": "Saliency can guide weight compression.",
        "mechanism": "Protect salient weights.",
    }
    alternate = {
        "candidate_id": "alternate",
        "title": "Domain-agnostic LLM compression",
        "hypothesis": "A shared protocol compresses structured payloads.",
        "mechanism": "Encode repeated structures.",
    }
    candidates = []
    for index in range(3):
        item = paper(
            f"paper:w{index}",
            f"Saliency-aware LLM weight compression {index}",
            "Saliency protects important LLM weights during model compression.",
        )
        item["matched_candidate_ids"] = ["winner"]
        candidates.append(item)
    for index in range(6):
        item = paper(
            f"paper:a{index}",
            f"Domain-agnostic LLM payload compression {index}",
            "A domain-agnostic protocol compresses structured LLM payload data.",
        )
        item["matched_candidate_ids"] = ["alternate"]
        item["composite_score"] = 0.99
        candidates.append(item)

    accepted, _, _ = assess_relevance(
        candidates,
        "llm compression",
        revised_candidate=revised,
        alternate_candidate=alternate,
        max_results=12,
        minimum_score=0.2,
    )

    assert sum("winner" in item["matched_candidate_ids"] for item in accepted) == 3
    assert sum("alternate" in item["matched_candidate_ids"] for item in accepted) == 6


def test_granite_style_invalid_candidate_is_projected_and_bounded() -> None:
    papers, ledger = evidence()
    raw = {
        "candidate_id": "wrong",
        "title": "Compression direction",
        "hypothesis": "Compression-aware calibration improves deployment robustness.",
        "mechanism": "Calibration preserves sensitive activations.",
        "novelty_delta": "Combine calibration and low-rank compression.",
        "supporting_paper_ids": [],
        "predictions": ["Not reported"] * 10,
        "falsifier": "No reproducible difference.",
        "proposed_experiment": "Compare calibrated and uncalibrated compressed models.",
        "risks": [],
        "novel_insights": ["Not reported"] * 10,
        "applicability": "broad",
        "target_domain": "domain_specific",
    }

    projected = project_hypothesis_candidate(
        raw,
        topic="llm compression",
        candidate_id="mechanism",
        research_scope={"mode": "broad", "label": "Broad topic"},
        evidence_scope_options=[],
        evidence_ledger=ledger,
        evidence_papers=papers,
        model="granite4:350m",
    )

    assert projected["candidate_id"] == "mechanism"
    assert projected["applicability"] == "broad"
    assert projected["target_domain"] is None
    assert 1 <= len(projected["supporting_paper_ids"]) <= 2
    assert projected["predictions"] == ["Not reported"]
    assert projected["novel_insights"] == ["Not reported"]
    assert projected["generation_provenance"]["mode"] == "draft_projected"
    assert projected["generation_provenance"]["human_review_required"] is True


def test_unverified_proposal_rewrites_overclaims_and_reports_two_coverages() -> None:
    papers, ledger = evidence()
    candidate = project_hypothesis_candidate(
        {
            "title": "Novel Saliency Compression",
            "hypothesis": "A novel compressor will improve model quality.",
            "mechanism": "Saliency protects important weights.",
            "novelty_delta": "The first saliency mechanism.",
            "predictions": ["SAC will outperform every baseline."],
            "falsifier": "No replicated improvement.",
            "proposed_experiment": "Compare SAC with fixed baselines.",
            "risks": ["Benchmark sensitivity"],
            "novel_insights": ["New understanding of saliency"],
        },
        topic="llm compression",
        candidate_id="mechanism",
        research_scope={"mode": "broad", "label": "Broad topic"},
        evidence_scope_options=[],
        evidence_ledger=ledger,
        evidence_papers=papers,
        generation_mode="draft_projected",
    )
    novelty = {"status": "uncertain", "search_breadth": {"adequate": False}}

    proposal = build_conservative_proposal(
        topic="llm compression",
        final_candidate=candidate,
        novelty_verification=novelty,
        evidence_papers=papers,
        model_content={
            "abstract": "This novel method will outperform every existing method.",
            "limitations_and_risks": ["Benchmark sensitivity", "Benchmark sensitivity"],
        },
        generation_mode="draft_projected",
    )
    _, audit = validate_research_proposal(proposal, papers, novelty)

    rendered = " ".join(
        [proposal["title"], proposal["hypothesis"], proposal["abstract"], *proposal["predictions"]]
    ).casefold()
    assert "novel method" not in rendered
    assert "will outperform" not in rendered
    assert proposal["predictions"][0].startswith("Test whether ")
    assert len(proposal["limitations_and_risks"]) == len(set(proposal["limitations_and_risks"]))
    assert audit["claim_grounding_coverage"] == 1.0
    assert audit["claim_support_coverage"] == 0.0
    assert audit["review_reasons"] == [
        "novelty_not_established",
        "model_output_recovered",
        "prior_art_breadth_inadequate",
    ]


def test_normalized_exact_proposal_can_retain_verified_status() -> None:
    papers, ledger = evidence()
    candidate = project_hypothesis_candidate(
        {},
        topic="llm compression",
        candidate_id="mechanism",
        research_scope={"mode": "broad", "label": "Broad topic"},
        evidence_scope_options=[],
        evidence_ledger=ledger,
        evidence_papers=papers,
        generation_mode="structured_normalized",
    )
    proposal = build_conservative_proposal(
        topic="llm compression",
        final_candidate=candidate,
        novelty_verification={"status": "novelty_supported"},
        evidence_papers=papers,
        model_content={
            "title": "Verified direction",
            "hypothesis": "Calibration changes compression robustness.",
            "rationale": "The cited evidence motivates the comparison.",
            "proposed_method": "Compare calibrated and control models.",
            "predictions": ["Calibration improves robustness."],
            "falsifier": "No replicated difference.",
            "novel_insights": ["Calibration transfers across methods."],
            "limitations_and_risks": ["Dataset sensitivity"],
            "abstract": "A controlled study of calibration and compression robustness.",
        },
        generation_mode="structured_normalized",
    )

    assert proposal["proposal_status"] == "novelty_supported"
    assert proposal["generation_provenance"]["human_review_required"] is False


def test_llama_style_domain_metadata_is_application_owned_under_broad_scope() -> None:
    papers, ledger = evidence()
    raw = {
        "title": "Broad compression hypothesis",
        "hypothesis": "A shared calibration objective improves multiple compression methods.",
        "mechanism": "The objective protects activation outliers.",
        "novelty_delta": "Unifies two compression families.",
        "predictions": ["Calibration changes robustness."],
        "falsifier": "No change across methods.",
        "proposed_experiment": "Compare methods on shared benchmarks.",
        "novel_insights": ["A shared objective may transfer."],
        "applicability": "domain_specific",
        "target_domain": "NLP",
    }

    projected = project_hypothesis_candidate(
        raw,
        topic="llm compression",
        candidate_id="method-or-measurement",
        research_scope={"mode": "broad", "label": "Broad topic"},
        evidence_scope_options=[],
        evidence_ledger=ledger,
        evidence_papers=papers,
        model="llama3.2:3b",
        semantic_draft="A shared calibration objective may transfer across compression methods.",
    )

    assert projected["applicability"] == "broad"
    assert projected["target_domain"] is None
    assert {link["paper_id"] for link in projected["grounding_links"]} == set(
        projected["supporting_paper_ids"]
    )


def test_domain_and_cross_domain_projection_use_application_scope_policy() -> None:
    papers, ledger = evidence()
    scopes = [
        {"label": "Weights", "paper_ids": ["paper:awq", "paper:svd"]},
        {"label": "Caches", "paper_ids": ["paper:cache"]},
    ]
    domain = project_hypothesis_candidate(
        {},
        topic="llm compression",
        candidate_id="mechanism",
        research_scope={
            "mode": "domain",
            "label": "Weights",
            "supporting_paper_ids": ["paper:awq", "paper:svd"],
        },
        evidence_scope_options=scopes,
        evidence_ledger=ledger,
        evidence_papers=papers,
    )
    cross_domain = project_hypothesis_candidate(
        {},
        topic="llm compression",
        candidate_id="cross-domain-transfer",
        research_scope={"mode": "broad", "label": "Broad topic"},
        evidence_scope_options=scopes,
        evidence_ledger=ledger,
        evidence_papers=papers,
    )

    assert domain["applicability"] == "domain_specific"
    assert domain["target_domain"] == "Weights"
    assert domain["supporting_paper_ids"][0] in {"paper:awq", "paper:svd"}
    assert cross_domain["applicability"] == "cross_domain"
    assert "paper:cache" in cross_domain["supporting_paper_ids"]


def test_projected_candidate_forces_uncertain_novelty() -> None:
    papers, ledger = evidence()
    candidate = project_hypothesis_candidate(
        {},
        topic="llm compression",
        candidate_id="mechanism",
        research_scope={"mode": "broad", "label": "Broad topic"},
        evidence_scope_options=[],
        evidence_ledger=ledger,
        evidence_papers=papers,
    )
    alternate = {**candidate, "candidate_id": "alternate"}
    verifications = {
        "verifications": [
            {
                "candidate_id": "mechanism",
                "status": "novelty_supported",
                "materially_matching_paper_ids": [],
                "limitations": [],
            },
            {
                "candidate_id": "alternate",
                "status": "uncertain",
                "materially_matching_paper_ids": [],
                "limitations": [],
            },
        ]
    }

    _, verification = choose_verified_candidate(candidate, alternate, verifications)

    assert verification["status"] == "uncertain"
    assert any("projection" in item for item in verification["limitations"])


def test_conservative_proposal_is_valid_and_source_bound() -> None:
    papers, ledger = evidence()
    candidate = project_hypothesis_candidate(
        {},
        topic="llm compression",
        candidate_id="mechanism",
        research_scope={"mode": "broad", "label": "Broad topic"},
        evidence_scope_options=[],
        evidence_ledger=ledger,
        evidence_papers=papers,
    )
    proposal = build_conservative_proposal(
        topic="llm compression",
        final_candidate=candidate,
        novelty_verification={"status": "uncertain"},
        evidence_papers=papers,
        model="granite4:350m",
    )

    audited, audit = validate_research_proposal(proposal, papers, {"status": "uncertain"})

    assert audited["proposal_status"] == "unverified_research_direction"
    assert audited["generation_provenance"]["human_review_required"] is True
    assert set(audited["references"]) <= {item["paper_id"] for item in papers}
    assert audit["evidence_coverage"] > 0


def test_pipeline_manager_completes_hypothesis_stage_via_draft_projection() -> None:
    papers, ledger = evidence()
    config = PipelineConfig.model_validate(
        {
            "version": "4.5",
            "context_init": [
                "topic",
                "research_scope",
                "evidence_scope_options",
                "evidence_ledger",
                "gap_map",
                "formatted_papers",
                "top_papers",
            ],
            "steps": [
                {
                    "name": "Generate Independent Hypotheses",
                    "type": "ollama_call",
                    "model": "tiny:test",
                    "inputs": [
                        "topic",
                        "research_scope",
                        "evidence_ledger",
                        "gap_map",
                        "formatted_papers",
                    ],
                    "outputs": ["hypothesis_candidates"],
                    "prompt_template": "Topic {topic}; scope {research_scope}; ledger "
                    "{evidence_ledger}; gaps {gap_map}; papers {formatted_papers}; axis {axis}",
                    "variants": ["mechanism"],
                    "variant_key": "axis",
                    "response_schema": "hypothesis_candidate",
                    "generation_strategy": "draft_conditioned",
                    "invalid_output": "recover",
                }
            ],
            "final_outputs": ["hypothesis_candidates"],
        }
    )

    class FakeAgent:
        def draft(self, prompt: str, **kwargs):
            del prompt, kwargs
            return "Calibration may protect activation outliers during compression.", {
                "request_count": 1,
                "response_characters": 64,
            }

        def ask(self, prompt: str, **kwargs):
            del prompt
            raw = {
                "title": "Calibration-aware compression",
                "hypothesis": "Calibration improves compressed-model robustness.",
                "mechanism": "Calibration protects activation outliers.",
                "novelty_delta": "A combined calibration objective.",
                "supporting_paper_ids": [],
                "predictions": ["Robustness changes."],
                "falsifier": "No reproducible change.",
                "proposed_experiment": "Compare calibrated and control models.",
                "novel_insights": ["Calibration may transfer."],
                "applicability": "domain_specific",
                "target_domain": "NLP",
            }
            recovered = kwargs["final_recovery"](raw, ValueError("invalid contract"))
            recovered = kwargs["response_validator"](recovered)
            return recovered, {"request_count": 1, "recovery_mode": "conservative_fallback"}

        def close(self) -> None:
            return None

    manager = PipelineManager(config, agent_factory=lambda *args: FakeAgent())
    manager._validated_model_metadata = {"tiny:test": {"allocated_context_length": 16_384}}
    context = manager.run_pipeline(
        {
            "topic": "llm compression",
            "research_scope": {"mode": "broad", "label": "Broad topic"},
            "evidence_scope_options": {"options": []},
            "evidence_ledger": ledger,
            "gap_map": {"unresolved_questions": []},
            "formatted_papers": "source backed papers",
            "top_papers": papers,
        }
    )

    assert context["_outcome"] == "proposal_complete"
    assert context["hypothesis_candidates"][0]["applicability"] == "broad"
    assert context["hypothesis_candidates"][0]["supporting_paper_ids"]
    assert context["_llm_metrics"]["Generate Independent Hypotheses"][0]["request_count"] == 2
