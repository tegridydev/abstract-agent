from __future__ import annotations

import copy
import html
import json
from pathlib import Path

import pytest

import agent.agent as agent_module
from agent.agent import PipelineManager, load_pipeline_config
from agent.contracts.pipeline import PipelineConfig
from agent.contracts.sources import SourceStatus
from agent.llm.response_policies import (
    critique_generation_schema,
    hypothesis_generation_schema,
    proposal_generation_schema,
)
from agent.pipeline.config import load_config
from agent.pipeline.output.views import build_prompt_view
from agent.pipeline.stages.critiques import normalize_critique_item
from agent.pipeline.stages.hypotheses import project_hypothesis_candidate
from agent.pipeline.stages.planning import normalize_research_plan
from agent.pipeline.stages.proposal import build_conservative_proposal
from agent.pipeline.stages.validity import (
    audit_hypothesis_candidates,
    finalize_revision_candidates,
    gate_revised_candidate,
    prepare_revision_repair_packets,
)

PAPER_IDS = ("doi:10.1/calibration", "doi:10.1/quantization")


def _evidence_ledger() -> dict:
    return {
        "cards": [
            {
                "paper_id": PAPER_IDS[0],
                "findings": [
                    {
                        "claim": "Activation-aware calibration preserves outlier channels.",
                        "source_passage": (
                            "Activation-aware calibration preserves activation outlier channels "
                            "and held-out task accuracy after weight quantization."
                        ),
                    }
                ],
                "method": "Matched quantization experiments.",
                "limitations": [],
                "relevance": "Calibration and quantization evidence.",
            },
            {
                "paper_id": PAPER_IDS[1],
                "findings": [
                    {
                        "claim": "Quantization changes held-out accuracy.",
                        "source_passage": (
                            "Weight quantization was evaluated with held-out task accuracy on "
                            "matched pretrained language models."
                        ),
                    }
                ],
                "method": "Controlled model comparison.",
                "limitations": [],
                "relevance": "Language-model weight quantization evidence.",
            },
        ]
    }


def _papers() -> list[dict]:
    return [
        {
            "paper_id": paper_id,
            "relevance_judgement": {"role": "direct_evidence"},
        }
        for paper_id in PAPER_IDS
    ]


def _scientific_spec() -> dict:
    return {
        "title": "Calibration-aware weight quantization",
        "study_design": "experimental",
        "system": "pretrained language models under weight quantization",
        "exposure_or_intervention": {
            "name": "activation-aware quantization",
            "operational_definition": (
                "Quantize model weights with scales selected from a fixed calibration set."
            ),
        },
        "comparator": {
            "name": "weight-only quantization",
            "operational_definition": (
                "Quantize matched model weights without activation-derived scales."
            ),
        },
        "primary_outcome": {
            "name": "held-out task accuracy",
            "metric": "macro-average accuracy percentage",
            "expected_relation": "increases",
            "decision_rule": "The paired mean improvement exceeds 1 percentage point.",
        },
        "secondary_outcomes": [],
        "mechanism": {
            "cause": "activation-derived quantization scales",
            "mediator": "preservation of activation outlier channels",
            "measurable_consequence": "smaller held-out task accuracy loss",
        },
        "novelty_baseline": "weight-only quantization without activation calibration",
        "novelty_difference": "Preregistered testing of outlier preservation as a mediator.",
        "experiment": {
            "units": "three pretrained language models",
            "procedure": (
                "Create matched quantized copies and evaluate the same held-out task suite."
            ),
            "controls": ["identical base weights", "identical evaluation examples"],
        },
        "evidence_slots": ["E1", "E2"],
        "confounders": ["calibration-set composition"],
        "risks": ["model-family sensitivity"],
        "potential_insights": ["Outlier preservation may mediate post-quantization accuracy."],
    }


def _candidate() -> dict:
    return project_hypothesis_candidate(
        _scientific_spec(),
        topic="language model weight compression",
        candidate_id="mechanism",
        research_scope={
            "mode": "broad",
            "label": "Broad topic",
            "source": "default",
            "supporting_paper_ids": [],
        },
        evidence_scope_options=[],
        evidence_ledger=_evidence_ledger(),
        evidence_papers=_papers(),
        model="test:model",
        generation_mode="structured_exact",
    )


def test_v5_wire_contract_is_atomic_and_application_owned() -> None:
    hypothesis = hypothesis_generation_schema()
    critique = critique_generation_schema(["mechanism"])
    proposal = proposal_generation_schema()

    assert hypothesis["properties"]["primary_outcome"]["type"] == "object"
    assert "expected_relation" not in hypothesis["properties"]
    assert critique["properties"]["critiques"]["items"]["properties"]["candidate_id"]["enum"] == [
        "mechanism"
    ]
    assert "issues" not in critique["properties"]["critiques"]["items"]["properties"]
    assert set(proposal["properties"]) == {"rationale", "limitations_and_risks"}
    assert "abstract" not in proposal["properties"]


def test_valid_v5_candidate_renders_relation_once_and_passes_semantic_audit() -> None:
    candidate = _candidate()
    audited, audits = audit_hypothesis_candidates(
        [candidate], _evidence_ledger(), "language model weight compression"
    )

    assert candidate["hypothesis"].count("will be higher") == 1
    assert "noninferiority" not in candidate["falsifier"].casefold()
    assert audits[0]["status"] == "pass"
    assert audits[0]["grounding_tier"] in {"contextual", "mechanism_aligned"}
    assert audited[0]["generation_provenance"]["scientific_integrity"] == "pass"


def test_grounding_audit_uses_available_embeddings_without_owning_scientific_truth() -> None:
    class Embeddings:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    _, audits = audit_hypothesis_candidates(
        [_candidate()],
        _evidence_ledger(),
        "language model weight compression",
        embedding_service=Embeddings(),
    )

    assert audits[0]["grounding_method"] == "embedding"
    assert audits[0]["grounding_tier"] == "mechanism_aligned"
    assert audits[0]["status"] == "pass"


def test_legacy_and_fallback_candidates_are_never_reported_as_exact_v5_science() -> None:
    legacy = project_hypothesis_candidate(
        {
            "title": "Legacy compression study",
            "study_design": "experimental",
            "system": "language models",
            "exposure_or_intervention": "activation-aware compression",
            "comparator": "weight-only compression",
            "primary_outcome": "held-out accuracy",
            "expected_relation": "increases",
            "mechanism": "Activation calibration preserves outlier channels.",
            "novelty_delta": "Test the mediator directly.",
            "experiment": "Compare matched compressed models on held-out tasks.",
            "evidence_slots": ["E1", "E2"],
            "confounders": ["model family"],
            "risks": ["calibration sensitivity"],
            "secondary_predictions": [],
            "potential_insights": ["Calibration effects may transfer."],
        },
        topic="language model weight compression",
        candidate_id="legacy",
        research_scope={
            "mode": "broad",
            "label": "Broad topic",
            "source": "default",
            "supporting_paper_ids": [],
        },
        evidence_scope_options=[],
        evidence_ledger=_evidence_ledger(),
        evidence_papers=_papers(),
        generation_mode="structured_normalized",
    )
    fallback = project_hypothesis_candidate(
        {},
        topic="language model weight compression",
        candidate_id="fallback",
        research_scope={
            "mode": "broad",
            "label": "Broad topic",
            "source": "default",
            "supporting_paper_ids": [],
        },
        evidence_scope_options=[],
        evidence_ledger=_evidence_ledger(),
        evidence_papers=_papers(),
        generation_mode="conservative_fallback",
    )

    _, audits = audit_hypothesis_candidates(
        [legacy, fallback], _evidence_ledger(), "language model weight compression"
    )

    assert audits[0]["status"] == "revisable"
    assert audits[1]["status"] == "fail"
    assert all(candidate.get("scientific_spec") is None for candidate in (legacy, fallback))


def test_reported_compression_failure_pattern_is_rejected_before_prior_art() -> None:
    malformed = copy.deepcopy(_candidate())
    malformed["scientific_spec"]["exposure_or_intervention"] = {
        "name": "proposed method",
        "operational_definition": "unspecified method",
    }
    malformed["scientific_spec"]["primary_outcome"] = {
        "name": "PSNR will differ",
        "metric": "SSIM will differ",
        "expected_relation": "differs",
        "decision_rule": "Compare the two methods.",
    }
    malformed["hypothesis"] = (
        "Lossless image compression PSNR will differ between the proposed method and baseline."
    )
    malformed["proposed_experiment"] = (
        "Evaluate PSNR and SSIM for lossless images while controlling sampling frequency."
    )
    malformed["predictions"] = ["PSNR will differ will differ.", "PSNR will differ will differ."]

    _, audits = audit_hypothesis_candidates(
        [malformed], _evidence_ledger(), "data compression techniques"
    )
    categories = {item["category"] for item in audits[0]["findings"]}

    assert audits[0]["status"] == "fail"
    assert {"intervention", "outcome", "falsifiability", "internal_consistency"} <= categories

    _, revision_audit, outcome = gate_revised_candidate(
        malformed,
        [{"candidate_id": "mechanism", "unresolved_finding_ids": []}],
        _evidence_ledger(),
        "data compression techniques",
    )
    assert revision_audit["status"] == "fail"
    assert outcome["status"] == "insufficient_valid_candidates"
    assert outcome["proposal_eligible"] is False


def test_deterministic_finding_ids_are_content_derived_and_stable() -> None:
    placeholder = copy.deepcopy(_candidate())
    placeholder["scientific_spec"]["exposure_or_intervention"] = {
        "name": "proposed method",
        "operational_definition": "unspecified method",
    }
    identical_conditions = copy.deepcopy(_candidate())
    identical_conditions["scientific_spec"]["comparator"] = copy.deepcopy(
        identical_conditions["scientific_spec"]["exposure_or_intervention"]
    )

    _, first = audit_hypothesis_candidates(
        [placeholder], _evidence_ledger(), "language model weight compression"
    )
    _, repeated = audit_hypothesis_candidates(
        [placeholder], _evidence_ledger(), "language model weight compression"
    )
    _, different = audit_hypothesis_candidates(
        [identical_conditions], _evidence_ledger(), "language model weight compression"
    )

    first_id = first[0]["findings"][0]["finding_id"]
    assert first_id.startswith("finding:")
    assert first_id == repeated[0]["findings"][0]["finding_id"]
    assert first_id != different[0]["findings"][0]["finding_id"]


def test_revision_gate_requires_explicit_resolution_of_blocking_findings() -> None:
    candidate = _candidate()
    scorecard = [
        {
            "candidate_id": "mechanism",
            "unresolved_finding_ids": ["mechanism:critic:1"],
            "unresolved_major_revisions": ["Define the intervention."],
        }
    ]

    _, _, blocked = gate_revised_candidate(
        candidate, scorecard, _evidence_ledger(), "language model weight compression"
    )
    assert blocked["missing_addressed_finding_ids"] == ["mechanism:critic:1"]
    assert blocked["status"] == "insufficient_valid_candidates"

    resolved = copy.deepcopy(candidate)
    resolved["scientific_spec"]["addressed_finding_ids"] = ["mechanism:critic:1"]
    _, _, accepted = gate_revised_candidate(
        resolved, scorecard, _evidence_ledger(), "language model weight compression"
    )
    assert accepted["status"] == "candidate_valid"
    assert accepted["proposal_eligible"] is True


def test_revision_gate_rejects_a_claimed_finding_when_scientific_content_is_unchanged() -> None:
    selected = _candidate()
    claimed = copy.deepcopy(selected)
    claimed["scientific_spec"]["addressed_finding_ids"] = ["finding:experiment"]
    scorecard = [
        {
            "candidate_id": "mechanism",
            "unresolved_finding_ids": ["finding:experiment"],
            "unresolved_major_revisions": ["Make the procedure reproducible."],
            "blocking_findings": [
                {
                    "finding_id": "finding:experiment",
                    "source": "methodology_critic",
                    "category": "feasibility",
                    "severity": "major",
                    "field_paths": ["scientific_spec.experiment"],
                    "issue": "The procedure is underspecified.",
                    "required_revision": "Make the procedure reproducible.",
                }
            ],
        }
    ]

    _, _, outcome = gate_revised_candidate(
        claimed,
        scorecard,
        _evidence_ledger(),
        "language model weight compression",
        selected,
    )

    assert outcome["proposal_eligible"] is False
    assert outcome["falsely_addressed_finding_ids"] == ["finding:experiment"]
    assert outcome["missing_addressed_finding_ids"] == ["finding:experiment"]
    assert outcome["changed_scientific_fields"] == []


def test_one_focused_revision_repair_is_prepared_and_can_pass_the_authoritative_gate() -> None:
    selected = _candidate()
    initial = copy.deepcopy(selected)
    finding_id = "finding:experiment"
    scorecard = [
        {
            "candidate_id": "mechanism",
            "unresolved_finding_ids": [finding_id],
            "unresolved_major_revisions": ["Add a preregistered sensitivity analysis."],
            "blocking_findings": [
                {
                    "finding_id": finding_id,
                    "source": "methodology_critic",
                    "category": "feasibility",
                    "severity": "major",
                    "field_paths": ["scientific_spec.experiment"],
                    "issue": "The procedure lacks a sensitivity analysis.",
                    "required_revision": "Add a preregistered sensitivity analysis.",
                }
            ],
        }
    ]
    initial_candidate, initial_audit, initial_outcome = gate_revised_candidate(
        initial,
        scorecard,
        _evidence_ledger(),
        "language model weight compression",
        selected,
    )
    papers = [
        {
            **paper,
            "title": f"Weight compression evidence {index}",
        }
        for index, paper in enumerate(_papers(), start=1)
    ]

    packets = prepare_revision_repair_packets(
        initial_candidate,
        initial_audit,
        initial_outcome,
        scorecard,
        _evidence_ledger(),
        {"mode": "broad", "label": "Broad topic"},
        papers,
    )

    assert len(packets) == 1
    assert packets[0]["blocking_findings"][0]["finding_id"] == finding_id
    assert {card["slot"] for card in packets[0]["evidence_cards"]} == {"E1", "E2"}

    repair = copy.deepcopy(initial)
    repair["scientific_spec"]["experiment"]["procedure"] += (
        " Preregister a calibration-set sensitivity analysis before evaluation."
    )
    repair["scientific_spec"]["addressed_finding_ids"] = [finding_id]
    repaired, _, outcome = finalize_revision_candidates(
        initial_candidate,
        initial_audit,
        initial_outcome,
        [repair],
        scorecard,
        _evidence_ledger(),
        "language model weight compression",
        selected,
    )

    assert outcome["proposal_eligible"] is True
    assert outcome["attempt_count"] == 2
    assert outcome["repair_accepted"] is True
    assert outcome["repair_disposition"] == "accepted"
    assert repaired["generation_provenance"]["human_review_required"] is True


def test_focused_repair_must_address_findings_discovered_by_the_initial_gate() -> None:
    selected = _candidate()
    finding_id = "finding:new-gate-issue"
    initial_audit = {
        "candidate_id": "mechanism",
        "status": "revisable",
        "findings": [
            {
                "finding_id": finding_id,
                "source": "deterministic",
                "category": "feasibility",
                "severity": "major",
                "field_paths": ["scientific_spec.experiment"],
                "issue": "The gate found an incomplete procedure.",
                "required_revision": "Add a preregistered sensitivity analysis.",
            }
        ],
        "grounding_tier": "contextual",
    }
    initial_outcome = {
        "status": "insufficient_valid_candidates",
        "proposal_eligible": False,
        "failure_reasons": ["scientific_validation"],
        "missing_addressed_finding_ids": [],
    }
    repaired = copy.deepcopy(selected)
    repaired["scientific_spec"]["experiment"]["procedure"] += (
        " Preregister a calibration-set sensitivity analysis."
    )

    _, _, still_blocked = finalize_revision_candidates(
        selected,
        initial_audit,
        initial_outcome,
        [repaired],
        [],
        _evidence_ledger(),
        "language model weight compression",
        selected,
    )
    assert still_blocked["proposal_eligible"] is False
    assert still_blocked["missing_addressed_finding_ids"] == [finding_id]

    repaired["scientific_spec"]["addressed_finding_ids"] = [finding_id]
    _, _, accepted = finalize_revision_candidates(
        selected,
        initial_audit,
        initial_outcome,
        [repaired],
        [],
        _evidence_ledger(),
        "language model weight compression",
        selected,
    )
    assert accepted["proposal_eligible"] is True
    assert accepted["repair_disposition"] == "accepted"


def test_failed_or_axis_changing_revision_repair_finishes_gracefully() -> None:
    selected = _candidate()
    initial_candidate, initial_audit, initial_outcome = gate_revised_candidate(
        selected,
        [
            {
                "candidate_id": "mechanism",
                "unresolved_finding_ids": ["finding:one"],
                "unresolved_major_revisions": ["Revise the experiment."],
            }
        ],
        _evidence_ledger(),
        "language model weight compression",
        selected,
    )
    invalid_repair = copy.deepcopy(selected)
    invalid_repair["generation_provenance"]["mode"] = "draft_projected"

    retained, _, outcome = finalize_revision_candidates(
        initial_candidate,
        initial_audit,
        initial_outcome,
        [invalid_repair],
        [
            {
                "candidate_id": "mechanism",
                "unresolved_finding_ids": ["finding:one"],
                "unresolved_major_revisions": ["Revise the experiment."],
            }
        ],
        _evidence_ledger(),
        "language model weight compression",
        selected,
    )

    assert retained["candidate_id"] == "mechanism"
    assert outcome["status"] == "insufficient_valid_candidates"
    assert outcome["repair_attempted"] is True
    assert outcome["repair_accepted"] is False
    assert outcome["repair_disposition"] == "discarded_non_exact_repair"

    axis_changing_repair = copy.deepcopy(selected)
    axis_changing_repair["generation_provenance"]["effective_axis"] = "cross-domain-application"
    _, _, axis_outcome = finalize_revision_candidates(
        initial_candidate,
        initial_audit,
        initial_outcome,
        [axis_changing_repair],
        [
            {
                "candidate_id": "mechanism",
                "unresolved_finding_ids": ["finding:one"],
                "unresolved_major_revisions": ["Revise the experiment."],
            }
        ],
        _evidence_ledger(),
        "language model weight compression",
        selected,
    )
    assert axis_outcome["repair_disposition"] == "discarded_identity_or_axis_mismatch"


def test_valid_initial_revision_skips_repair_and_grounding_failure_gets_a_closed_packet() -> None:
    candidate = _candidate()
    validated, audit, valid_outcome = gate_revised_candidate(
        candidate,
        [{"candidate_id": "mechanism", "unresolved_finding_ids": []}],
        _evidence_ledger(),
        "language model weight compression",
        candidate,
    )
    assert (
        prepare_revision_repair_packets(
            validated,
            audit,
            valid_outcome,
            [],
            _evidence_ledger(),
            {"mode": "broad"},
            _papers(),
        )
        == []
    )
    final_candidate, _, final_outcome = finalize_revision_candidates(
        validated,
        audit,
        valid_outcome,
        [],
        [],
        _evidence_ledger(),
        "language model weight compression",
        candidate,
    )
    assert final_candidate["candidate_id"] == "mechanism"
    assert final_outcome["proposal_eligible"] is True

    packets = prepare_revision_repair_packets(
        {**candidate, "supporting_paper_ids": [*PAPER_IDS, "paper:not-in-ledger"]},
        {
            "findings": [
                "malformed finding",
                {
                    "finding_id": "finding:audit",
                    "source": "deterministic",
                    "category": "evidence",
                    "severity": "major",
                    "issue": "The support is too weak.",
                    "required_revision": "Use directly aligned evidence.",
                },
            ]
        },
        {
            "proposal_eligible": False,
            "failure_reasons": ["insufficient_grounding"],
        },
        [],
        _evidence_ledger(),
        {"mode": "broad"},
        [{**paper, "title": f"Evidence for {paper['paper_id']}"} for paper in _papers()],
    )
    finding_ids = {item["finding_id"] for item in packets[0]["blocking_findings"]}
    assert {"finding:audit", "gate:grounding"} <= finding_ids
    assert len(packets[0]["evidence_cards"]) == 2


def test_pipeline_orchestration_executes_exactly_one_mapped_revision_repair() -> None:
    selected = _candidate()
    finding_id = "finding:procedure"
    scorecard = [
        {
            "candidate_id": "mechanism",
            "unresolved_finding_ids": [finding_id],
            "unresolved_major_revisions": ["Add a sensitivity analysis."],
            "blocking_findings": [
                {
                    "finding_id": finding_id,
                    "source": "methodology_critic",
                    "category": "feasibility",
                    "severity": "major",
                    "field_paths": ["scientific_spec.experiment"],
                    "issue": "The procedure has no sensitivity analysis.",
                    "required_revision": "Add a sensitivity analysis.",
                }
            ],
        }
    ]
    repaired_spec = _scientific_spec()
    repaired_spec["experiment"]["procedure"] += (
        " Preregister a calibration-set sensitivity analysis."
    )
    repaired_spec["addressed_finding_ids"] = [finding_id]

    class RepairAgent:
        calls = 0

        def __init__(self, model, host, timeout):
            self.model = model

        def ask(self, prompt, **_kwargs):
            RepairAgent.calls += 1
            assert "finding:procedure" in prompt
            assert "<repair_packet>" in prompt
            return repaired_spec, {
                "request_count": 1,
                "compilation_outcome": "exact",
                "generation_characters": 400,
            }

        def close(self):
            return None

    config = PipelineConfig.model_validate(
        {
            "context_init": [
                "topic",
                "selected_candidate",
                "revised_candidate",
                "candidate_scorecard",
                "evidence_ledger",
                "research_scope",
                "top_papers",
                "evidence_scope_options",
                "evidence_papers",
            ],
            "steps": [
                {
                    "name": "Audit Initial Revision",
                    "type": "tool_call",
                    "function": "pipeline.stages.gate_revised_candidate",
                    "inputs": [
                        "revised_candidate",
                        "candidate_scorecard",
                        "evidence_ledger",
                        "topic",
                        "selected_candidate",
                    ],
                    "outputs": [
                        "initially_validated_revised_candidate",
                        "initial_revision_validity_audit",
                        "initial_revision_outcome",
                    ],
                },
                {
                    "name": "Prepare Revision Repair",
                    "type": "tool_call",
                    "function": "pipeline.stages.prepare_revision_repair_packets",
                    "inputs": [
                        "initially_validated_revised_candidate",
                        "initial_revision_validity_audit",
                        "initial_revision_outcome",
                        "candidate_scorecard",
                        "evidence_ledger",
                        "research_scope",
                        "top_papers",
                    ],
                    "outputs": ["revision_repair_packets"],
                },
                {
                    "name": "Repair Revised Candidate",
                    "type": "ollama_call",
                    "model": "model:test",
                    "inputs": ["revision_repair_packets"],
                    "outputs": ["revision_repair_candidates"],
                    "map_input": "revision_repair_packets",
                    "map_item_key": "repair_packet",
                    "prompt_template": "<repair_packet>{repair_packet}</repair_packet>",
                    "response_schema": "hypothesis_candidate",
                    "invalid_output": "recover",
                    "allow_partial_recovery": True,
                    "policy": "hypotheses.repair",
                    "prompt_version": "8",
                },
                {
                    "name": "Gate Revised Candidate",
                    "type": "tool_call",
                    "function": "pipeline.stages.finalize_revision_candidates",
                    "inputs": [
                        "initially_validated_revised_candidate",
                        "initial_revision_validity_audit",
                        "initial_revision_outcome",
                        "revision_repair_candidates",
                        "candidate_scorecard",
                        "evidence_ledger",
                        "topic",
                        "selected_candidate",
                    ],
                    "outputs": [
                        "validated_revised_candidate",
                        "revision_validity_audit",
                        "research_outcome",
                    ],
                },
            ],
        }
    )
    papers = [{**paper, "title": f"Evidence for {paper['paper_id']}"} for paper in _papers()]

    manager = PipelineManager(config, agent_factory=RepairAgent)
    manager._validated_model_metadata = {"model:test": {"allocated_context_length": 16_384}}
    context = manager.run_pipeline(
        {
            "topic": "language model weight compression",
            "selected_candidate": selected,
            "revised_candidate": selected,
            "candidate_scorecard": scorecard,
            "evidence_ledger": _evidence_ledger(),
            "research_scope": {"mode": "broad", "label": "Broad topic"},
            "top_papers": papers,
            "evidence_scope_options": {"options": []},
            "evidence_papers": papers,
        }
    )

    assert RepairAgent.calls == 1
    assert context["research_outcome"]["repair_disposition"] == "accepted"
    assert context["validated_revised_candidate"]["scientific_spec"]["addressed_finding_ids"] == [
        finding_id
    ]

    RepairAgent.calls = 0
    no_repair_context = PipelineManager(config, agent_factory=RepairAgent).run_pipeline(
        {
            "topic": "language model weight compression",
            "selected_candidate": selected,
            "revised_candidate": selected,
            "candidate_scorecard": [{"candidate_id": "mechanism", "unresolved_finding_ids": []}],
            "evidence_ledger": _evidence_ledger(),
            "research_scope": {"mode": "broad", "label": "Broad topic"},
            "top_papers": papers,
            "evidence_scope_options": {"options": []},
            "evidence_papers": papers,
        }
    )
    assert RepairAgent.calls == 0
    assert no_repair_context["revision_repair_candidates"] == []
    assert no_repair_context["research_outcome"]["repair_attempted"] is False


def test_protected_scientific_prompt_views_retain_their_stage_contracts() -> None:
    candidate = _candidate()
    candidates = []
    audits = []
    for candidate_id in ("mechanism", "boundary", "measurement"):
        item = copy.deepcopy(candidate)
        item["candidate_id"] = candidate_id
        item["title"] = f"{candidate_id} candidate"
        candidates.append(item)
        audits.append(
            {
                "candidate_id": candidate_id,
                "status": "pass",
                "findings": [],
                "grounding_tier": "contextual",
            }
        )
    cards = _evidence_ledger()["cards"] + [
        {
            "paper_id": "doi:10.1/awq",
            "findings": [
                {
                    "claim": "AWQ protects activation-salient channels during quantization.",
                    "source_passage": "Activation-aware scales protect salient weight channels.",
                }
            ],
            "method": "Activation-aware weight quantization.",
            "limitations": ["Model-family sensitivity."],
        },
        {
            "paper_id": "doi:10.1/huffman",
            "findings": [
                {
                    "claim": "Entropy coding reduces coded weight storage.",
                    "source_passage": "Huffman coding reduces storage for skewed symbols.",
                }
            ],
            "method": "Huffman entropy coding.",
            "limitations": ["Decode overhead."],
        },
    ]
    top_papers = [
        {
            "paper_id": card["paper_id"],
            "title": (
                "AWQ activation-aware quantization"
                if "awq" in card["paper_id"]
                else "Huffman entropy coding"
                if "huffman" in card["paper_id"]
                else f"Evidence {card['paper_id']}"
            ),
            "relevance_judgement": {"role": "direct_evidence"},
        }
        for card in cards
    ]
    context = {
        "topic": "language model weight compression",
        "research_scope": {"mode": "broad", "label": "Broad topic"},
        "gap_map": {
            "unresolved_questions": [
                {
                    "status": "unknown",
                    "statement": "Can AWQ saliency combine with Huffman coding?",
                    "paper_ids": ["doi:10.1/awq", "doi:10.1/huffman"],
                }
            ]
        },
        "evidence_ledger": {"cards": cards},
        "top_papers": top_papers,
        "audited_hypothesis_candidates": candidates,
        "candidate_validity_audits": audits,
        "selected_candidate": candidates[0],
        "candidate_scorecard": [
            {
                "candidate_id": "mechanism",
                "blocking_findings": [
                    {
                        "finding_id": "finding:complete",
                        "source": "evidence_critic",
                        "category": "evidence",
                        "severity": "major",
                        "issue": "The mechanism cites unrelated evidence.",
                        "required_revision": "Use AWQ or Huffman evidence that supports it.",
                    }
                ],
            }
        ],
    }

    def document(name: str, budget: int) -> dict:
        view = build_prompt_view(name, list(context), context, budget=budget)
        raw = view.text.removeprefix("<input_view>").removesuffix("</input_view>")
        return json.loads(html.unescape(raw))

    hypothesis = document("hypothesis_v5", 6_000)["data"]
    hypothesis_titles = {item["title"] for item in hypothesis["evidence_cards"]}
    assert {"AWQ activation-aware quantization", "Huffman entropy coding"} <= hypothesis_titles
    assert hypothesis["evidence_gaps"][0]["evidence_slots"]

    critique = document("critique_v5", 8_000)["data"]
    assert len(critique["candidates"]) == 3
    for item in critique["candidates"]:
        assert {"exposure_or_intervention", "primary_outcome", "mechanism", "experiment"} <= set(
            item["scientific_spec"]
        )
        assert item["deterministic_audit"]["candidate_id"] == item["candidate_id"]

    revision = document("revision_v5", 6_000)["data"]
    assert revision["blocking_findings"][0]["finding_id"] == "finding:complete"
    assert revision["blocking_findings"][0]["required_revision"].startswith("Use AWQ")
    assert revision["selected_candidate"]["scientific_spec"]["primary_outcome"]

    with pytest.raises(ValueError, match="protected prompt view"):
        build_prompt_view("critique_v5", list(context), context, budget=1_024)


def test_legacy_critic_issues_become_selection_blocking_findings() -> None:
    normalized, events = normalize_critique_item(
        {
            "candidate_id": "mechanism",
            "grounding": 4,
            "novelty": 3,
            "falsifiability": 2,
            "feasibility": 3,
            "clarity": 3,
            "issues": ["The intervention is undefined."],
            "required_revisions": ["Operationally define the intervention."],
        },
        expected_candidate_ids={"mechanism"},
    )

    assert normalized is not None
    assert normalized["findings"][0]["severity"] == "fatal"
    assert normalized["findings"][0]["category"] == "intervention"
    assert "undefined_intervention" in normalized["fatal_flags"]
    assert any(event.get("operation") == "derive" for event in events)


def test_critic_finding_ids_are_stable_application_owned_and_source_distinct() -> None:
    raw = {
        "candidate_id": "mechanism",
        "grounding": 3,
        "novelty": 3,
        "falsifiability": 3,
        "feasibility": 3,
        "clarity": 3,
        "findings": [
            {
                "finding_id": "model-controlled-id",
                "source": "legacy",
                "category": "evidence",
                "severity": "major",
                "issue": "The mechanism lacks direct evidence.",
                "required_revision": "Ground the mechanism in a supplied passage.",
            }
        ],
    }

    evidence, _ = normalize_critique_item(
        raw,
        expected_candidate_ids={"mechanism"},
        critic_source="evidence_critic",
    )
    evidence_repeat, _ = normalize_critique_item(
        raw,
        expected_candidate_ids={"mechanism"},
        critic_source="evidence_critic",
    )
    methodology, _ = normalize_critique_item(
        raw,
        expected_candidate_ids={"mechanism"},
        critic_source="methodology_critic",
    )

    evidence_finding = evidence["findings"][0]
    methodology_finding = methodology["findings"][0]
    assert evidence_finding["finding_id"].startswith("finding:")
    assert evidence_finding["finding_id"] != "model-controlled-id"
    assert evidence_finding["finding_id"] == evidence_repeat["findings"][0]["finding_id"]
    assert evidence_finding["finding_id"] != methodology_finding["finding_id"]
    assert evidence_finding["source"] == "evidence_critic"
    assert methodology_finding["source"] == "methodology_critic"


def test_planning_keeps_model_enrichment_but_restores_topic_anchor() -> None:
    plan, events = normalize_research_plan(
        {
            "topic_interpretation": "A model authored interpretation",
            "assumptions": ["A model authored assumption"],
            "required_concepts": ["unrelated domain"],
            "mechanism_query": "mechanism literature",
            "approaches_query": "approach literature",
            "prior_art_query": "prior art literature",
            "exclusions": [],
        },
        "LLM weight compression",
    )

    assert plan["topic_interpretation"] == "A model authored interpretation"
    assert plan["required_concepts"][0] == "LLM weight compression"
    assert events[0]["operation"] == "baseline_merge"


def test_partial_provider_request_can_be_both_successful_and_failed() -> None:
    status = SourceStatus(
        provider="semantic_scholar",
        status="partial",
        result_count=2,
        duration_seconds=0.2,
        query_count=1,
        planned_query_count=1,
        attempted_query_count=1,
        succeeded_query_count=1,
        failed_query_count=1,
        error_category="http_429",
    )

    assert status.succeeded_query_count == status.failed_query_count == 1


def test_pipeline_stops_after_an_insufficient_candidate_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = []

    def resolve(path: str):
        if path == "test.gate":
            return lambda topic: {
                "status": "insufficient_valid_candidates",
                "proposal_eligible": False,
                "findings": [{"issue": f"No valid candidate for {topic}"}],
            }

        def should_not_run(research_outcome):
            called.append(research_outcome)
            raise AssertionError("downstream prior-art/proposal work must not run")

        return should_not_run

    monkeypatch.setattr(agent_module, "_resolve_tool", resolve)
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Candidate Gate",
                    "type": "tool_call",
                    "function": "test.gate",
                    "inputs": ["topic"],
                    "outputs": ["research_outcome"],
                },
                {
                    "name": "Prior Art",
                    "type": "tool_call",
                    "function": "test.prior_art",
                    "inputs": ["research_outcome"],
                    "outputs": ["proposal"],
                },
            ],
        }
    )

    context = PipelineManager(config).run_pipeline({"topic": "compression"})

    assert called == []
    assert context["_outcome"] == "insufficient_valid_candidates"
    assert "proposal" not in context


def test_proposal_abstract_is_composed_from_the_validated_candidate() -> None:
    candidate = _candidate()
    proposal = build_conservative_proposal(
        topic="language model weight compression",
        final_candidate=candidate,
        novelty_verification={"status": "uncertain"},
        evidence_papers=[{"paper_id": paper_id, "title": "Evidence"} for paper_id in PAPER_IDS],
        model_content={
            "rationale": "Model-authored rationale.",
            "limitations_and_risks": ["One model authored limitation."],
            "abstract": "This must never be copied.",
        },
        generation_mode="structured_exact",
    )

    assert "This must never be copied" not in proposal["abstract"]
    assert candidate["hypothesis"] in proposal["abstract"]
    assert proposal["rationale"] == "Model-authored rationale."


def test_proposal_fallback_references_and_bounds_remain_application_owned() -> None:
    candidate = _candidate()
    candidate["title"] = " ".join(["calibration"] * 80)
    candidate["supporting_paper_ids"] = ["doi:10.1/not-present"]

    proposal = build_conservative_proposal(
        topic="language model weight compression",
        final_candidate=candidate,
        novelty_verification={"status": "uncertain"},
        evidence_papers=[{"paper_id": PAPER_IDS[0], "title": "Fallback evidence"}],
        generation_mode="structured_exact",
    )

    assert proposal["references"] == [PAPER_IDS[0]]
    assert proposal["title"].endswith("...")
    assert len(proposal["title"]) <= 240
    with pytest.raises(ValueError, match="without evidence papers"):
        build_conservative_proposal(
            topic="language model weight compression",
            final_candidate=candidate,
            novelty_verification={"status": "uncertain"},
            evidence_papers=[],
        )


def test_default_v6_gates_precede_prior_art_and_proposal_steps() -> None:
    config = load_pipeline_config()
    positions = {step.name: index for index, step in enumerate(config.steps)}

    assert str(config.version) == "6.0"
    assert positions["Audit Candidate Pool"] < positions["Evidence Grounding Critique"]
    assert positions["Gate Revised Candidate"] < positions["Build Novelty Queries"]
    assert positions["Gate Revised Candidate"] < positions["Compose Research Proposal"]
    generation = config.steps[positions["Generate Candidate Kernels"]]
    revision = config.steps[positions["Revise Winning Candidate"]]
    assert generation.max_output_tokens == 640
    assert generation.validation_attempts == 2
    assert generation.item_failure == "degrade"
    assert generation.prompt_version == "1"
    assert revision.type == "tool_call"
    repair = config.steps[positions["Repair Revised Candidate"]]
    assert repair.prompt_version == "8"
    assert repair.prompt_view_budget == 6000
    assert repair.item_failure == "degrade"


def test_modular_pipeline_loader_validates_trusted_local_resources(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
version: '5.0'
context_init: [query]
steps:
  - name: Compact query
    type: tool_call
    function: agent_helpers.compact_search_query
    inputs: [query]
    outputs: [compact_query]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert str(config.version) == "5.0"
    assert config.steps[0].function == "agent_helpers.compact_search_query"
    with pytest.raises(ValueError, match="definition is missing"):
        load_config(tmp_path / "missing.yaml")


def test_modular_pipeline_loader_wraps_contract_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("steps: []", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid built in pipeline definition"):
        load_config(config_path)


def test_candidate_units_preserve_evidence_identity_through_kernel_protocol_and_critique():
    from agent.pipeline.stages.candidate_units import (
        audit_candidate_pool,
        compile_candidate_critiques,
        compile_candidate_units,
        prepare_all_candidate_critique_packets,
        prepare_candidate_kernel_packets,
        prepare_candidate_protocol_packets,
        select_candidate_v6,
    )

    scope = {"mode": "broad", "label": "Broad topic"}
    ledger = _evidence_ledger()
    papers = _papers()
    gap_map = {
        "unresolved_questions": [
            {"statement": "Compare calibration methods", "paper_ids": list(PAPER_IDS)},
            {"statement": "Unrelated gap", "paper_ids": ["unavailable"]},
            *[{"statement": "Open question", "paper_ids": []} for _ in range(4)],
        ]
    }
    kernels = prepare_candidate_kernel_packets(
        "language model weight compression",
        scope,
        {"options": []},
        ledger,
        gap_map,
        papers,
        max_evidence_items=1,
    )
    assert len(kernels) == 3
    assert kernels[-1]["unit_id"] == "boundary-condition-or-failure-mode"
    for packet in kernels:
        slots = {card["slot"] for card in packet["input"]["evidence_cards"]}
        assert len(slots) == 1
        assert len(packet["input"]["evidence_gaps"]) == 3
        assert all(set(gap["evidence_slots"]) <= slots for gap in packet["input"]["evidence_gaps"])

    spec = _scientific_spec()
    protocol_keys = {"experiment", "confounders", "risks", "potential_insights"}
    kernel = {key: value for key, value in spec.items() if key not in protocol_keys}
    protocol = {key: value for key, value in spec.items() if key in protocol_keys}
    # Generation input selects one source; only cite the evidence actually supplied.
    kernel["evidence_slots"] = [kernels[0]["input"]["evidence_cards"][0]["slot"]]

    def succeeded(value):
        return {"status": "succeeded", "value": value, "metadata": {"integrity": "normalized"}}

    packets = prepare_candidate_protocol_packets(kernels, [succeeded(kernel), {"status": "failed"}])
    assert len(packets) == 1
    candidates = compile_candidate_units(
        packets,
        [succeeded(protocol)],
        "language model weight compression",
        scope,
        {"options": []},
        ledger,
        papers,
    )
    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == kernels[0]["unit_id"]
    assert candidates[0]["generation_provenance"]["mode"] == "structured_normalized"
    assert set(candidates[0]["supporting_paper_ids"]) <= set(PAPER_IDS)
    assert not compile_candidate_units(
        packets,
        [{"status": "failed"}],
        "compression",
        scope,
        {"options": []},
        ledger,
        papers,
    )

    complete, audits, outcome = audit_candidate_pool(candidates, ledger, "model compression")
    assert len(complete) == 1
    assert outcome["status"] == "continue"
    critique_packets = prepare_all_candidate_critique_packets(complete, audits, ledger)
    assert critique_packets[0]["unit_id"] == candidates[0]["candidate_id"]
    assert len(critique_packets[0]["input"]["evidence_cards"]) == len(
        candidates[0]["supporting_paper_ids"]
    )
    draft = {
        "candidate_id": "invented",
        "grounding": 5,
        "falsifiability": 5,
        "feasibility": 5,
        "clarity": 5,
        "verdict": "pass",
        "findings": [],
    }
    critiques = compile_candidate_critiques(
        critique_packets, [succeeded(draft)], critic_source="evidence_critic"
    )
    assert critiques["critiques"][0]["candidate_id"] == candidates[0]["candidate_id"]
    assert critiques["critiques"][0]["score_source"] == "model"
    fallback = compile_candidate_critiques(critique_packets, [], critic_source="methodology_critic")
    assert fallback["critiques"][0]["score_source"] == "fallback"
    _, _, scorecard, selection = select_candidate_v6(complete, critiques, fallback, audits)
    assert scorecard[0]["critic_coverage"] == 1
    assert selection["status"] == "continue"
    _, _, scorecard, selection = select_candidate_v6(complete, fallback, fallback, audits)
    assert not any(row["eligible"] for row in scorecard)
    assert selection["status"] == "evidence_only"


def test_candidate_pool_excludes_projected_content_and_migration_preserves_exact_candidates():
    from agent.pipeline.migrations import migrate_legacy_context
    from agent.pipeline.stages.candidate_units import audit_candidate_pool

    candidate = _candidate()
    context = {
        "topic": "compression",
        "_active_step": "Select Candidate",
        "hypothesis_candidates": [candidate],
        "obsolete": "discard",
        "_pending_step_outputs": {"old": "data"},
    }
    original = copy.deepcopy(context)
    migrated, step = migrate_legacy_context(
        context, schema_version="5.0", config=load_pipeline_config()
    )
    assert context == original
    assert step == "Audit Candidate Pool"
    assert len(migrated["hypothesis_candidates"]) == 1
    assert "obsolete" not in migrated and "_pending_step_outputs" not in migrated
    candidate["generation_provenance"]["mode"] = "draft_projected"
    complete, _, outcome = audit_candidate_pool([candidate], _evidence_ledger(), "compression")
    assert complete == []
    assert outcome["status"] == "evidence_only"
    migrated, step = migrate_legacy_context(
        context, schema_version="5.0", config=load_pipeline_config()
    )
    assert step == "Prepare Candidate Kernels"
    assert "hypothesis_candidates" not in migrated
