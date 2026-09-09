from __future__ import annotations

from rich.console import Console

from agent.pipeline.migrations import migrate_legacy_context
from agent.pipeline.model_io import build_capability_reports
from agent.pipeline.prompting import PromptFit, plan_prompt
from agent.pipeline.stages import (
    compile_candidate_critiques,
    compile_novelty_decisions,
    prepare_candidate_protocol_packets,
)
from agent.reporting import PipelineEvent, RichProgressReporter
from agent.schemas import PipelineConfig


def _succeeded(value: dict, *, integrity: str = "exact") -> dict:
    return {
        "status": "succeeded",
        "value": value,
        "metadata": {"integrity": integrity},
    }


def test_prompt_planner_accounts_for_every_component_and_allocated_context() -> None:
    plan = plan_prompt(
        components={"system": "s" * 100, "prompt": "p" * 500, "schema": "x" * 250},
        output_tokens=256,
        allocated_context_tokens=4096,
    )

    assert plan.fit is PromptFit.FIT
    assert set(plan.component_tokens) == {"system", "prompt", "schema"}
    assert plan.estimated_input_tokens == sum(plan.component_tokens.values())
    assert plan.context_tokens == 4096


def test_prompt_planner_returns_degraded_instead_of_throwing() -> None:
    plan = plan_prompt(
        components={"prompt": "large" * 10_000},
        output_tokens=1024,
        allocated_context_tokens=2048,
    )

    assert plan.fit is PromptFit.DEGRADED
    assert plan.can_execute is False
    assert plan.reason


def test_failed_kernel_unit_is_not_forwarded_to_protocol_generation() -> None:
    packet_input = {
        "topic": "compression",
        "innovation_axis": "mechanism",
        "evidence_cards": [],
        "evidence_gaps": [],
    }
    packets = [
        {
            "unit_id": "one",
            "configured_axis": "one",
            "axis_resolution": {},
            "input": packet_input,
        },
        {
            "unit_id": "two",
            "configured_axis": "two",
            "axis_resolution": {},
            "input": packet_input,
        },
    ]
    outcomes = [
        {
            "status": "degraded",
            "category": "model_output_invalid",
            "reason": "schema echo",
        },
        _succeeded({"title": "usable"}),
    ]

    prepared = prepare_candidate_protocol_packets(packets, outcomes)

    assert [item["unit_id"] for item in prepared] == ["two"]
    assert prepared[0]["kernel_integrity"] == "exact"


def test_critique_compiler_preserves_identity_and_falls_back_per_unit() -> None:
    packets = [{"unit_id": "candidate-one"}, {"unit_id": "candidate-two"}]
    valid = {
        "grounding": 5,
        "falsifiability": 4,
        "feasibility": 4,
        "clarity": 5,
        "verdict": "pass",
        "findings": [],
    }
    drafts = [
        _succeeded(valid),
        {"status": "degraded", "reason": "truncated JSON"},
    ]

    result = compile_candidate_critiques(
        packets,
        drafts,
        critic_source="evidence_critic",
    )

    assert [item["candidate_id"] for item in result["critiques"]] == [
        "candidate-one",
        "candidate-two",
    ]
    assert result["critiques"][0]["score_source"] == "model"
    assert result["critiques"][1]["score_source"] == "fallback"


def test_novelty_compiler_owns_identity_and_rejects_out_of_packet_citations() -> None:
    packets = [
        {
            "candidate_id": "candidate-one",
            "prior_art": [{"paper_id": "paper:known"}],
        }
    ]
    draft = {
        "status": "novelty_supported",
        "closest_prior_art_paper_ids": ["paper:invented"],
        "materially_matching_paper_ids": [],
        "reasoning": "The cited work differs materially.",
        "limitations": [],
    }

    result = compile_novelty_decisions(packets, [_succeeded(draft)])
    decision = result["verifications"][0]

    assert decision["candidate_id"] == "candidate-one"
    assert decision["status"] == "uncertain"
    assert decision["closest_prior_art_paper_ids"] == []


def test_capability_report_separates_probe_data_from_observed_contracts() -> None:
    report = build_capability_reports(
        {
            "tiny:model": {
                "capabilities": ["completion"],
                "context_length": 8192,
                "allocated_context_length": 4096,
                "structured_output_mode": "json",
            }
        },
        compilation_events=[
            {"model": "tiny:model", "policy_id": "hypotheses.kernel", "outcome": "exact"},
            {
                "model": "tiny:model",
                "policy_id": "hypotheses.kernel",
                "outcome": "fallback",
            },
        ],
        stage_outcomes=[
            {
                "status": "degraded",
                "metadata": {"model": "tiny:model"},
            }
        ],
    )["tiny:model"]

    assert report["transport"] == "available"
    assert report["allocated_context_length"] == 4096
    assert report["observed_policies"]["hypotheses.kernel"] == {
        "attempts": 2,
        "exact": 1,
        "normalized": 0,
        "projected": 0,
        "fallback": 1,
    }
    assert report["degraded_units"] == 1


def test_legacy_generative_checkpoint_migrates_to_first_safe_v6_unit() -> None:
    config = PipelineConfig.model_validate(
        {
            "version": "6.0",
            "context_init": ["topic", "research_scope", "evidence_ledger"],
            "steps": [
                {
                    "name": "Prepare Candidate Kernels",
                    "type": "tool_call",
                    "function": "pipeline.stages.prepare_candidate_kernel_packets",
                    "inputs": ["topic"],
                    "outputs": ["candidate_kernel_packets"],
                }
            ],
        }
    )
    source = {
        "topic": "compression",
        "research_scope": {"mode": "broad"},
        "evidence_ledger": {"cards": []},
        "legacy_private_value": "discard me",
        "_active_step": "Generate Independent Hypotheses",
    }

    migrated, step = migrate_legacy_context(source, schema_version="5.0", config=config)

    assert step == "Prepare Candidate Kernels"
    assert migrated["_active_step"] == step
    assert "legacy_private_value" not in migrated
    assert source["legacy_private_value"] == "discard me"


def test_progress_reporter_exposes_mapped_unit_position_without_truncating_details() -> None:
    console = Console(width=72, record=True, force_terminal=False)
    reporter = RichProgressReporter(console)
    reporter.emit(PipelineEvent("pipeline_started", total_steps=2))
    reporter.emit(
        PipelineEvent(
            "model_request_started",
            step_name="Generate Candidate Kernels",
            data={"unit_index": 2, "unit_total": 3, "attempt": 1},
        )
    )
    reporter.close()

    rendered = console.export_text()
    assert "unit 2/3" in rendered
    assert "model calls 1" in rendered
