from __future__ import annotations

import inspect
from pathlib import Path

import agent.pipeline.executor as executor
from agent.agent import load_pipeline_config
from agent.pipeline import (
    PipelineState,
    execute_tool_step,
    load_resume_checkpoint,
    resolve_tool,
    save_results,
)


def test_pipeline_package_owns_configuration_prompts_and_stage_sections() -> None:
    root = Path(__file__).parents[1] / "agent" / "pipeline"

    assert (root / "default.yaml").is_file()
    assert (root / "prompts" / "identify_evidence_scopes.txt").is_file()
    assert (root / "prompts" / "map_evidence_gaps.txt").is_file()
    assert not (root.parent / "agents_config.yaml").exists()
    assert not list((root.parent / "prompts").glob("*.txt"))
    assert {path.stem for path in (root / "stages").glob("*.py") if path.stem != "__init__"} >= {
        "planning",
        "relevance",
        "scopes_and_gaps",
        "evidence",
        "hypotheses",
        "critiques",
        "novelty",
        "proposal",
    }


def test_generic_tool_executor_contains_no_scientific_schema_branches() -> None:
    source = inspect.getsource(executor)

    assert "response_schema" not in source
    assert "hypothesis_candidate" not in source
    assert "research_proposal" not in source


def test_pipeline_public_api_exposes_modular_runtime_boundaries() -> None:
    assert PipelineState
    assert callable(execute_tool_step)
    assert callable(resolve_tool)
    assert callable(save_results)
    assert callable(load_resume_checkpoint)


def test_default_pipeline_uses_deterministic_relevance_and_bounded_units() -> None:
    config = load_pipeline_config()
    by_name = {step.name: step for step in config.steps}

    assert by_name["Assess Literature Relevance"].function == "agent_helpers.assess_relevance"
    assert by_name["Assess Prior Art Relevance"].function == "agent_helpers.assess_relevance"
    assert "Judge Literature Relevance" not in by_name
    assert by_name["Generate Candidate Kernels"].item_failure == "degrade"
    assert by_name["Generate Candidate Protocols"].item_failure == "degrade"
    assert by_name["Revise Winning Candidate"].type == "tool_call"
    assert by_name["Compose Research Proposal"].generation_strategy == "direct_structured"
    for name in (
        "Evidence Grounding Critique",
        "Methodology Critique",
        "Verify Candidate Novelty",
    ):
        recovery = by_name[name]
        assert recovery.map_input
        assert recovery.item_failure == "degrade"
    assert all(
        getattr(step, "invalid_output", None) == "recover"
        for step in config.steps
        if step.type == "ollama_call"
    )


def test_v6_prompts_are_versioned_delimited_and_role_separated() -> None:
    config = load_pipeline_config()
    llm_steps = [step for step in config.steps if step.type == "ollama_call"]

    assert len(llm_steps) == 10
    assert all(step.policy and int(step.prompt_version) >= 1 for step in llm_steps)
    assert all("untrusted data" in (step.system_prompt or "") for step in llm_steps)
    assert all(
        "<input_data>" in step.prompt_template
        or "{input_view}" in step.prompt_template
        or "<repair_packet>" in step.prompt_template
        for step in llm_steps
    )
    hypothesis = next(step for step in llm_steps if step.name == "Generate Candidate Kernels")
    assert hypothesis.map_input == "candidate_kernel_packets"
    assert "application metadata" in hypothesis.prompt_template

    prompt_root = Path(__file__).parents[1] / "agent" / "pipeline" / "prompts"
    assert len(list(prompt_root.glob("*.txt"))) >= 17
    assert len(list((prompt_root / "system").glob("*.txt"))) == 4
