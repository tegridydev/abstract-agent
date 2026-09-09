from __future__ import annotations

import json

import pytest

from agent.literature.relevance import assess_relevance
from agent.pipeline.config import load_config
from agent.pipeline.diagnostics import bounded_output_diagnostic, generation_summary
from agent.pipeline.executor import execute_tool_step, map_step_result
from agent.pipeline.persistence import load_resume_checkpoint, save_failure_diagnostic, save_results
from agent.schemas import PipelineConfig, ToolStep


def _config() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "version": "4.7",
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Prepare",
                    "type": "tool_call",
                    "function": "agent_helpers.build_search_query",
                    "inputs": ["topic"],
                    "outputs": ["plan"],
                }
            ],
            "final_outputs": ["plan"],
        }
    )


def test_diagnostics_bound_content_and_summarize_recovery_modes() -> None:
    diagnostic = bounded_output_diagnostic("abcdef", tail_chars=3)

    assert diagnostic["response_characters"] == 6
    assert len(diagnostic["response_sha256"]) == 64
    assert diagnostic["response_tail"] == "def"
    assert generation_summary(
        {
            "_generation_events": [
                {"mode": "structured_exact"},
                {"mode": "draft_projected"},
                {},
            ]
        }
    ) == {
        "modes": {"structured_exact": 2, "draft_projected": 1},
        "fallback_count": 1,
        "human_review_required": True,
    }


def test_generic_executor_maps_supported_result_shapes_and_rejects_drift() -> None:
    assert map_step_result(None, [], "tool") == {}
    assert map_step_result((1, 2), ["left", "right"], "tool") == {
        "left": 1,
        "right": 2,
    }
    assert map_step_result(1, ["value"], "tool") == {"value": 1}
    assert map_step_result({"left": 1, "right": 2}, ["left", "right"], "tool") == {
        "left": 1,
        "right": 2,
    }
    with pytest.raises(ValueError, match="returned 1 values"):
        map_step_result((1,), ["left", "right"], "tool")
    with pytest.raises(ValueError, match="declared outputs"):
        map_step_result({}, ["left", "right"], "tool")

    step = ToolStep.model_validate(
        {
            "name": "Add",
            "type": "tool_call",
            "function": "test.add",
            "inputs": ["source"],
            "input_bindings": {"value": "source"},
            "params": {"increment": 2},
            "outputs": ["result"],
        }
    )
    result = execute_tool_step(
        step,
        {"source": 3},
        resolver=lambda _name: lambda value, increment: value + increment,
        overrides={"increment": 4},
    )
    assert result == {"result": 7}


def test_modular_config_loader_reports_missing_and_invalid_resources(tmp_path) -> None:
    with pytest.raises(ValueError, match="definition is missing"):
        load_config(tmp_path / "missing.yaml")

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("steps: [", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid built in pipeline definition"):
        load_config(invalid)


def test_v46_persistence_round_trip_and_checksum_guard(tmp_path) -> None:
    config = _config()
    context = {"topic": "llm compression", "_active_step": "Prepare", "_resume_count": 0}
    diagnostic = save_failure_diagnostic(
        context,
        config,
        "llm compression",
        tmp_path,
        "transport interrupted",
        status="interrupted",
    )

    topic, restored, step = load_resume_checkpoint(diagnostic, config)
    assert (topic, step) == ("llm compression", "Prepare")
    assert restored["topic"] == "llm compression"

    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    payload["diagnostic_context"]["topic"] = "tampered"
    diagnostic.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_resume_checkpoint(diagnostic, config)

    text_path, json_path = save_results(
        {"topic": "llm compression", "plan": {"facets": ["compression"]}},
        config,
        "llm compression",
        tmp_path,
    )
    assert "Pipeline:" in text_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == "6.0"


def test_relevance_embedding_path_background_and_invalid_identity() -> None:
    candidates = [
        {
            "paper_id": "paper:survey",
            "title": "Survey of LLM Compression",
            "summary": "LLM compression quantization review.",
            "relevance_score": 0.9,
        }
    ]

    class Embeddings:
        def embed(self, texts):
            assert len(texts) == 2
            return [[1.0, 0.0], [1.0, 0.0]]

    accepted, decisions, rejected = assess_relevance(
        candidates,
        "llm compression",
        minimum_score=0.1,
        embedding_service=Embeddings(),
    )
    assert not rejected
    assert accepted[0]["relevance_judgement"]["role"] == "background"
    assert decisions[0]["alignment_method"] == "embedding"

    with pytest.raises(ValueError, match="canonical paper IDs"):
        assess_relevance([{"title": "LLM compression"}], "llm compression")
    with pytest.raises(ValueError, match="positive"):
        assess_relevance([], "llm compression", max_results=0)
