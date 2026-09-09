from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from ollama import ResponseError
from pydantic import BaseModel, Field, ValidationError

import agent.agent as agent_module
from agent.schemas import PipelineConfig


def test_truncated_json_detection_matrix() -> None:
    assert agent_module._is_truncated_json(ValueError("bad"), "{}", "length")
    decoder = json.JSONDecodeError("missing", '{"a":', 5)
    assert agent_module._is_truncated_json(decoder, '{"a":', None)
    assert agent_module._is_truncated_json(ValueError("unexpected EOF"), "{", None)
    assert not agent_module._is_truncated_json(ValueError("bad value"), "", None)


def test_discard_incomplete_gap_tail_only_changes_invalid_final_items() -> None:
    assert agent_module._discard_incomplete_gap_tail([1]) == [1]
    value = {
        "established": [
            {"statement": "Valid", "supporting_paper_ids": ["p1"]},
            {"incomplete": True},
        ],
        "contradictions": "not-a-list",
        "unresolved_questions": [],
    }
    result = agent_module._discard_incomplete_gap_tail(value)
    assert len(result["established"]) == 1
    assert result["contradictions"] == "not-a-list"


def test_ollama_format_schema_strips_nonstructural_keys_recursively() -> None:
    schema = {
        "title": "Root",
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 5, "description": "omit"},
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
    }
    cleaned = agent_module._ollama_format_schema(schema)
    assert "title" not in cleaned
    assert "maxLength" not in cleaned["properties"]["name"]
    assert cleaned["properties"]["items"]["items"]["anyOf"][1] == {"type": "null"}


def test_slotted_schema_requires_exact_application_slots() -> None:
    schema = agent_module._slotted_relevance_format_schema(["p01", "p02"])
    assessments = schema["properties"]["assessments"]
    assert assessments["required"] == ["p01", "p02"]
    assert set(assessments["properties"]) == {"p01", "p02"}


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [(400, "grammar failed", True), (400, "other bad request", False), (500, "grammar", False)],
)
def test_format_initialization_error_detection(status: int, message: str, expected: bool) -> None:
    assert agent_module._is_format_initialization_error(ResponseError(message, status)) is expected


def test_resolve_tool_accepts_builtins_and_rejects_unknown() -> None:
    assert callable(agent_module._resolve_tool("agent_helpers.build_search_query"))
    with pytest.raises(ValueError, match="unknown built in"):
        agent_module._resolve_tool("unknown.function")
    with pytest.raises(ValueError, match="unknown built in"):
        agent_module._resolve_tool("agent_helpers.not_present")


def contract_config(params: dict[str, Any] | None = None) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Tool",
                    "type": "tool_call",
                    "function": "agent_helpers.build_search_query",
                    "inputs": ["topic"],
                    "input_bindings": {"research_plan": "topic"},
                    "params": params or {},
                    "outputs": ["queries"],
                }
            ],
        }
    )


def test_tool_contract_validation_reports_missing_and_unexpected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def strict(research_plan: str, *, option: str = "x") -> None:
        del research_plan, option

    monkeypatch.setattr(agent_module, "_resolve_tool", lambda path: strict)
    valid = contract_config()
    agent_module._validate_tool_contracts(valid)
    with pytest.raises(ValueError, match="unexpected arguments"):
        agent_module._validate_tool_contracts(contract_config({"extra": 1}))

    def two_required(first: str, second: str) -> None:
        del first, second

    monkeypatch.setattr(agent_module, "_resolve_tool", lambda path: two_required)
    with pytest.raises(ValueError, match="missing required arguments"):
        agent_module._validate_tool_contracts(valid)

    def extras(research_plan: str, **kwargs: Any) -> None:
        del research_plan, kwargs

    monkeypatch.setattr(agent_module, "_resolve_tool", lambda path: extras)
    agent_module._validate_tool_contracts(contract_config({"extra": 1}))


def test_load_pipeline_config_wraps_file_yaml_schema_and_contract_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Missing:
        def read_text(self, **kwargs: Any) -> str:
            del kwargs
            raise FileNotFoundError

    monkeypatch.setattr(agent_module, "DEFAULT_CONFIG_PATH", Missing())
    with pytest.raises(ValueError, match="definition is missing"):
        agent_module.load_pipeline_config()

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("[unterminated")
    monkeypatch.setattr(agent_module, "DEFAULT_CONFIG_PATH", invalid_yaml)
    with pytest.raises(ValueError, match="invalid built in"):
        agent_module.load_pipeline_config()

    invalid_schema = tmp_path / "schema.yaml"
    invalid_schema.write_text(yaml.safe_dump({"steps": []}))
    monkeypatch.setattr(agent_module, "DEFAULT_CONFIG_PATH", invalid_schema)
    with pytest.raises(ValueError, match="invalid built in"):
        agent_module.load_pipeline_config()

    monkeypatch.setattr(agent_module, "DEFAULT_CONFIG_PATH", Path("agent/pipeline/default.yaml"))
    monkeypatch.setattr(
        agent_module,
        "_validate_tool_contracts",
        lambda config: (_ for _ in ()).throw(ValueError("contract drift")),
    )
    with pytest.raises(ValueError, match="contract drift"):
        agent_module.load_pipeline_config()


def test_response_value_attribute_mapping_and_default() -> None:
    class Object:
        value = 2

    assert agent_module._response_value(Object(), "value") == 2
    assert agent_module._response_value({"value": 3}, "value") == 3
    assert agent_module._response_value(None, "value", 4) == 4


def test_referenced_paper_ids_only_collects_schema_reference_fields() -> None:
    value = {
        "paper_id": "p1",
        "supporting_paper_ids": ["p2", 3],
        "references": ["p3"],
        "nested": [{"closest_paper_ids": ["p4"]}],
        "plain": "p5",
    }
    assert agent_module._referenced_paper_ids(value) == {"p1", "p2", "p3", "p4"}


def test_paper_reference_validator_rejects_unknown_and_undeclared_claim_ids() -> None:
    papers = [{"paper_id": "doi:10.1/example"}, {"paper_id": "p2"}]
    corrections: list[dict[str, str]] = []
    validate = agent_module._paper_reference_validator(papers, corrections)
    with pytest.raises(ValueError, match="unknown paper IDs"):
        validate({"paper_id": "unknown"})
    with pytest.raises(ValueError, match="included in references"):
        validate({"references": [], "claim_evidence": [{"paper_ids": ["p2"]}]})
    normalized = validate({"paper_id": "10.1/EXAMPLE"})
    assert normalized["paper_id"] == "doi:10.1/example"
    assert corrections


def test_candidate_critique_validator_identity_and_normalizer_paths() -> None:
    validator = agent_module._candidate_critique_validator({"a", "b"}, exact=True)
    with pytest.raises(ValueError, match="duplicate"):
        validator({"critiques": [{"candidate_id": "a"}, {"candidate_id": "a"}]})
    with pytest.raises(ValueError, match="do not match"):
        validator({"critiques": [{"candidate_id": "a"}, {"candidate_id": "x"}]})
    sparse = agent_module._candidate_critique_validator(
        {"a", "b"}, exact=False, normalize=lambda value: {"critiques": value}
    )
    assert sparse([{"candidate_id": "a"}])["critiques"][0]["candidate_id"] == "a"


def test_shorten_bounded_text_branches() -> None:
    assert agent_module._shorten_bounded_text("short", 10) == "short"
    assert agent_module._shorten_bounded_text("abcdef", 3) == "abc"
    assert agent_module._shorten_bounded_text("alpha beta gamma", 12) == "alpha..."
    assert agent_module._shorten_bounded_text("abcdefghijk", 8) == "abcde..."


class Bounded(BaseModel):
    title: str = Field(max_length=5)
    items: list[int] = Field(max_length=2)


def test_normalize_schema_bounds_repairs_only_supported_excesses() -> None:
    value = {"title": "a title too long", "items": [1, 2, 3]}
    with pytest.raises(ValidationError) as caught:
        Bounded.model_validate(value)
    normalized, events = agent_module._normalize_schema_bounds(value, caught.value)
    assert normalized == {"title": "a...", "items": [1, 2]}
    assert {event["kind"] for event in events} == {"string", "list"}


def test_app_owned_metadata_normalization_matrix() -> None:
    original = {"candidate_id": "wrong", "applicability": "broad", "target_domain": "x"}
    untouched, events = agent_module._normalize_app_owned_metadata(original, None)
    assert untouched is original and events == []
    normalized, events = agent_module._normalize_app_owned_metadata(
        original, "hypothesis_candidate", expected_candidate_id="expected"
    )
    assert normalized["target_domain"] is None
    assert normalized["candidate_id"] == "expected"
    assert len(events) == 2
