import json

from pydantic import ValidationError

from agent.agent import AbstractAgent, _is_structural_output_error
from agent.llm.response_policies import (
    gap_generation_schema,
    normalize_gap_response,
    normalize_scope_response,
    scope_generation_schema,
)
from agent.schemas import EvidenceScopeOptions


def test_scope_policy_recovers_reported_small_model_failure() -> None:
    value = {
        "options": [
            {
                "scope_id": "scope_9",
                "label": "One paper scope",
                "description": "This option cannot meet the evidence minimum.",
                "paper_ids": ["paper:1"],
            },
            {
                "scope_id": "invented",
                "label": "A very long scope label " * 8,
                "description": "A useful cross-paper evidence cluster.",
                "paper_ids": ["paper:1", "unknown", "paper:2", "paper:2"],
            },
        ]
    }

    normalized, events = normalize_scope_response(value, {"paper:1", "paper:2"})

    validated = EvidenceScopeOptions.model_validate(normalized)
    assert len(validated.options) == 1
    assert validated.options[0].scope_id == "scope_1"
    assert validated.options[0].paper_ids == ["paper:1", "paper:2"]
    assert len(validated.options[0].label) <= 80
    assert {event["operation"] for event in events} >= {"drop", "shorten", "filter"}


def test_scope_policy_isolates_malformed_duplicates_and_excess_options() -> None:
    valid = [
        {
            "label": f"Scope {index}",
            "description": "A cross-paper evidence cluster " * (10 if index == 1 else 1),
            "paper_ids": [f"paper:{index * 2 - 1}", f"paper:{index * 2}"],
        }
        for index in range(1, 6)
    ]
    value = {
        "options": [
            "not-an-object",
            {"label": "", "description": "x", "paper_ids": []},
            {"label": "Bad IDs", "description": "Valid description", "paper_ids": "paper:1"},
            valid[0],
            {**valid[0], "paper_ids": ["paper:1", "paper:2"]},
            *valid[1:],
        ]
    }

    normalized, events = normalize_scope_response(
        value, {f"paper:{index}" for index in range(1, 11)}
    )

    assert [option["scope_id"] for option in normalized["options"]] == [
        "scope_1",
        "scope_2",
        "scope_3",
        "scope_4",
    ]
    assert len(normalized["options"][0]["description"]) <= 240
    reasons = {event["reason"] for event in events}
    assert "scope option is not an object" in reasons
    assert "scope display text is incomplete" in reasons
    assert "scope label is duplicated" in reasons
    assert any(event["operation"] == "limit" for event in events)

    assert normalize_scope_response({"options": "invalid"}, set()) == ({"options": []}, [])


def test_gap_policy_derives_status_from_collection_and_filters_unknown_ids() -> None:
    value = {
        "established": [
            {
                "statement": "A supported result",
                "paper_ids": ["paper:1", "invented"],
                "status": "unknown",
            }
        ],
        "contradictions": [],
        "unresolved_questions": [
            {
                "statement": "Why does performance vary?",
                "paper_ids": [],
                "status": "contradictory",
            }
        ],
    }

    normalized, events = normalize_gap_response(value, {"paper:1"}, topic="compression")

    assert normalized["established"][0]["status"] == "established"
    assert normalized["established"][0]["paper_ids"] == ["paper:1"]
    assert normalized["unresolved_questions"][0]["status"] == "unknown"
    assert sum(event["operation"] == "derive" for event in events) == 2


def test_gap_policy_uses_explicit_unknown_fallback_when_nothing_is_safe() -> None:
    normalized, events = normalize_gap_response(
        {
            "established": [
                {"statement": "Unsupported", "paper_ids": ["invented"], "status": "established"}
            ]
        },
        {"paper:1"},
        topic="LLM compression",
    )

    fallback = normalized["unresolved_questions"][0]
    assert fallback["status"] == "unknown"
    assert fallback["paper_ids"] == []
    assert "LLM compression" in fallback["statement"]
    assert any(event["operation"] == "fallback" for event in events)


def test_gap_policy_isolates_malformed_items_and_limits_collections() -> None:
    unresolved = [
        {
            "statement": f"Unresolved question {index}",
            "paper_ids": "not-a-list" if index == 0 else [],
            "status": "unknown",
        }
        for index in range(6)
    ]
    normalized, events = normalize_gap_response(
        {
            "established": [
                "invalid",
                {"statement": "", "paper_ids": ["paper:1"]},
                {"statement": "No citation", "paper_ids": "paper:1"},
            ],
            "contradictions": "not-a-list",
            "unresolved_questions": unresolved,
        },
        {"paper:1"},
        topic="compression",
    )

    assert len(normalized["unresolved_questions"]) == 5
    assert normalized["unresolved_questions"][0]["paper_ids"] == []
    operations = {event["operation"] for event in events}
    assert {"drop", "replace", "limit"} <= operations

    fallback, _ = normalize_gap_response(None, set(), topic="word " * 100)
    assert len(fallback["unresolved_questions"][0]["statement"]) <= 240


def test_generation_contracts_exclude_application_owned_fields() -> None:
    scope_schema = json.dumps(scope_generation_schema())
    gap_schema = json.dumps(gap_generation_schema())

    assert "scope_id" not in scope_schema
    assert "status" not in gap_schema
    assert "paper_ids" in scope_schema and "paper_ids" in gap_schema


def test_structural_error_classifier_does_not_confuse_bounds_with_format_failure() -> None:
    try:
        EvidenceScopeOptions.model_validate(
            {
                "options": [
                    {
                        "scope_id": "scope_1",
                        "label": "Valid label",
                        "description": "Valid description",
                        "paper_ids": ["only-one"],
                    }
                ]
            }
        )
    except ValidationError as error:
        assert not _is_structural_output_error(error)
    else:  # pragma: no cover - contract mutation guard
        raise AssertionError("expected scope validation to fail")

    try:
        EvidenceScopeOptions.model_validate({})
    except ValidationError as error:
        assert _is_structural_output_error(error)


class _Client:
    def __init__(self) -> None:
        self.calls = []
        self.responses = iter(["not json", '{"options":[]}'])

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.responses)
        return iter([{"message": {"content": content}, "done": True}])


def test_structurally_ignored_schema_downgrades_model_for_the_run() -> None:
    agent = AbstractAgent.__new__(AbstractAgent)
    agent.model = "test:model"
    agent.host = "http://localhost:11434"
    agent.inactivity_timeout = 600
    agent.client = _Client()

    value, metrics = agent.ask("Return options", response_schema="evidence_scope_options")

    assert value == {"options": []}
    assert agent.client.calls[0]["format"] != "json"
    assert agent.client.calls[1]["format"] == "json"
    assert metrics["grammar_fallback_count"] == 1
    assert agent._runtime_format_mode_override == "json"
