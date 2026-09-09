import json

import pytest
from ollama import ResponseError

from agent.agent import (
    AbstractAgent,
    ModelOutputError,
    OllamaRuntimeError,
    _candidate_critique_validator,
    _identity_scoped_format_schema,
    _normalize_app_owned_metadata,
    _ollama_format_schema,
    _reconcile_evidence_card_duplicates,
    _slotted_relevance_format_schema,
)
from agent.agent_helpers import normalize_paper_references
from agent.schemas import ResearchProposal


def ledger(paper_id: str) -> str:
    return json.dumps(
        {
            "cards": [
                {
                    "paper_id": paper_id,
                    "findings": [{"claim": "Claim", "source_passage": "Passage"}],
                    "method": "Not reported",
                    "limitations": [],
                    "relevance": "Relevant",
                }
            ]
        }
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []
        self.calls = []

    def chat(self, **kwargs):
        self.messages.append(kwargs["messages"])
        self.calls.append(kwargs)
        content = next(self.responses)
        midpoint = len(content) // 2
        return iter(
            [
                {"message": {"content": content[:midpoint]}, "model": "test:model"},
                {
                    "message": {"content": content[midpoint:]},
                    "model": "test:model",
                    "done": True,
                    "eval_count": 12,
                },
            ]
        )


def abstract_agent(responses) -> AbstractAgent:
    agent = AbstractAgent.__new__(AbstractAgent)
    agent.model = "test:model"
    agent.host = "http://localhost:11434"
    agent.inactivity_timeout = 600
    agent.client = FakeClient(responses)
    return agent


def length_limited_agent(responses) -> AbstractAgent:
    agent = abstract_agent(responses)
    source = iter(responses)

    def chat(**kwargs):
        content = next(source)
        return iter(
            [
                {
                    "message": {"content": content},
                    "done": True,
                    "done_reason": "length",
                    "eval_count": kwargs["options"]["num_predict"],
                }
            ]
        )

    agent.client.chat = chat
    return agent


def proposal() -> str:
    return json.dumps(
        {
            "proposal_status": "unverified_research_direction",
            "title": "Test proposal",
            "hypothesis": "A testable hypothesis",
            "rationale": "Evidence-grounded rationale",
            "proposed_method": "Controlled experiment",
            "predictions": ["A measurable prediction"],
            "falsifier": "No measurable difference",
            "novel_insights": ["A cautious insight"],
            "limitations_and_risks": [],
            "claim_evidence": [
                {
                    "claim": "A proposed inference",
                    "paper_ids": [],
                    "relationship": "proposed_inference",
                    "confidence": "low",
                }
            ],
            "abstract": "A concise abstract.",
            "references": ["paper:1"],
        }
    )


def test_ollama_schema_removes_grammar_bounds_but_keeps_structure() -> None:
    schema = _ollama_format_schema(ResearchProposal.model_json_schema())
    encoded = json.dumps(schema)

    assert "maxLength" not in encoded
    assert "minLength" not in encoded
    assert "maxItems" not in encoded
    assert "minItems" not in encoded
    assert schema["required"]
    assert schema["properties"]["proposal_status"]["enum"]
    assert schema["properties"]["claim_evidence"]["items"]["$ref"]
    assert schema["$defs"]["ClaimEvidenceLink"]["additionalProperties"] is False


def test_focused_recovery_schema_constrains_the_requested_identity() -> None:
    critique = _identity_scoped_format_schema("candidate_critiques", "candidate:2")
    novelty = _identity_scoped_format_schema("novelty_verifications", "candidate:2")

    assert critique["$defs"]["CandidateScore"]["properties"]["candidate_id"] == {
        "enum": ["candidate:2"],
        "type": "string",
    }
    assert novelty["$defs"]["NoveltyVerification"]["properties"]["candidate_id"] == {
        "enum": ["candidate:2"],
        "type": "string",
    }


def test_compact_schema_is_prompted_and_sent_to_ollama() -> None:
    agent = abstract_agent([proposal()])

    agent.ask("Compose", response_schema="research_proposal")

    prompted_schema = agent.client.messages[0][-1]["content"]
    assert "maxLength" not in prompted_schema
    assert '"required"' in prompted_schema
    assert "maxLength" not in json.dumps(agent.client.calls[0]["format"])


def test_schema_echo_gets_bounded_instance_repair_and_contract_scoped_downgrade() -> None:
    schema_echo = json.dumps(ResearchProposal.model_json_schema())
    agent = abstract_agent([schema_echo, proposal(), ledger("paper:1")])

    value, metrics = agent.ask("Compose", response_schema="research_proposal")
    ledger_value, _ = agent.ask("Extract", response_schema="evidence_ledger")

    assert value["title"] == "Test proposal"
    assert metrics["grammar_fallback_count"] == 1
    assert metrics["format_mode"] == "json_fallback"
    assert '"$defs":' not in agent.client.messages[1][-1]["content"]
    assert "DATA INSTANCE" in agent.client.messages[1][-1]["content"]
    assert agent.client.calls[1]["format"] == "json"
    assert isinstance(agent.client.calls[2]["format"], dict)
    assert ledger_value["cards"][0]["paper_id"] == "paper:1"


def test_non_domain_hypothesis_metadata_is_repaired_without_changing_content() -> None:
    raw = {
        "candidate_id": "invented",
        "applicability": "cross_domain",
        "target_domain": "Data Compression and Signal Processing",
        "hypothesis": "Preserve this scientific claim.",
    }

    normalized, events = _normalize_app_owned_metadata(
        raw, "hypothesis_candidate", expected_candidate_id="mechanism"
    )

    assert normalized["candidate_id"] == "mechanism"
    assert normalized["target_domain"] is None
    assert normalized["hypothesis"] == raw["hypothesis"]
    assert {event["path"] for event in events} == {"candidate_id", "target_domain"}


def test_slotted_relevance_schema_requires_each_batch_slot() -> None:
    schema = _ollama_format_schema(_slotted_relevance_format_schema(["p01", "p02"]))
    assessments = schema["properties"]["assessments"]

    assert assessments["required"] == ["p01", "p02"]
    assert assessments["additionalProperties"] is False
    assert assessments["properties"]["p01"]["$ref"].endswith("PaperRelevanceJudgement")


@pytest.mark.parametrize("during_stream", [False, True])
def test_grammar_failure_retries_once_in_validated_json_mode(during_stream) -> None:
    agent = abstract_agent([proposal()])
    fallback_client = agent.client
    original_chat = fallback_client.chat
    failed = False

    def chat(**kwargs):
        nonlocal failed
        fallback_client.calls.append(kwargs)
        if not failed:
            failed = True
            error = ResponseError("Failed to initialize samplers: failed to parse grammar", 400)
            if during_stream:

                def rejected_stream():
                    raise error
                    yield

                return rejected_stream()
            raise error
        fallback_client.calls.pop()
        return original_chat(**kwargs)

    fallback_client.chat = chat
    value, metrics = agent.ask("Compose", response_schema="research_proposal")

    assert value["title"] == "Test proposal"
    assert fallback_client.calls[0]["format"] != "json"
    assert fallback_client.calls[1]["format"] == "json"
    assert metrics["request_count"] == 2
    assert metrics["format_mode"] == "json_fallback"
    assert metrics["grammar_fallback_count"] == 1


def test_unrelated_http_400_does_not_use_json_fallback() -> None:
    agent = abstract_agent([])
    agent.client.chat = lambda **kwargs: (_ for _ in ()).throw(ResponseError("invalid option", 400))

    with pytest.raises(OllamaRuntimeError, match="HTTP 400"):
        agent.ask("Compose", response_schema="research_proposal")


def test_json_fallback_regenerates_overlong_scientific_text() -> None:
    invalid = json.loads(proposal())
    invalid["abstract"] = "x" * 2401
    agent = abstract_agent([json.dumps(invalid), proposal()])
    original_chat = agent.client.chat
    failed = False

    def chat(**kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise ResponseError("Failed to initialize samplers: failed to parse grammar", 400)
        return original_chat(**kwargs)

    agent.client.chat = chat
    value, metrics = agent.ask("Compose", response_schema="research_proposal")

    assert value["abstract"] == "A concise abstract."
    assert metrics["request_count"] == 3
    assert metrics["recovery_mode"] == "regenerated"
    assert "schema_normalized_paths" not in metrics
    assert all(call["format"] == "json" for call in agent.client.calls)


def test_evidence_ledger_regenerates_reported_overlong_fields() -> None:
    value = json.loads(ledger("paper:1"))
    value["cards"] = [
        {
            **value["cards"][0],
            "paper_id": f"paper:{index}",
            "findings": [
                {
                    "claim": "A source grounded compression finding " * 10,
                    "source_passage": (
                        "The source reports a concrete compression result " * 10
                        if index == 3
                        else "Concise source passage"
                    ),
                }
            ],
        }
        for index in range(1, 4)
    ]
    agent = abstract_agent([json.dumps(value), ledger("paper:1")])

    result, metrics = agent.ask("Extract evidence", response_schema="evidence_ledger")

    assert result["cards"][0]["paper_id"] == "paper:1"
    assert result["cards"][0]["findings"][0]["source_passage"] == "Passage"
    assert result["cards"][0]["provenance"]["extraction_mode"] == "legacy_model"
    assert metrics["request_count"] == 2
    assert metrics["recovery_mode"] == "regenerated"
    assert "schema_normalizations" not in metrics


def test_evidence_recovery_regenerates_excess_findings() -> None:
    value = json.loads(ledger("paper:1"))
    value["cards"][0]["findings"] = [
        {"claim": f"Claim {index}", "source_passage": f"Passage {index}"} for index in range(1, 4)
    ]
    agent = abstract_agent([json.dumps(value), ledger("paper:1")])

    result, metrics = agent.ask("Recover evidence", response_schema="evidence_ledger")

    assert [item["claim"] for item in result["cards"][0]["findings"]] == [
        "Claim",
    ]
    assert metrics["request_count"] == 2
    assert metrics["recovery_mode"] == "regenerated"
    assert "schema_normalizations" not in metrics


def test_list_trimming_cannot_bypass_semantic_identity_coverage() -> None:
    value = {
        "options": [
            {
                "scope_id": f"scope_{index}",
                "label": f"Scope {index}",
                "description": f"Description {index}",
                "paper_ids": [f"paper:{index}:a", f"paper:{index}:b"],
            }
            for index in range(1, 6)
        ]
    }
    agent = abstract_agent([json.dumps(value), json.dumps(value)])

    def require_fifth_scope(response):
        if not any(option["scope_id"] == "scope_5" for option in response["options"]):
            raise ValueError("scope_5 is required")
        return response

    with pytest.raises(ModelOutputError, match="after 2 attempts"):
        agent.ask(
            "Identify scopes",
            response_schema="evidence_scope_options",
            response_validator=require_fifth_scope,
        )

    assert len(agent.client.calls) == 2


def test_identical_evidence_cards_are_deduplicated() -> None:
    card = json.loads(ledger("paper:1"))["cards"][0]

    result, event = _reconcile_evidence_card_duplicates({"cards": [card, dict(card)]}, exact=False)

    assert result["cards"] == [card]
    assert event["identical_paper_ids"] == ["paper:1"]
    assert event["recovery_paper_ids"] == []


def test_exact_evidence_recovery_rejects_conflicting_card() -> None:
    first = json.loads(ledger("paper:1"))["cards"][0]
    second = {**first, "relevance": "Different generated relevance"}

    with pytest.raises(ValueError, match="conflicting duplicate evidence cards"):
        _reconcile_evidence_card_duplicates({"cards": [first, second]}, exact=True)


def test_initial_ledger_recovers_every_group_when_all_conflict() -> None:
    first = json.loads(ledger("paper:1"))["cards"][0]
    second = {**first, "relevance": "Different generated relevance"}

    result, event = _reconcile_evidence_card_duplicates({"cards": [first, second]}, exact=False)

    assert result["cards"] == []
    assert event["recovery_paper_ids"] == ["paper:1"]


def validator(value):
    normalized, _, unknown = normalize_paper_references(
        value, [{"paper_id": "doi:10.1109/tsp.2006.881199"}]
    )
    if unknown:
        raise ValueError(
            f"unknown IDs: {sorted(unknown)}. Allowed paper IDs: doi:10.1109/tsp.2006.881199"
        )
    return normalized


def test_structured_response_retries_unknown_id_and_canonicalizes_correction() -> None:
    agent = abstract_agent([ledger("doi:10.9/invented"), ledger("10.1109/tsp.2006.881199")])

    value, metrics = agent.ask(
        "Extract evidence",
        response_schema="evidence_ledger",
        response_validator=validator,
    )

    assert value["cards"][0]["paper_id"] == "doi:10.1109/tsp.2006.881199"
    assert metrics["request_count"] == 2
    assert metrics["streamed_characters"] == len(ledger("doi:10.9/invented")) + len(
        ledger("10.1109/tsp.2006.881199")
    )
    assert metrics["response_characters"] == len(ledger("10.1109/tsp.2006.881199"))
    assert metrics["eval_count"] == 12
    assert len(agent.client.messages) == 2
    assert "Allowed paper IDs" in agent.client.messages[1][-1]["content"]
    assert agent.client.calls[-1]["stream"] is True
    assert agent.client.calls[-1]["options"]["num_predict"] == 2048


def test_structured_response_fails_after_one_semantic_retry() -> None:
    agent = abstract_agent([ledger("doi:10.9/invented"), ledger("doi:10.8/invented")])

    with pytest.raises(ModelOutputError, match="after 2 attempts"):
        agent.ask(
            "Extract evidence",
            response_schema="evidence_ledger",
            response_validator=validator,
        )

    assert len(agent.client.messages) == 2


def test_final_recovery_uses_last_schema_valid_response_after_bounded_repair() -> None:
    incomplete = json.dumps(
        {
            "assessments": {
                "p01": {
                    "score": 5,
                    "role": "direct_evidence",
                    "rationale": "Central to the topic",
                }
            }
        }
    )
    agent = abstract_agent([incomplete, incomplete])

    def require_two_slots(value):
        missing = {"p01", "p02"} - set(value["assessments"])
        if missing:
            raise ValueError(f"missing={sorted(missing)}")
        return value

    def recover(value, error):
        assert "p02" in str(error)
        value["assessments"]["p02"] = {
            "score": 1,
            "role": "irrelevant",
            "rationale": "Conservatively excluded",
        }
        return value

    value, metrics = agent.ask(
        "Judge two papers",
        response_schema="slotted_paper_relevance_assessments",
        response_validator=require_two_slots,
        format_schema_override=_slotted_relevance_format_schema(["p01", "p02"]),
        final_recovery=recover,
    )

    assert value["assessments"]["p02"]["role"] == "irrelevant"
    assert metrics["request_count"] == 2
    assert metrics["recovery_mode"] == "conservative_fallback"


def test_truncated_json_gets_clean_larger_deterministic_retry() -> None:
    agent = abstract_agent(
        [
            '{"cards":[{"paper_id":"doi:10.1109/tsp.2006.881199","findings":[',
            ledger("doi:10.1109/tsp.2006.881199"),
        ]
    )

    value, metrics = agent.ask(
        "Extract evidence",
        response_schema="evidence_ledger",
        response_validator=validator,
        max_output_tokens=512,
        repair_output_tokens=1024,
    )

    assert value["cards"][0]["paper_id"] == "doi:10.1109/tsp.2006.881199"
    assert agent.client.calls[0]["options"]["num_predict"] == 512
    assert agent.client.calls[1]["options"] == {"temperature": 0, "num_predict": 1024}
    assert "Previous response" not in agent.client.messages[1][-1]["content"]
    assert "Regenerate the complete object" in agent.client.messages[1][-1]["content"]
    assert metrics["recovery_mode"] == "regenerated"


def test_disabled_reasoning_is_sent_to_thinking_model() -> None:
    agent = abstract_agent([ledger("doi:10.1109/tsp.2006.881199")])

    agent.ask(
        "Extract evidence",
        response_schema="evidence_ledger",
        response_validator=validator,
        reasoning_mode="disabled",
    )

    assert agent.client.calls[-1]["think"] is False


@pytest.mark.parametrize("partial_content", ["", '{"cards":['])
def test_mandatory_reasoning_gets_separate_budget_and_bounded_retry(
    partial_content,
) -> None:
    agent = abstract_agent([])
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return iter(
                [
                    {
                        "message": {
                            "content": partial_content,
                            "thinking": "reasoning" * 100,
                        },
                        "done": True,
                        "done_reason": "length",
                        "eval_count": 2560,
                    }
                ]
            )
        content = ledger("doi:10.1109/tsp.2006.881199")
        return iter(
            [
                {
                    "message": {"content": content, "thinking": ""},
                    "done": True,
                    "done_reason": "stop",
                }
            ]
        )

    agent.client.chat = chat
    value, metrics = agent.ask(
        "Extract evidence",
        response_schema="evidence_ledger",
        response_validator=validator,
        max_output_tokens=512,
        reasoning_mode="low",
        generation_token_limit=8192,
    )

    assert value["cards"]
    assert calls[0]["think"] == "low"
    assert calls[0]["options"]["num_predict"] == 2560
    assert calls[1]["options"]["num_predict"] == 5120
    assert metrics["thinking_characters"] == len("reasoning" * 100)
    assert metrics["request_count"] == 2


def test_prompt_only_profile_omits_ollama_format_argument() -> None:
    agent = abstract_agent([ledger("doi:10.1109/tsp.2006.881199")])

    agent.ask(
        "Extract evidence",
        response_schema="evidence_ledger",
        response_validator=validator,
        structured_output_mode="prompt",
    )

    assert "format" not in agent.client.calls[-1]


def test_runtime_format_rejection_falls_back_through_json_to_prompt() -> None:
    agent = abstract_agent([ledger("doi:10.1109/tsp.2006.881199")])
    original_chat = agent.client.chat
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if "format" in kwargs:
            raise ResponseError("format is unsupported", 400)
        return original_chat(**kwargs)

    agent.client.chat = chat
    value, metrics = agent.ask(
        "Extract evidence",
        response_schema="evidence_ledger",
        response_validator=validator,
    )

    assert value["cards"]
    assert len(calls) == 3
    assert calls[1]["format"] == "json"
    assert "format" not in calls[2]
    assert metrics["format_mode"] == "prompt_fallback"


def test_reasoning_budget_exhaustion_has_specific_private_diagnostic() -> None:
    agent = abstract_agent([])

    def chat(**kwargs):
        return iter(
            [
                {
                    "message": {"content": "", "thinking": "private trace"},
                    "done": True,
                    "done_reason": "length",
                }
            ]
        )

    agent.client.chat = chat
    with pytest.raises(ModelOutputError) as raised:
        agent.ask(
            "Extract evidence",
            response_schema="evidence_ledger",
            max_output_tokens=512,
            reasoning_mode="low",
            generation_token_limit=4096,
        )

    assert raised.value.diagnostic["category"] == "reasoning_budget_exhausted"
    assert raised.value.diagnostic["thinking_characters"] > 0
    assert "private trace" not in json.dumps(raised.value.diagnostic)


def test_twice_truncated_gap_map_recovers_only_complete_items() -> None:
    truncated = (
        '{"established":[{"statement":"Complete finding","paper_ids":[],"status":"unknown"}],'
        '"contradictions":[],"unresolved_questions":[{"statement":"unfinished'
    )
    agent = length_limited_agent([truncated, truncated])

    value, metrics = agent.ask(
        "Map gaps",
        response_schema="gap_map",
        max_output_tokens=512,
        repair_output_tokens=1024,
        allow_partial_recovery=True,
    )

    assert [item["statement"] for item in value["established"]] == ["Complete finding"]
    assert value["unresolved_questions"] == []
    assert metrics["recovery_mode"] == "validated_partial"
    assert metrics["output_limit_reached"] is True


def test_generated_token_count_confirms_output_limit_when_done_reason_is_missing() -> None:
    truncated = (
        '{"established":[{"statement":"Complete finding","paper_ids":[],"status":"unknown"}],'
        '"contradictions":[],"unresolved_questions":[{"statement":"unfinished'
    )
    agent = abstract_agent([])

    def chat(**kwargs):
        return iter(
            [
                {
                    "message": {"content": truncated},
                    "done": True,
                    "eval_count": kwargs["options"]["num_predict"],
                }
            ]
        )

    agent.client.chat = chat
    value, metrics = agent.ask(
        "Map gaps",
        response_schema="gap_map",
        max_output_tokens=512,
        allow_partial_recovery=True,
        validation_attempts=1,
    )

    assert value["established"][0]["statement"] == "Complete finding"
    assert metrics["output_limit_reached"] is True
    assert metrics.get("done_reason") is None


def test_partial_recovery_is_opt_in() -> None:
    truncated = (
        '{"established":[{"statement":"Complete finding","paper_ids":[],"status":"unknown"}],'
        '"contradictions":[],"unresolved_questions":[{"statement":"unfinished'
    )
    agent = length_limited_agent([truncated, truncated])

    with pytest.raises(ModelOutputError, match="output token limit"):
        agent.ask("Map gaps", response_schema="gap_map")


def test_partial_gap_map_without_complete_items_fails_closed() -> None:
    truncated = '{"established":[{"statement":"unfinished'
    agent = length_limited_agent([truncated, truncated])

    with pytest.raises(ModelOutputError, match="output token limit"):
        agent.ask(
            "Map gaps",
            response_schema="gap_map",
            allow_partial_recovery=True,
        )


def test_initial_candidate_critique_allows_subset_but_rejects_bad_ids() -> None:
    validator = _candidate_critique_validator({"a", "b"}, exact=False)

    assert validator({"critiques": [{"candidate_id": "a"}]})["critiques"]
    with pytest.raises(ValueError, match="duplicate"):
        validator({"critiques": [{"candidate_id": "a"}, {"candidate_id": "a"}]})
    with pytest.raises(ValueError, match="extra=.*x"):
        validator({"critiques": [{"candidate_id": "x"}]})


def test_recovered_candidate_critique_requires_exact_requested_id() -> None:
    validator = _candidate_critique_validator({"b"}, exact=True)

    with pytest.raises(ValueError, match="missing=.*b"):
        validator({"critiques": [{"candidate_id": "a"}]})


def test_stream_progress_reports_attempt_and_accumulated_characters() -> None:
    agent = abstract_agent([ledger("doi:10.1109/tsp.2006.881199")])
    progress = []

    agent.ask(
        "Extract evidence",
        response_schema="evidence_ledger",
        response_validator=validator,
        max_output_tokens=4096,
        progress_callback=lambda attempt, characters: progress.append((attempt, characters)),
    )

    assert progress[-1] == (1, len(ledger("doi:10.1109/tsp.2006.881199")))
    assert agent.client.calls[-1]["options"]["num_predict"] == 4096


def test_stream_timeout_is_not_retried_and_has_actionable_message() -> None:
    class ReadTimeout(Exception):
        pass

    agent = abstract_agent([])
    agent.client.chat = lambda **kwargs: (_ for _ in ()).throw(ReadTimeout("stalled"))

    with pytest.raises(OllamaRuntimeError, match="600 seconds.*inactivity-timeout"):
        agent.ask("Extract evidence", response_schema="evidence_ledger")
