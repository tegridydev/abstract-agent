import json

import pytest

import agent.agent as agent_module
from agent.agent import (
    ModelOutputError,
    PipelineManager,
    build_parser,
    load_pipeline_config,
    load_resume_checkpoint,
    save_failure_diagnostic,
    save_results,
)
from agent.checkpoints import context_checksum
from agent.reporting import PipelineEvent
from agent.schemas import (
    RESPONSE_SCHEMAS,
    ModelAssignments,
    PipelineConfig,
    SlottedPaperRelevanceAssessments,
)


class FakeAgent:
    def __init__(self, model, host, timeout):
        self.model = model

    def ask(self, prompt, **kwargs):
        return "generated", {"model": self.model, "eval_count": 3}

    def close(self):
        return None


class GrammarFallbackAgent(FakeAgent):
    def ask(self, prompt, **kwargs):
        return "generated", {
            "model": self.model,
            "request_count": 2,
            "grammar_fallback_count": 1,
            "format_mode": "json_fallback",
        }


class ConservativeRecoveryAgent(FakeAgent):
    def ask(self, prompt, **kwargs):
        del prompt
        value = kwargs["final_recovery"](None, ValueError("invalid model output"))
        value = RESPONSE_SCHEMAS[kwargs["response_schema"]].model_validate(value).model_dump()
        validator = kwargs.get("response_validator")
        if validator:
            value = validator(value)
        return value, {"request_count": 2, "recovery_mode": "conservative_fallback"}


class RecordingReporter:
    def __init__(self):
        self.events: list[PipelineEvent] = []

    def emit(self, event):
        self.events.append(event)

    def close(self):
        return None


def pipeline_config(minimum=1):
    return PipelineConfig.model_validate(
        {
            "minimum_unique_papers": minimum,
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Literature Search",
                    "type": "tool_call",
                    "function": "example.search",
                    "inputs": ["topic"],
                    "outputs": ["top_papers"],
                },
                {
                    "name": "Generate result",
                    "type": "ollama_call",
                    "model": "model:1",
                    "inputs": ["topic"],
                    "outputs": ["result"],
                    "prompt_template": "Generate a result for {topic}",
                },
            ],
            "final_outputs": ["top_papers", "result"],
        }
    )


@pytest.mark.parametrize(
    ("schema", "initial_context", "output", "assertion"),
    [
        (
            "research_plan",
            {"topic": "llm compression"},
            "research_plan",
            lambda value: value["required_concepts"] == ["llm compression"],
        ),
        (
            "evidence_scope_options",
            {"topic": "llm compression", "top_papers": []},
            "evidence_scope_options",
            lambda value: value == {"options": []},
        ),
        (
            "gap_map",
            {"topic": "llm compression", "evidence_ledger": {"cards": []}},
            "gap_map",
            lambda value: value["unresolved_questions"][0]["status"] == "unknown",
        ),
        (
            "candidate_critiques",
            {
                "topic": "llm compression",
                "hypothesis_candidates": [
                    {"candidate_id": "mechanism"},
                    {"candidate_id": "transfer"},
                ],
            },
            "critiques",
            lambda value: (
                {item["candidate_id"] for item in value["critiques"]} == {"mechanism", "transfer"}
            ),
        ),
        (
            "novelty_verifications",
            {
                "topic": "llm compression",
                "revised_candidate": {"candidate_id": "mechanism"},
                "alternate_candidate": {"candidate_id": "transfer"},
            },
            "verifications",
            lambda value: all(item["status"] == "uncertain" for item in value["verifications"]),
        ),
    ],
)
def test_pipeline_contracts_have_conservative_model_independent_recovery(
    schema, initial_context, output, assertion
) -> None:
    inputs = list(initial_context)
    config = PipelineConfig.model_validate(
        {
            "context_init": inputs,
            "steps": [
                {
                    "name": f"Recover {schema}",
                    "type": "ollama_call",
                    "model": "tiny:model",
                    "inputs": inputs,
                    "outputs": [output],
                    "prompt_template": "Return a valid result for {topic}",
                    "response_schema": schema,
                    "max_output_tokens": 256,
                    "invalid_output": "recover",
                }
            ],
        }
    )
    manager = PipelineManager(
        config,
        agent_factory=ConservativeRecoveryAgent,
        knowledge_enabled=False,
    )

    result = manager.run_pipeline(initial_context)
    manager.close()

    assert assertion(result[output])


@pytest.mark.parametrize(
    ("refresh", "expected"),
    [
        ("balanced", {"cache_ttl_seconds": 7 * 86_400}),
        ("always", {"use_cache": False}),
        ("offline", {"offline": True}),
    ],
)
def test_pipeline_applies_knowledge_refresh_policy(
    monkeypatch, tmp_path, refresh, expected
) -> None:
    calls = []

    def aggregate_sources(topic, **kwargs):
        calls.append({"topic": topic, **kwargs})
        return [], 0, []

    monkeypatch.setattr(agent_module.multi_source, "aggregate_sources", aggregate_sources)
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Retrieve Literature Candidates",
                    "type": "tool_call",
                    "function": "multi_source.aggregate_sources",
                    "inputs": ["topic"],
                    "outputs": ["papers", "fetched", "statuses"],
                }
            ],
        }
    )
    manager = PipelineManager(
        config,
        knowledge_enabled=True,
        knowledge_dir=tmp_path / "knowledge",
        knowledge_refresh=refresh,
    )

    manager.run_pipeline({"topic": "compression"})
    manager.close()

    assert calls[0]["knowledge_store"] is not None
    for key, value in expected.items():
        assert calls[0][key] == value


def test_pipeline_completes_with_insufficient_evidence_outcome(monkeypatch) -> None:
    manager = PipelineManager(pipeline_config(minimum=2), agent_factory=FakeAgent)
    monkeypatch.setattr(agent_module, "_resolve_tool", lambda path: lambda **kwargs: [{}])

    result = manager.run_pipeline({"topic": "test"})

    assert result["_outcome"] == "insufficient_evidence"
    assert result["research_outcome"]["status"] == "insufficient_evidence"


def test_pipeline_override_records_warning(monkeypatch) -> None:
    manager = PipelineManager(
        pipeline_config(minimum=2),
        allow_insufficient_literature=True,
        agent_factory=FakeAgent,
    )
    monkeypatch.setattr(agent_module, "_resolve_tool", lambda path: lambda **kwargs: [{}])

    result = manager.run_pipeline({"topic": "test"})

    assert result["result"] == "generated"
    assert result["_warnings"]


def test_pipeline_reports_validated_json_grammar_fallback() -> None:
    reporter = RecordingReporter()
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Compose Research Proposal",
                    "type": "ollama_call",
                    "model": "model:1",
                    "inputs": ["topic"],
                    "outputs": ["result"],
                    "prompt_template": "Generate a result for {topic}",
                }
            ],
        }
    )
    manager = PipelineManager(config, agent_factory=GrammarFallbackAgent, reporter=reporter)

    result = manager.run_pipeline({"topic": "test"})

    assert "continued in JSON mode" in result["_warnings"][0]
    assert any(event.kind == "warning" for event in reporter.events)


def test_pipeline_can_resume_at_failed_step_without_repeating_prior_tools(monkeypatch) -> None:
    manager = PipelineManager(pipeline_config(), agent_factory=FakeAgent)
    monkeypatch.setattr(
        agent_module,
        "_resolve_tool",
        lambda path: (_ for _ in ()).throw(AssertionError("prior tool reran")),
    )

    result = manager.run_pipeline(
        {"topic": "test", "top_papers": [{"paper_id": "paper:1"}], "result": "stale"},
        start_step="Generate result",
    )

    assert result["result"] == "generated"
    assert result["_resumed_from_step"] == "Generate result"
    assert result["_resume_count"] == 1


def test_output_error_preserves_completed_variants_and_resume_runs_only_remainder() -> None:
    class VariantAgent:
        def __init__(self):
            self.calls: list[str] = []
            self.failed = False

        def ask(self, prompt, **_kwargs):
            axis = next(item for item in ("one", "two", "three") if item in prompt)
            self.calls.append(axis)
            if axis == "three" and not self.failed:
                self.failed = True
                raise ModelOutputError(
                    "output token limit reached",
                    diagnostic={
                        "attempts": 1,
                        "category": "truncated_json",
                        "output_limit_reached": True,
                    },
                )
            return {
                "topic_interpretation": f"plan {axis}",
                "assumptions": [],
                "required_concepts": [axis],
                "mechanism_query": f"{axis} mechanism",
                "approaches_query": f"{axis} approaches",
                "prior_art_query": f"{axis} prior art",
                "exclusions": [],
            }, {
                "request_count": 1,
                "response_characters": 100,
                "generation_characters": 100,
                "compilation_outcome": "exact",
                "output_route": "content_schema",
                "format_mode": "schema",
            }

        def close(self):
            return None

    config = PipelineConfig.model_validate(
        {
            "version": "5.0",
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Generate variants",
                    "type": "ollama_call",
                    "model": "tiny:test",
                    "inputs": ["topic"],
                    "outputs": ["plans"],
                    "prompt_template": "Plan {topic} on axis {axis}",
                    "response_schema": "research_plan",
                    "policy": "planning.research",
                    "invalid_output": "abort",
                    "variants": ["one", "two", "three"],
                    "variant_key": "axis",
                    "validation_attempts": 1,
                }
            ],
            "final_outputs": ["plans"],
        }
    )
    agent = VariantAgent()
    manager = PipelineManager(config, agent_factory=lambda *_args: agent)

    with pytest.raises(ModelOutputError, match="output token limit"):
        manager.run_pipeline({"topic": "compression"})

    pending = manager.last_context["_pending_step_outputs"]["Generate variants"]
    assert pending["completed_count"] == 2
    assert pending["failed_variant"] == "three"
    assert [item["topic_interpretation"] for item in pending["values"]] == [
        "plan one",
        "plan two",
    ]

    result = manager.run_pipeline(manager.last_context, start_step="Generate variants")

    assert agent.calls == ["one", "two", "three", "three"]
    assert [item["topic_interpretation"] for item in result["plans"]] == [
        "plan one",
        "plan two",
        "plan three",
    ]
    assert "Generate variants" not in result["_pending_step_outputs"]


def test_resume_checkpoint_is_validated_against_current_config(tmp_path) -> None:
    config = pipeline_config()
    diagnostic = tmp_path / "failed.json"
    diagnostic = save_failure_diagnostic(
        {"topic": "test", "top_papers": [], "_active_step": "Generate result"},
        config,
        "test",
        tmp_path,
        "failed",
    )

    topic, context, step = load_resume_checkpoint(diagnostic, config)

    assert (topic, step) == ("test", "Generate result")
    assert context["_active_step"] == "Generate result"


def test_resume_checkpoint_rejects_context_tampering(tmp_path) -> None:
    config = pipeline_config()
    diagnostic = save_failure_diagnostic(
        {"topic": "test", "top_papers": [], "_active_step": "Generate result"},
        config,
        "test",
        tmp_path,
        "failed",
    )
    payload = json.loads(diagnostic.read_text())
    payload["diagnostic_context"]["topic"] = "edited"
    diagnostic.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="checksum"):
        load_resume_checkpoint(diagnostic, config)


def test_v4_checkpoint_is_intentionally_not_resumable_under_v41(tmp_path) -> None:
    diagnostic = tmp_path / "legacy.json"
    diagnostic.write_text(json.dumps({"schema_version": "4.0"}))

    with pytest.raises(ValueError, match="older results remain available in history"):
        load_resume_checkpoint(diagnostic, pipeline_config())


def test_v47_checkpoint_can_resume_when_it_matches_the_v48_pipeline(tmp_path) -> None:
    config = load_pipeline_config()
    context = {"topic": "llm compression", "_active_step": "Plan Research", "_resume_count": 0}
    payload = {
        "schema_version": "4.7",
        "pipeline": {"name": config.pipeline_name, "version": "4.7"},
        "run_info": {"topic": "llm compression"},
        "resumable": True,
        "resume_step": "Plan Research",
        "context_checksum": context_checksum(context),
        "diagnostic_context": context,
    }
    diagnostic = tmp_path / "v47-failure.json"
    diagnostic.write_text(json.dumps(payload), encoding="utf-8")

    topic, restored, step = load_resume_checkpoint(diagnostic, config)

    assert (topic, step) == ("llm compression", "Plan Research")
    assert {key: restored[key] for key in context} == context
    assert restored["_migrated_from"]["schema_version"] == "4.7"


def test_pipeline_surfaces_provider_failures_as_warnings(monkeypatch) -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Retrieve Literature Candidates",
                    "type": "tool_call",
                    "function": "example.search",
                    "inputs": ["topic"],
                    "outputs": ["literature_candidates", "count", "source_statuses"],
                }
            ],
        }
    )
    reporter = RecordingReporter()
    manager = PipelineManager(config, agent_factory=FakeAgent, reporter=reporter)
    monkeypatch.setattr(
        agent_module,
        "_resolve_tool",
        lambda path: (
            lambda **kwargs: (
                [],
                0,
                [
                    {
                        "provider": "doaj",
                        "status": "error",
                        "error_category": "http_400",
                    }
                ],
            )
        ),
    )

    result = manager.run_pipeline({"topic": "test"})

    assert "doaj: 1/1 attempted queries failed" in result["_warnings"][0]
    assert any(event.kind == "warning" for event in reporter.events)


def test_llm_variants_are_independent_calls() -> None:
    config = PipelineConfig.model_validate(
        {
            "minimum_unique_papers": 1,
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Generate candidates",
                    "type": "ollama_call",
                    "model": "model:1",
                    "inputs": ["topic"],
                    "outputs": ["candidates"],
                    "variants": ["mechanism", "method"],
                    "variant_key": "axis",
                    "prompt_template": "Generate a result for {topic} using {axis}",
                }
            ],
            "final_outputs": ["candidates"],
        }
    )
    manager = PipelineManager(config, agent_factory=FakeAgent)

    result = manager.run_pipeline({"topic": "test"})

    assert result["candidates"] == ["generated", "generated"]
    assert len(result["_llm_metrics"]["Generate candidates"]) == 2


def test_mapped_relevance_batch_requires_exact_id_coverage() -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic", "literature_candidates", "batches"],
            "steps": [
                {
                    "name": "Judge batches",
                    "type": "ollama_call",
                    "model": "model:1",
                    "inputs": ["topic", "batches"],
                    "outputs": ["decisions"],
                    "map_input": "batches",
                    "map_item_key": "batch",
                    "response_schema": "paper_relevance_assessments",
                    "prompt_template": "Judge {batch} for the topic {topic}",
                }
            ],
        }
    )

    class IncompleteAgent(FakeAgent):
        def ask(self, prompt, **kwargs):
            value = {
                "assessments": [
                    {
                        "paper_id": "paper:1",
                        "score": 5,
                        "role": "direct_evidence",
                        "rationale": "Relevant",
                    }
                ]
            }
            return kwargs["response_validator"](value), {"request_count": 1}

    manager = PipelineManager(config, agent_factory=IncompleteAgent)
    with pytest.raises(ValueError, match="assess every paper exactly once"):
        manager.run_pipeline(
            {
                "topic": "test",
                "literature_candidates": [
                    {"paper_id": "paper:1"},
                    {"paper_id": "paper:2"},
                ],
                "batches": [{"paper_ids": ["paper:1", "paper:2"]}],
            }
        )


def test_slotted_relevance_batch_conservatively_recovers_missing_paper() -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic", "literature_candidates", "batches"],
            "steps": [
                {
                    "name": "Judge batches",
                    "type": "ollama_call",
                    "model": "model:1",
                    "inputs": ["topic", "batches"],
                    "outputs": ["decisions"],
                    "map_input": "batches",
                    "map_item_key": "batch",
                    "response_schema": "slotted_paper_relevance_assessments",
                    "invalid_output": "recover",
                    "prompt_template": "Judge {batch} for the topic {topic}",
                }
            ],
        }
    )

    class IncompleteSlottedAgent(FakeAgent):
        def ask(self, prompt, **kwargs):
            required = kwargs["format_schema_override"]["properties"]["assessments"]["required"]
            assert required == ["p01", "p02"]
            value = {
                "assessments": {
                    "p01": {
                        "score": 5,
                        "role": "direct_evidence",
                        "rationale": "Relevant",
                    }
                }
            }
            try:
                kwargs["response_validator"](value)
            except ValueError as error:
                recovered = kwargs["final_recovery"](value, error)
                recovered = kwargs["response_validator"](recovered)
                return recovered, {"request_count": 2, "recovery_mode": "conservative_fallback"}

    reporter = RecordingReporter()
    manager = PipelineManager(config, agent_factory=IncompleteSlottedAgent, reporter=reporter)
    result = manager.run_pipeline(
        {
            "topic": "test",
            "literature_candidates": [
                {"paper_id": "paper:1"},
                {"paper_id": "paper:2"},
            ],
            "batches": [
                {
                    "paper_ids": ["paper:1", "paper:2"],
                }
            ],
        }
    )

    assessments = result["decisions"][0]["assessments"]
    assert [item["paper_id"] for item in assessments] == ["paper:1", "paper:2"]
    assert assessments[1]["score"] == 1
    assert assessments[1]["role"] == "irrelevant"
    assert result["_normalization_events"][0]["policy"] == "per_slot_validation"
    assert any("conservatively excluded 1" in warning for warning in result["_warnings"])
    assert any(event.kind == "warning" for event in reporter.events)


def test_slotted_relevance_recovery_preserves_batch_with_one_long_rationale() -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic", "literature_candidates", "batches"],
            "steps": [
                {
                    "name": "Judge batches",
                    "type": "ollama_call",
                    "model": "model:1",
                    "inputs": ["topic", "batches"],
                    "outputs": ["decisions"],
                    "map_input": "batches",
                    "map_item_key": "batch",
                    "response_schema": "slotted_paper_relevance_assessments",
                    "invalid_output": "recover",
                    "prompt_template": "Judge {batch} for the topic {topic}",
                }
            ],
        }
    )

    class LongRationaleAgent(FakeAgent):
        def ask(self, prompt, **kwargs):
            value = {
                "assessments": {
                    "p01": {
                        "score": 5,
                        "role": "direct_evidence",
                        "rationale": "relevant " * 40,
                    },
                    "p02": {
                        "score": 4,
                        "role": "transferable_method",
                        "rationale": "Valid concise rationale",
                    },
                }
            }
            recovered = kwargs["final_recovery"](value, ValueError("rationale too long"))
            recovered = SlottedPaperRelevanceAssessments.model_validate(recovered).model_dump()
            recovered = kwargs["response_validator"](recovered)
            return recovered, {"request_count": 2, "recovery_mode": "conservative_fallback"}

    manager = PipelineManager(config, agent_factory=LongRationaleAgent)
    result = manager.run_pipeline(
        {
            "topic": "compression",
            "literature_candidates": [
                {"paper_id": "paper:1"},
                {"paper_id": "paper:2"},
            ],
            "batches": [{"paper_ids": ["paper:1", "paper:2"]}],
        }
    )

    assessments = result["decisions"][0]["assessments"]
    assert [assessment["score"] for assessment in assessments] == [5, 4]
    assert len(assessments[0]["rationale"]) <= 240
    assert result["_normalization_events"][0]["normalized_slots"] == ["p01"]


def test_pipeline_uses_interactive_evidence_scope_selection(monkeypatch) -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Validate Evidence Scopes",
                    "type": "tool_call",
                    "function": "example.scopes",
                    "inputs": ["topic"],
                    "outputs": ["evidence_scope_options", "research_scope"],
                }
            ],
        }
    )
    options = {
        "options": [
            {
                "scope_id": "scope_1",
                "label": "Scientific imaging",
                "description": "Imaging papers",
                "paper_ids": ["paper:1", "paper:2"],
            }
        ]
    }
    monkeypatch.setattr(
        agent_module,
        "_resolve_tool",
        lambda path: (
            lambda **kwargs: (
                options,
                {
                    "mode": "broad",
                    "label": "Broad topic",
                    "source": "default",
                    "supporting_paper_ids": [],
                },
            )
        ),
    )
    manager = PipelineManager(
        config,
        requested_scope="ask",
        scope_selector=lambda topic, choices: {
            "mode": "domain",
            "label": choices[0]["label"],
            "source": "interactive",
            "supporting_paper_ids": choices[0]["paper_ids"],
        },
    )

    result = manager.run_pipeline({"topic": "compression"})

    assert result["research_scope"]["mode"] == "domain"
    assert result["research_scope"]["label"] == "Scientific imaging"


def test_pipeline_can_resume_scope_validation_with_bound_input() -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["initial_evidence_scope_options", "top_papers"],
            "steps": [
                {
                    "name": "Validate Evidence Scopes",
                    "type": "tool_call",
                    "function": "agent_helpers.validate_scope_options",
                    "inputs": ["initial_evidence_scope_options", "top_papers"],
                    "input_bindings": {"evidence_scope_options": "initial_evidence_scope_options"},
                    "outputs": ["evidence_scope_options", "research_scope"],
                }
            ],
        }
    )
    manager = PipelineManager(config)
    context = {
        "initial_evidence_scope_options": {
            "options": [
                {
                    "scope_id": "scope_1",
                    "label": "Scientific imaging",
                    "description": "Compression techniques for scientific images",
                    "paper_ids": ["paper:1", "paper:2"],
                }
            ]
        },
        "top_papers": [
            {
                "paper_id": "paper:1",
                "relevance_judgement": {"role": "direct_evidence"},
            },
            {
                "paper_id": "paper:2",
                "relevance_judgement": {"role": "transferable_method"},
            },
        ],
    }

    result = manager.run_pipeline(context, start_step="Validate Evidence Scopes")

    assert result["evidence_scope_options"]["options"][0]["scope_id"] == "scope_1"
    assert result["research_scope"]["mode"] == "broad"
    assert result["_resume_count"] == 1


def test_pipeline_emits_presentation_neutral_progress(monkeypatch) -> None:
    reporter = RecordingReporter()
    manager = PipelineManager(pipeline_config(), agent_factory=FakeAgent, reporter=reporter)
    monkeypatch.setattr(
        agent_module,
        "_resolve_tool",
        lambda path: lambda **kwargs: [{"paper_id": "paper:1", "matched_queries": ["test"]}],
    )

    manager.run_pipeline({"topic": "test"})

    assert [event.kind for event in reporter.events] == [
        "pipeline_started",
        "step_started",
        "step_completed",
        "step_started",
        "model_request_started",
        "model_request_completed",
        "step_completed",
        "pipeline_completed",
    ]
    assert reporter.events[2].data["paper_count"] == 1


def test_pipeline_reports_schema_bounds_normalization() -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Generate bounded result",
                    "type": "ollama_call",
                    "model": "model:1",
                    "inputs": ["topic"],
                    "outputs": ["result"],
                    "prompt_template": "Generate a bounded result for {topic}",
                }
            ],
        }
    )
    reporter = RecordingReporter()

    class NormalizingAgent(FakeAgent):
        def ask(self, prompt, **kwargs):
            return "generated", {
                "request_count": 1,
                "schema_normalization_count": 2,
                "schema_normalized_paths": [
                    "cards[0].findings[0].claim",
                    "cards[1].findings",
                ],
                "schema_normalizations": [
                    {
                        "path": "cards[0].findings[0].claim",
                        "kind": "string",
                        "original_length": 300,
                        "limit": 240,
                    },
                    {
                        "path": "cards[1].findings",
                        "kind": "list",
                        "original_length": 3,
                        "limit": 2,
                    },
                ],
            }

    result = PipelineManager(
        config, agent_factory=NormalizingAgent, reporter=reporter
    ).run_pipeline({"topic": "compression"})

    assert result["_normalization_events"] == [
        {
            "step": "Generate bounded result",
            "kind": "schema_bounds",
            "fields": ["cards[0].findings[0].claim", "cards[1].findings"],
            "operations": [
                {
                    "path": "cards[0].findings[0].claim",
                    "kind": "string",
                    "original_length": 300,
                    "limit": 240,
                },
                {
                    "path": "cards[1].findings",
                    "kind": "list",
                    "original_length": 3,
                    "limit": 2,
                },
            ],
        }
    ]
    assert "shortened 2 overlong generated fields" in result["_warnings"][0]
    assert any(event.kind == "warning" for event in reporter.events)


def test_conflicting_initial_evidence_card_uses_focused_recovery() -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic", "top_papers"],
            "steps": [
                {
                    "name": "Extract Evidence Ledger",
                    "type": "ollama_call",
                    "model": "model:1",
                    "inputs": ["topic"],
                    "outputs": ["initial_evidence_ledger"],
                    "response_schema": "evidence_ledger",
                    "prompt_template": "Extract evidence for {topic}",
                },
                {
                    "name": "Find Missing Evidence Cards",
                    "type": "tool_call",
                    "function": "agent_helpers.prepare_missing_evidence_cards",
                    "inputs": ["top_papers", "initial_evidence_ledger"],
                    "outputs": ["evidence_recovery_packets"],
                },
            ],
        }
    )

    def card(paper_id, relevance):
        return {
            "paper_id": paper_id,
            "findings": [{"claim": "Claim", "source_passage": "Passage"}],
            "method": "Not reported",
            "limitations": [],
            "relevance": relevance,
        }

    class ConflictingLedgerAgent(FakeAgent):
        def ask(self, prompt, **kwargs):
            value = {
                "cards": [
                    card("paper:1", "First interpretation"),
                    card("paper:1", "Conflicting interpretation"),
                    card("paper:2", "Uncontested evidence"),
                ]
            }
            return kwargs["response_validator"](value), {"request_count": 1}

    top_papers = [
        {
            "paper_id": paper_id,
            "title": f"Paper {paper_id}",
            "summary": "Compression evidence",
            "source": "Test",
        }
        for paper_id in ("paper:1", "paper:2")
    ]
    result = PipelineManager(config, agent_factory=ConflictingLedgerAgent).run_pipeline(
        {"topic": "compression", "top_papers": top_papers}
    )

    assert [card["paper_id"] for card in result["initial_evidence_ledger"]["cards"]] == ["paper:2"]
    assert [packet["paper_id"] for packet in result["evidence_recovery_packets"]] == ["paper:1"]
    assert result["_normalization_events"] == [
        {
            "step": "Extract Evidence Ledger",
            "kind": "evidence_card_duplicates",
            "identical_paper_ids": [],
            "recovery_paper_ids": ["paper:1"],
            "conflicting_paper_ids": ["paper:1"],
        }
    ]
    assert "scheduled focused recovery for paper:1" in result["_warnings"][0]


def test_pipeline_records_reference_corrections_and_actual_model_requests(monkeypatch) -> None:
    config = PipelineConfig.model_validate(
        {
            "minimum_unique_papers": 1,
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Retrieve Literature",
                    "type": "tool_call",
                    "function": "example.search",
                    "inputs": ["topic"],
                    "outputs": ["top_papers"],
                },
                {
                    "name": "Extract Evidence",
                    "type": "ollama_call",
                    "model": "model:1",
                    "inputs": ["topic"],
                    "outputs": ["evidence_ledger"],
                    "response_schema": "evidence_ledger",
                    "prompt_template": "Extract evidence for {topic}",
                },
            ],
        }
    )
    reporter = RecordingReporter()

    class ReferenceAgent(FakeAgent):
        def ask(self, prompt, **kwargs):
            value = {
                "cards": [
                    {
                        "paper_id": "10.1109/TSP.2006.881199",
                        "findings": [{"claim": "Claim", "source_passage": "Passage"}],
                        "method": "Not reported",
                        "limitations": [],
                        "relevance": "Relevant",
                    }
                ]
            }
            return kwargs["response_validator"](value), {"request_count": 2}

    manager = PipelineManager(config, agent_factory=ReferenceAgent, reporter=reporter)
    monkeypatch.setattr(
        agent_module,
        "_resolve_tool",
        lambda path: (
            lambda **kwargs: [
                {
                    "paper_id": "doi:10.1109/tsp.2006.881199",
                    "matched_queries": ["test"],
                }
            ]
        ),
    )

    result = manager.run_pipeline({"topic": "test"})

    assert result["evidence_ledger"]["cards"][0]["paper_id"] == "doi:10.1109/tsp.2006.881199"
    assert result["_reference_corrections"] == [
        {
            "step": "Extract Evidence",
            "supplied_id": "10.1109/TSP.2006.881199",
            "canonical_id": "doi:10.1109/tsp.2006.881199",
        }
    ]
    completed = [event for event in reporter.events if event.kind == "step_completed"]
    assert completed[-1].data["model_calls"] == 2


def test_role_model_overrides_resolve_without_mutating_config() -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Plan topic",
                    "type": "ollama_call",
                    "model": "yaml:model",
                    "model_role": "planning_evidence",
                    "inputs": ["topic"],
                    "outputs": ["result"],
                    "prompt_template": "Generate a plan for {topic}",
                }
            ],
        }
    )
    manager = PipelineManager(
        config,
        agent_factory=FakeAgent,
        model_assignments=ModelAssignments(planning_evidence="selected:model"),
    )

    result = manager.run_pipeline({"topic": "test"})

    assert result["_resolved_models_by_step"] == {"Plan topic": "selected:model"}
    assert result["_llm_metrics"]["Plan topic"]["model"] == "selected:model"
    assert config.steps[0].model == "yaml:model"


@pytest.mark.parametrize("missing", [False, True])
def test_sparse_critique_pipeline_calls_model_only_for_missing_candidate(missing) -> None:
    candidates = [{"candidate_id": "a"}, {"candidate_id": "b"}]

    def critique(candidate_id):
        return {
            "candidate_id": candidate_id,
            "grounding": 3,
            "novelty": 3,
            "falsifiability": 3,
            "feasibility": 3,
            "clarity": 3,
            "issues": ["issue"],
            "required_revisions": ["revision"],
        }

    initial = {"critiques": [critique("a"), *([] if missing else [critique("b")])]}
    config = PipelineConfig.model_validate(
        {
            "context_init": ["hypothesis_candidates", "initial_critiques"],
            "steps": [
                {
                    "name": "Find missing critiques",
                    "type": "tool_call",
                    "function": "agent_helpers.prepare_missing_candidate_critiques",
                    "inputs": ["hypothesis_candidates", "initial_critiques"],
                    "outputs": ["packets"],
                },
                {
                    "name": "Recover missing critiques",
                    "type": "ollama_call",
                    "model": "test:model",
                    "inputs": ["packets"],
                    "outputs": ["recovered"],
                    "map_input": "packets",
                    "map_item_key": "packet",
                    "response_schema": "candidate_critiques",
                    "prompt_template": "Review exactly this candidate packet: {packet}",
                },
                {
                    "name": "Merge critiques",
                    "type": "tool_call",
                    "function": "agent_helpers.merge_candidate_critiques",
                    "inputs": ["hypothesis_candidates", "initial_critiques", "recovered"],
                    "input_bindings": {"recovered_batches": "recovered"},
                    "outputs": ["merged"],
                },
            ],
        }
    )
    calls = []

    class SparseAgent(FakeAgent):
        def ask(self, prompt, **kwargs):
            calls.append(prompt)
            value = {"critiques": [critique("b")]}
            validator = kwargs.get("response_validator")
            return validator(value) if validator else value, {"model": self.model}

    manager = PipelineManager(config, agent_factory=SparseAgent)
    result = manager.run_pipeline(
        {"hypothesis_candidates": candidates, "initial_critiques": initial}
    )

    assert len(calls) == int(missing)
    assert [item["candidate_id"] for item in result["merged"]["critiques"]] == ["a", "b"]


def test_all_model_override_precedes_yaml_but_role_override_precedes_all() -> None:
    assignments = ModelAssignments(all_model="all:model", creative="creative:model")

    assert assignments.resolve("planning_evidence", "yaml:model") == "all:model"
    assert assignments.resolve("creative", "yaml:model") == "creative:model"


def test_preflight_probes_each_unique_resolved_model_once() -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Plan topic",
                    "type": "ollama_call",
                    "model": "yaml:model",
                    "model_role": "planning_evidence",
                    "inputs": ["topic"],
                    "outputs": ["plan"],
                    "prompt_template": "Generate a plan for {topic}",
                },
                {
                    "name": "Write topic",
                    "type": "ollama_call",
                    "model": "yaml:model",
                    "model_role": "writing",
                    "inputs": ["plan"],
                    "outputs": ["result"],
                    "prompt_template": "Generate final output from {plan}",
                },
            ],
        }
    )
    calls = []

    class FakeModelService:
        def __init__(self, host, timeout):
            calls.append((host, timeout))

        def validate_models(self, names, *, probe):
            calls.append((names, probe))
            return {}

        def close(self):
            return None

    manager = PipelineManager(
        config,
        model_assignments=ModelAssignments(all_model="selected:model"),
        model_service_factory=FakeModelService,
    )

    manager.preflight()

    assert calls[-1] == ({"selected:model"}, True)


def test_cli_accepts_all_and_role_specific_model_overrides() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--topic",
            "test",
            "--model",
            "fallback:model",
            "--creative-model",
            "creative:model",
        ]
    )

    assert args.model == "fallback:model"
    assert args.creative_model == "creative:model"


def test_cli_accepts_named_inactivity_timeout_alias() -> None:
    args = build_parser().parse_args(
        ["run", "--topic", "test", "--ollama-inactivity-timeout", "600"]
    )

    assert args.ollama_timeout == 600


def test_cli_accepts_deep_search_profile() -> None:
    args = build_parser().parse_args(["run", "--topic", "test", "--search-depth", "deep"])

    assert args.search_depth == "deep"


def test_cli_accepts_explicit_research_scope() -> None:
    args = build_parser().parse_args(["run", "--topic", "test", "--scope", "Scientific imaging"])

    assert args.scope == "Scientific imaging"


def test_prompt_budget_rejects_unsafe_model_context_usage() -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Generate result",
                    "type": "ollama_call",
                    "model": "tiny:model",
                    "inputs": ["topic"],
                    "outputs": ["result"],
                    "prompt_template": "Generate a result for {topic}",
                    "max_output_tokens": 1024,
                }
            ],
        }
    )
    manager = PipelineManager(config, agent_factory=FakeAgent)
    manager._validated_model_metadata = {"tiny:model": {"context_length": 1024}}

    with pytest.raises(ModelOutputError, match="allocated context"):
        manager.run_pipeline({"topic": "test"})


def test_pipeline_passes_negotiated_model_profile_to_agent(monkeypatch) -> None:
    calls = []

    class ProfileAgent(FakeAgent):
        def ask(self, prompt, **kwargs):
            calls.append(kwargs)
            return super().ask(prompt, **kwargs)

    manager = PipelineManager(pipeline_config(), agent_factory=ProfileAgent)
    manager._validated_model_metadata = {
        "model:1": {
            "context_length": 8192,
            "reasoning_mode": "disabled",
            "structured_output_mode": "schema",
        }
    }
    monkeypatch.setattr(agent_module, "_resolve_tool", lambda path: lambda **kwargs: [{}])

    manager.run_pipeline({"topic": "test"})

    assert calls[0]["reasoning_mode"] == "disabled"
    assert calls[0]["structured_output_mode"] == "schema"
    assert calls[0]["generation_token_limit"] > 0


def test_results_are_versioned_and_written_atomically(tmp_path) -> None:
    config = pipeline_config()
    text_path, json_path = save_results(
        {
            "top_papers": [],
            "result": "done",
            "source_statuses": [],
            "_pipeline_model_assignments": {"all_model": "selected:model"},
            "_resolved_models_by_step": {"Generate result": "selected:model"},
        },
        config,
        "test topic",
        tmp_path,
    )

    payload = json.loads(json_path.read_text())
    assert text_path.exists()
    assert payload["schema_version"] == "6.0"
    assert payload["run_info"]["model_assignments"]["all_model"] == "selected:model"
    assert payload["run_info"]["resolved_models_by_step"] == {"Generate result": "selected:model"}
    assert not list(tmp_path.glob(".*"))


def test_insufficient_evidence_diagnostic_is_atomic_and_versioned(tmp_path) -> None:
    path = save_failure_diagnostic(
        {"literature_candidates": [{"paper_id": "paper:1"}], "_search_depth": "deep"},
        pipeline_config(),
        "test topic",
        tmp_path,
        "only one paper",
    )

    payload = json.loads(path.read_text())
    assert path.name.startswith("failed_test_topic_")
    assert payload["schema_version"] == "6.0"
    assert payload["status"] == "insufficient_evidence"
    assert payload["run_info"]["search_depth"] == "deep"
    assert payload["diagnostic_context"]["literature_candidates"]


def test_structured_failure_diagnostic_records_status(tmp_path) -> None:
    path = save_failure_diagnostic(
        {"_failure_diagnostic": {"step": "Map Evidence Gaps", "category": "truncated_json"}},
        pipeline_config(),
        "test topic",
        tmp_path,
        "response truncated",
        status="structured_output_or_ollama_error",
    )

    payload = json.loads(path.read_text())
    assert payload["status"] == "structured_output_or_ollama_error"
    assert payload["diagnostic_context"]["_failure_diagnostic"]["category"] == "truncated_json"


def test_v43_evidence_recovery_diagnostic_is_view_only_in_v46(tmp_path) -> None:
    config = load_pipeline_config()
    context = {
        "topic": "llm compression",
        "top_papers": [
            {
                "provider": "crossref",
                "provider_id": "10.1/example",
                "title": "Compression",
                "summary": "A source backed compression result.",
                "source": "crossref",
                "paper_id": "doi:10.1/example",
            }
        ],
        "research_scope": {
            "mode": "broad",
            "label": "Broad topic",
            "source": "default",
            "supporting_paper_ids": [],
        },
        "evidence_recovery_packets": [{"paper_id": "doi:10.1/example"}],
        "initial_evidence_ledger": {"cards": []},
        "_active_step": "Recover Missing Evidence Cards",
        "_resume_count": 0,
    }
    payload = {
        "schema_version": "4.3",
        "pipeline": {"name": config.pipeline_name, "version": "4.3"},
        "run_info": {"topic": "llm compression"},
        "resumable": True,
        "resume_step": "Recover Missing Evidence Cards",
        "context_checksum": context_checksum(context),
        "diagnostic_context": context,
    }
    path = tmp_path / "failed.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="only v4.7 through v6.0 diagnostics"):
        load_resume_checkpoint(path, config)
