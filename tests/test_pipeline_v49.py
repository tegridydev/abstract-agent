from __future__ import annotations

import html
import json

import pytest

from agent import agent_helpers, multi_source
from agent.agent import PipelineManager
from agent.contracts.pipeline import PipelineConfig
from agent.contracts.sources import SourceStatus
from agent.pipeline.output.views import build_prompt_view
from agent.pipeline.stages.critiques import normalize_critique_item
from agent.pipeline.stages.hypotheses import project_hypothesis_candidate
from agent.pipeline.stages.proposal import build_conservative_proposal


def _paper(paper_id: str, candidate_id: str | None = None) -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Evidence {paper_id}",
        "summary": "Calibration-aware compression changes measured robustness under controls.",
        "source": "semantic_scholar",
        "sources": ["semantic_scholar"],
        "matched_candidate_ids": [candidate_id] if candidate_id else [],
        "relevance_judgement": {"role": "direct_evidence"},
    }


def _card(paper_id: str) -> dict:
    return {
        "paper_id": paper_id,
        "findings": [
            {
                "claim": "Calibration changes robustness.",
                "source_passage": "Calibration changes robustness under a controlled comparison.",
            }
        ],
        "method": "Controlled comparison",
        "limitations": ["One benchmark"],
        "relevance": "Compression robustness evidence",
    }


def _candidate(candidate_id: str, alignment: float = 0.3) -> dict:
    return {
        "candidate_id": candidate_id,
        "title": candidate_id,
        "supporting_paper_ids": ["paper:1", "paper:2"],
        "grounding_links": [{"alignment_score": alignment}],
        "generation_provenance": {"mode": "structured_exact"},
    }


def _critique(candidate_id: str, *, fatal: bool = False) -> dict:
    finding = {
        "category": "internal_consistency",
        "severity": "fatal",
        "issue": "The predicted direction contradicts the mechanism.",
        "required_revision": "Align the mechanism and predicted direction.",
    }
    return {
        "candidate_id": candidate_id,
        "grounding": 4,
        "novelty": 4,
        "falsifiability": 4,
        "feasibility": 4,
        "clarity": 4,
        "issues": [finding["issue"]] if fatal else ["No material issue."],
        "required_revisions": [finding["required_revision"]] if fatal else ["None."],
        "findings": [finding] if fatal else [],
        "fatal_flags": ["internal_inconsistency"] if fatal else [],
        "score_source": "model",
    }


def test_prompt_view_is_bounded_complete_json_with_explicit_omissions() -> None:
    context = {
        "topic": "llm compression",
        "evidence_ledger": {"cards": [_card(f"doi:10.1/{index}") for index in range(20)]},
        "gap_map": {
            "unresolved_questions": [
                {"statement": "x" * 500, "paper_ids": [f"doi:10.1/{index}"]} for index in range(10)
            ]
        },
    }

    view = build_prompt_view("hypothesis_v5", list(context), context, budget=3_000)
    raw = view.text.removeprefix("<input_view>").removesuffix("</input_view>")
    document = json.loads(html.unescape(raw))

    assert view.character_count <= 3_000
    assert document["omitted_items"] > 0
    assert document["content_hash"] == view.content_hash
    assert "[truncated]" not in view.text
    assert set(document["evidence_slot_map"]) <= {f"E{index}" for index in range(1, 21)}


def test_compact_hypothesis_spec_renders_canonical_relation_and_evidence_slots() -> None:
    papers = [_paper("paper:1"), _paper("paper:2")]
    result = project_hypothesis_candidate(
        {
            "title": "Calibration-aware compression " * 20,
            "study_design": "experimental",
            "system": "compressed language models",
            "exposure_or_intervention": "calibration-aware quantization",
            "comparator": "ordinary quantization",
            "primary_outcome": "held-out robustness",
            "expected_relation": "increases",
            "mechanism": "Calibration protects activation outliers.",
            "novelty_delta": "Directly tests robustness rather than aggregate accuracy.",
            "experiment": "Randomize calibration and evaluate fixed held-out tasks.",
            "evidence_slots": ["E2", "E1"],
            "confounders": ["dataset shift"],
            "risks": ["benchmark sensitivity"],
            "secondary_predictions": [
                {"outcome": "calibration error", "expected_relation": "decreases"}
            ],
            "potential_insights": ["Outlier protection may explain robustness."],
        },
        topic="llm compression",
        candidate_id="mechanism",
        research_scope={"mode": "broad", "label": "Broad topic"},
        evidence_scope_options=[],
        evidence_ledger={"cards": [_card("paper:1"), _card("paper:2")]},
        evidence_papers=papers,
        generation_mode="structured_exact",
    )

    assert "will be higher with calibration-aware quantization" in result["hypothesis"]
    assert result["supporting_paper_ids"] == ["paper:2", "paper:1"]
    assert "preregistered contrast" in result["falsifier"]
    assert "calibration error will be lower" in result["predictions"][1]
    assert len(result["title"]) <= 240


def test_prompt_view_and_critique_reject_unknown_contract_inputs() -> None:
    with pytest.raises(ValueError, match="unknown prompt view"):
        build_prompt_view("unknown", ["topic"], {"topic": "x"})
    with pytest.raises(ValueError, match="at least 1024"):
        build_prompt_view("compact_v1", ["topic"], {"topic": "x"}, budget=100)
    assert normalize_critique_item("invalid", expected_candidate_ids={"candidate"})[0] is None
    normalized, operations = normalize_critique_item(
        {"candidate_id": "unknown"}, expected_candidate_ids={"candidate"}
    )
    assert normalized is None
    assert "unknown critique" in operations[0]["reason"]


def test_fatal_candidate_is_excluded_before_selection() -> None:
    candidates = [_candidate("inconsistent"), _candidate("eligible")]
    reviews = {"critiques": [_critique("inconsistent", fatal=True), _critique("eligible")]}

    selected, _, scorecard = agent_helpers.select_candidate(candidates, reviews, reviews)

    assert selected["candidate_id"] == "eligible"
    rejected = next(item for item in scorecard if item["candidate_id"] == "inconsistent")
    assert rejected["eligible"] is False
    assert rejected["replacement_required"] is True
    assert rejected["fatal_flags"] == ["internal_inconsistency"]


def test_novelty_packet_never_cross_contaminates_candidate_prior_art() -> None:
    packet = agent_helpers.prepare_novelty_review_packet(
        _candidate("winner"),
        _candidate("alternate"),
        [_paper("paper:w", "winner"), _paper("paper:a", "alternate")],
    )

    winner, alternate = packet["candidate_packets"]
    assert winner["citation_slot_map"] == {"P1": "paper:w"}
    assert alternate["citation_slot_map"] == {"P1": "paper:a"}
    assert "paper:a" not in json.dumps(winner)
    assert "paper:w" not in json.dumps(alternate)


def test_novelty_prompt_view_uses_only_candidate_matched_prior_art() -> None:
    context = {
        "revised_candidate": _candidate("winner"),
        "alternate_candidate": _candidate("alternate"),
        "expanded_evidence": "must not enter the novelty view",
        "novelty_papers": [
            _paper("paper:w", "winner"),
            _paper("paper:a", "alternate"),
            _paper("paper:discovery"),
        ],
    }
    view = build_prompt_view(
        "novelty_v5",
        ["revised_candidate", "alternate_candidate", "expanded_evidence"],
        context,
    )
    raw = view.text.removeprefix("<input_view>").removesuffix("</input_view>")
    document = json.loads(html.unescape(raw))
    winner, alternate = document["data"]["candidate_packets"]

    assert winner["citation_slot_map"] == {"P1": "paper:w"}
    assert alternate["citation_slot_map"] == {"P1": "paper:a"}
    assert {"paper:w", "paper:a"} <= set(document["included_ids"])
    assert "paper:discovery" not in view.text
    assert "must not enter" not in view.text


def test_structured_critique_findings_drive_legacy_fields_and_fatal_flags() -> None:
    raw = _critique("candidate", fatal=True)
    raw["issues"] = []
    raw["required_revisions"] = []

    normalized, operations = normalize_critique_item(raw, expected_candidate_ids={"candidate"})

    assert normalized is not None
    assert normalized["score_source"] == "model"
    assert normalized["issues"] == [raw["findings"][0]["issue"]]
    assert normalized["required_revisions"] == [raw["findings"][0]["required_revision"]]
    assert normalized["fatal_flags"] == ["internal_inconsistency"]
    assert operations == []


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"planned_query_count": 0}, "attempted_query_count"),
        ({"succeeded_query_count": 2}, "succeeded_query_count"),
        ({"skipped_query_count": 2}, "skipped_query_count"),
    ],
)
def test_source_status_rejects_inconsistent_query_accounting(update, message) -> None:
    with pytest.raises(ValueError, match=message):
        SourceStatus(
            provider="provider",
            status="ok",
            result_count=1,
            duration_seconds=0,
            **update,
        )


def test_pipeline_records_prompt_view_and_all_stage_elapsed_metrics() -> None:
    captured: list[str] = []

    class Agent:
        def __init__(self, model, host, timeout):
            self.model = model

        def ask(self, prompt, **_kwargs):
            captured.append(prompt)
            return "result", {
                "model": self.model,
                "request_count": 1,
                "generation_characters": 6,
                "eval_count": 2,
            }

        def close(self):
            return None

    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Compact model stage",
                    "type": "ollama_call",
                    "model": "model:test",
                    "inputs": ["topic"],
                    "outputs": ["result"],
                    "prompt_template": "Review this bounded packet: {input_view}",
                    "prompt_view": "compact_v1",
                }
            ],
        }
    )

    context = PipelineManager(config, agent_factory=Agent).run_pipeline({"topic": "compression"})

    assert "<input_view>" in captured[0]
    metric = context["_llm_metrics"]["Compact model stage"]
    assert metric["prompt_view"] == "compact_v1"
    assert metric["prompt_view_characters"] <= 6_000
    assert context["_step_metrics"][0]["generation_characters"] == 6
    assert context["_step_metrics"][0]["output_tokens"] == 2


def test_revision_audit_downgrades_unchanged_candidate_with_major_findings() -> None:
    papers = [_paper("paper:1"), _paper("paper:2")]
    ledger = {"cards": [_card("paper:1"), _card("paper:2")]}
    spec = {
        "title": "Calibration-aware compression",
        "study_design": "experimental",
        "system": "compressed language models",
        "exposure_or_intervention": "calibration-aware quantization",
        "comparator": "ordinary quantization",
        "primary_outcome": "held-out robustness",
        "expected_relation": "increases",
        "mechanism": "Calibration protects activation outliers.",
        "novelty_delta": "Directly tests robustness rather than aggregate accuracy.",
        "experiment": "Randomize calibration and evaluate fixed held-out tasks.",
        "evidence_slots": ["E1", "E2"],
        "confounders": ["dataset shift"],
        "risks": ["benchmark sensitivity"],
        "secondary_predictions": [],
        "potential_insights": ["Outlier protection may explain robustness."],
    }
    original = project_hypothesis_candidate(
        spec,
        topic="llm compression",
        candidate_id="winner",
        research_scope={"mode": "broad", "label": "Broad topic"},
        evidence_scope_options=[],
        evidence_ledger=ledger,
        evidence_papers=papers,
        generation_mode="structured_exact",
    )

    class Agent:
        def __init__(self, model, host, timeout):
            self.model = model

        def ask(self, _prompt, **_kwargs):
            return spec, {"request_count": 1, "compilation_outcome": "exact"}

        def close(self):
            return None

    config = PipelineConfig.model_validate(
        {
            "context_init": [
                "topic",
                "selected_candidate",
                "research_scope",
                "candidate_scorecard",
                "evidence_ledger",
            ],
            "steps": [
                {
                    "name": "Revise Winning Candidate",
                    "type": "ollama_call",
                    "model": "model:test",
                    "inputs": [
                        "selected_candidate",
                        "research_scope",
                        "candidate_scorecard",
                        "evidence_ledger",
                    ],
                    "outputs": ["revised_candidate"],
                    "prompt_template": "Revise: {input_view}",
                    "prompt_view": "revision_v5",
                    "response_schema": "hypothesis_candidate",
                    "invalid_output": "recover",
                    "policy": "hypotheses.revise",
                }
            ],
        }
    )
    manager = PipelineManager(config, agent_factory=Agent)
    manager._validated_model_metadata = {"model:test": {"allocated_context_length": 16_384}}
    context = manager.run_pipeline(
        {
            "topic": "llm compression",
            "selected_candidate": original,
            "research_scope": {"mode": "broad", "label": "Broad topic"},
            "candidate_scorecard": [
                {
                    "candidate_id": "winner",
                    "unresolved_major_revisions": ["Add a meaningful control."],
                }
            ],
            "evidence_ledger": ledger,
            "evidence_scope_options": {"options": []},
            "top_papers": papers,
        }
    )

    assert context["_revision_audit"]["meaningful_change"] is False
    assert context["_revision_audit"]["unresolved_major_revisions"]
    assert context["_revision_audit"]["acceptance"] == "pending_authoritative_gate"
    assert context["revised_candidate"]["candidate_id"] == "winner"


def test_partial_429_opens_provider_circuit_and_counts_skips(monkeypatch) -> None:
    def fake_fetch(provider, _fetcher, _topic, _maximum, _cache):
        return [_paper("paper:partial")], multi_source.SourceStatus(
            provider=provider,
            status="partial",
            result_count=1,
            duration_seconds=0.1,
            error_category="http_429",
        )

    monkeypatch.setattr(multi_source, "_fetch_provider", fake_fetch)
    results = multi_source._fetch_provider_queries(
        "semantic_scholar", lambda *_args: [], ["one", "two", "three"], 3, None
    )
    status = multi_source._consolidate_statuses([item[2] for item in results])[0]

    assert [item[2].status for item in results] == ["partial", "skipped", "skipped"]
    assert status.planned_query_count == 3
    assert status.attempted_query_count == 1
    assert status.skipped_query_count == 2
    assert status.circuit_state == "open"
    assert status.circuit_reason == "http_429"


def test_proposal_cannot_override_audited_candidate_fields() -> None:
    candidate = {
        "candidate_id": "winner",
        "title": "Audited title",
        "hypothesis": "Audited hypothesis.",
        "mechanism": "Audited mechanism.",
        "proposed_experiment": "Audited method.",
        "predictions": ["Audited prediction."],
        "falsifier": "Audited falsifier.",
        "novel_insights": ["Audited insight."],
        "risks": ["Audited risk."],
        "supporting_paper_ids": ["paper:1"],
    }
    proposal = build_conservative_proposal(
        topic="compression",
        final_candidate=candidate,
        novelty_verification={"status": "uncertain"},
        evidence_papers=[_paper("paper:1")],
        model_content={
            "title": "Injected title",
            "hypothesis": "Injected hypothesis.",
            "proposed_method": "Injected method.",
            "predictions": ["Injected prediction."],
            "falsifier": "Injected falsifier.",
            "novel_insights": ["Injected insight."],
            "rationale": "Model rationale.",
            "abstract": "Model abstract.",
            "limitations_and_risks": ["Additional limitation."],
        },
        generation_mode="structured_exact",
    )

    assert proposal["title"] == "Audited title"
    assert proposal["hypothesis"] == "Audited hypothesis."
    assert proposal["proposed_method"] == "Audited method."
    assert proposal["falsifier"] == "Audited falsifier."
    assert proposal["predictions"] == ["Test whether audited prediction."]
