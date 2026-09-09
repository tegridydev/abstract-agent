from __future__ import annotations

import json

from agent.agent import PipelineManager
from agent.pipeline.errors import ModelTransportError
from agent.pipeline.model_io import consume_chat_stream
from agent.pipeline.recovery import (
    prepare_candidate_critique_packets,
    prepare_novelty_recovery_packets,
)
from agent.pipeline.stages import normalize_critique_item, normalize_novelty_item
from agent.schemas import PipelineConfig


def _candidate(candidate_id: str, paper_ids: list[str]) -> dict:
    return {
        "candidate_id": candidate_id,
        "title": f"Candidate {candidate_id}",
        "hypothesis": "A measurable hypothesis",
        "mechanism": "A mechanism",
        "novelty_delta": "A cautious difference",
        "supporting_paper_ids": paper_ids,
        "predictions": ["Prediction"],
        "falsifier": "Falsifier",
        "proposed_experiment": "Experiment",
        "novel_insights": ["Insight"],
        "risks": ["Risk"],
        "applicability": "broad",
    }


def test_critique_recovery_packet_contains_only_candidate_evidence() -> None:
    candidates = [_candidate("a", ["p1"]), _candidate("b", ["p2"])]
    packets = prepare_candidate_critique_packets(
        candidates,
        {"critiques": [{"candidate_id": "a"}]},
        {
            "cards": [
                {"paper_id": "p1", "finding": "A" * 1000},
                {"paper_id": "p2", "finding": "B" * 1000},
            ]
        },
        {"unresolved_questions": [{"statement": "Gap", "paper_ids": ["p2"]}]},
    )

    assert [packet["candidate_id"] for packet in packets] == ["b"]
    assert [card["paper_id"] for card in packets[0]["evidence_cards"]] == ["p2"]
    assert "p1" not in json.dumps(packets[0])
    assert len(json.dumps(packets[0])) <= 6000


def test_recovery_packet_hard_cap_handles_maximally_populated_candidate() -> None:
    candidate = _candidate("a", [f"paper:{index}" for index in range(8)])
    candidate.update(
        {
            "title": "t" * 240,
            "hypothesis": "h" * 1000,
            "mechanism": "m" * 1000,
            "novelty_delta": "n" * 600,
            "predictions": ["p" * 1000] * 6,
            "falsifier": "f" * 800,
            "proposed_experiment": "e" * 1200,
            "risks": ["r" * 1000] * 8,
            "novel_insights": ["i" * 1000] * 6,
        }
    )
    packets = prepare_candidate_critique_packets(
        [candidate],
        {"critiques": []},
        {"cards": [{"paper_id": f"paper:{index}", "finding": "x" * 5000} for index in range(8)]},
    )

    assert packets[0]["candidate_id"] == "a"
    assert len(json.dumps(packets[0], ensure_ascii=False, separators=(",", ":"))) <= 6000


def test_novelty_packet_prioritizes_candidate_matched_papers_and_is_bounded() -> None:
    papers = [
        {
            "paper_id": f"p{index}",
            "title": f"Paper {index}",
            "summary": "summary " * 500,
            "matched_candidate_ids": ["b"] if index >= 6 else ["a"],
            "composite_score": index / 10,
        }
        for index in range(12)
    ]
    packets = prepare_novelty_recovery_packets(
        _candidate("a", ["p1"]),
        _candidate("b", ["p9"]),
        {"verifications": [{"candidate_id": "a"}]},
        papers,
    )

    assert len(packets) == 1
    assert packets[0]["candidate_id"] == "b"
    assert len(packets[0]["prior_art"]) <= 6
    assert all(int(item["paper_id"][1:]) >= 6 for item in packets[0]["prior_art"])
    assert len(json.dumps(packets[0])) <= 6000


def test_novelty_normalization_filters_unknown_ids_and_downgrades_status() -> None:
    item, operations = normalize_novelty_item(
        {
            "candidate_id": "a",
            "status": "definitely_novel",
            "closest_prior_art_paper_ids": ["known", "unknown", "known"],
            "materially_matching_paper_ids": ["unknown"],
            "reasoning": "",
            "limitations": [],
        },
        expected_candidate_ids={"a"},
        allowed_paper_ids={"known"},
    )

    assert item is not None
    assert item["status"] == "uncertain"
    assert item["closest_prior_art_paper_ids"] == ["known"]
    assert item["materially_matching_paper_ids"] == []
    assert item["reasoning"]
    assert operations


def test_incomplete_critique_is_retained_but_excluded_from_ranking() -> None:
    item, operations = normalize_critique_item(
        {"candidate_id": "a", "grounding": 9, "issues": []},
        expected_candidate_ids={"a"},
    )

    assert item is not None
    assert item["grounding"] == 5
    assert item["score_source"] == "fallback"
    assert item["issues"] and item["required_revisions"]
    assert operations


def test_focused_recovery_transport_failure_uses_fallback_and_120_second_client() -> None:
    timeouts: list[float] = []

    class FailingAgent:
        def __init__(self, model: str, host: str, timeout: float) -> None:
            del model, host
            timeouts.append(timeout)

        def ask(self, *_args, **_kwargs):
            raise ModelTransportError("timed out")

        def close(self) -> None:
            return None

    config = PipelineConfig.model_validate(
        {
            "context_init": ["packets"],
            "steps": [
                {
                    "name": "Recover Missing Critiques",
                    "type": "ollama_call",
                    "model": "tiny:test",
                    "inputs": ["packets"],
                    "outputs": ["recovered"],
                    "map_input": "packets",
                    "map_item_key": "review_packet",
                    "prompt_template": "Review {review_packet}",
                    "response_schema": "candidate_critiques",
                    "invalid_output": "recover",
                    "policy": "critiques.recover",
                    "max_output_tokens": 512,
                    "inactivity_timeout_seconds": 120,
                    "transport_failure": "conservative_fallback",
                }
            ],
            "final_outputs": ["recovered"],
        }
    )
    manager = PipelineManager(config, agent_factory=FailingAgent)
    result = manager.run_pipeline(
        {"packets": [{"candidate_id": "a", "candidate": _candidate("a", [])}]}
    )

    assert timeouts == [120]
    assert result["recovered"][0]["critiques"][0]["candidate_id"] == "a"
    assert result["recovered"][0]["critiques"][0]["score_source"] == "fallback"
    assert (
        result["_llm_metrics"]["Recover Missing Critiques"][0]["compilation_outcome"] == "fallback"
    )
    assert (
        result["_llm_metrics"]["Recover Missing Critiques"][0]["recovery_source"]
        == "deterministic_fallback"
    )


def test_direct_structured_stage_never_invokes_semantic_draft() -> None:
    requests: list[str] = []

    class SinglePassAgent:
        def __init__(self, *_args) -> None:
            return None

        def draft(self, *_args, **_kwargs):
            raise AssertionError("direct structured generation must not request a semantic draft")

        def ask(self, prompt: str, **_kwargs):
            requests.append(prompt)
            return "complete", {"request_count": 1, "compilation_outcome": "exact"}

        def close(self) -> None:
            return None

    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic"],
            "steps": [
                {
                    "name": "Generate Once",
                    "type": "ollama_call",
                    "model": "tiny:test",
                    "inputs": ["topic"],
                    "outputs": ["result"],
                    "prompt_template": "Generate a result for {topic}",
                    "generation_strategy": "direct_structured",
                }
            ],
            "final_outputs": ["result"],
        }
    )

    result = PipelineManager(config, agent_factory=SinglePassAgent).run_pipeline(
        {"topic": "compression"}
    )

    assert result["result"] == "complete"
    assert len(requests) == 1
    assert result["_llm_metrics"]["Generate Once"]["request_count"] == 1


def test_stream_transport_accumulates_public_metrics_and_progress() -> None:
    progress: list[tuple[int, int]] = []
    result = consume_chat_stream(
        [
            {"message": {"content": '{"value":'}},
            {
                "message": {"content": "1}", "thinking": "done"},
                "done": True,
                "eval_count": 2,
            },
        ],
        attempt=1,
        progress_callback=lambda attempt, characters: progress.append((attempt, characters)),
    )

    assert result.content == '{"value":1}'
    assert result.thinking == "done"
    assert result.metrics["eval_count"] == 2
    assert result.first_token_seconds is not None
    assert progress[-1] == (1, len(result.content) + len(result.thinking))
