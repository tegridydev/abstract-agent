from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent import checkpoints, pipeline_runtime, response_validation
from agent.literature_types import PaperRecord, SearchQuery
from agent.schemas import ToolStep


def step(name: str, inputs: list[str], outputs: list[str]) -> ToolStep:
    return ToolStep.model_validate(
        {
            "name": name,
            "type": "tool_call",
            "function": "example.tool",
            "inputs": inputs,
            "outputs": outputs,
        }
    )


def test_checkpoint_normalization_checksum_and_read_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    normalized = checkpoints.normalize_context({"path": tmp_path, "nested": {"a": 1}})
    assert normalized["path"] == str(tmp_path)
    assert checkpoints.context_checksum({"b": 2, "a": 1}) == checkpoints.context_checksum(
        {"a": 1, "b": 2}
    )
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"status": "failed"}))
    assert checkpoints.read_checkpoint_object(path) == {"status": "failed"}

    path.write_text("[]")
    with pytest.raises(ValueError, match="root must be an object"):
        checkpoints.read_checkpoint_object(path)
    path.write_text("invalid")
    with pytest.raises(ValueError, match="could not load"):
        checkpoints.read_checkpoint_object(path)
    with pytest.raises(ValueError, match="could not load"):
        checkpoints.read_checkpoint_object(tmp_path / "missing.json")

    path.write_text("{}")
    monkeypatch.setattr(checkpoints, "MAX_CHECKPOINT_BYTES", 1)
    with pytest.raises(ValueError, match="safety limit"):
        checkpoints.read_checkpoint_object(path)


def test_search_query_phase_identity_validation_and_paper_record() -> None:
    assert SearchQuery(query_id="q", text="topic").phase == "literature"
    assert (
        SearchQuery(query_id="q", text="topic", phase="novelty", candidate_id="c").candidate_id
        == "c"
    )
    with pytest.raises(ValidationError, match="require candidate_id"):
        SearchQuery(query_id="q", text="topic", phase="novelty")
    with pytest.raises(ValidationError, match="cannot have candidate_id"):
        SearchQuery(query_id="q", text="topic", candidate_id="c")
    record = PaperRecord(
        provider="p",
        title="Paper",
        summary="Abstract",
        source="Provider",
        selection_score=0.878,
    )
    assert record.citations == 0 and record.sources == []
    assert record.selection_score == 0.878


def test_prepare_pipeline_context_fresh_and_resume_cleanup() -> None:
    steps = [step("First", ["topic"], ["one"]), step("Second", ["one"], ["two"])]
    initial = {"topic": "t", "one": "kept", "two": "stale"}
    fresh, fresh_steps = pipeline_runtime.prepare_pipeline_context(initial, steps, None)
    assert fresh == initial and fresh is not initial and fresh_steps is not steps

    resumed, remaining = pipeline_runtime.prepare_pipeline_context(initial, steps, "Second")
    assert resumed["one"] == "kept"
    assert "two" not in resumed
    assert resumed["_resume_count"] == 1
    assert resumed["_resumed_from_step"] == "Second"
    assert "_resume_timestamp_iso" in resumed
    assert [item.name for item in remaining] == ["Second"]


def test_prepare_pipeline_context_rejects_bad_resume_state() -> None:
    steps = [step("First", ["topic"], ["one"]), step("Second", ["one"], ["two"])]
    with pytest.raises(ValueError, match="not present"):
        pipeline_runtime.prepare_pipeline_context({"topic": "t"}, steps, "Missing")
    with pytest.raises(ValueError, match="missing inputs"):
        pipeline_runtime.prepare_pipeline_context({"topic": "t"}, steps, "Second")
    with pytest.raises(ValueError, match="count is invalid"):
        pipeline_runtime.prepare_pipeline_context(
            {"topic": "t", "one": 1, "_resume_count": "bad"}, steps, "Second"
        )
    with pytest.raises(ValueError, match="limit of 3"):
        pipeline_runtime.prepare_pipeline_context(
            {"topic": "t", "one": 1, "_resume_count": 3}, steps, "Second"
        )


def test_keyed_response_deduplicates_identical_and_rejects_conflicts() -> None:
    value = {"items": [{"id": "a", "score": 1}, {"id": "a", "score": 1}]}
    normalized, duplicates = response_validation.normalize_keyed_response_items(
        value, "items", "id"
    )
    assert normalized["items"] == [{"id": "a", "score": 1}]
    assert duplicates == ["a"]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        response_validation.normalize_keyed_response_items(
            {"items": [{"id": "a", "score": 1}, {"id": "a", "score": 2}]},
            "items",
            "id",
        )
    normalized, _ = response_validation.normalize_keyed_response_items(
        {"items": [{"id": "a", "score": 1}, {"id": "a", "score": 2}]},
        "items",
        "id",
        equivalent_keys=("id",),
    )
    assert len(normalized["items"]) == 1


def test_evidence_card_reconciliation_identical_conflicting_and_exact() -> None:
    card = {"paper_id": "p1", "findings": ["one"]}
    normalized, diagnostics = response_validation.reconcile_evidence_card_duplicates(
        {
            "cards": [
                card,
                dict(card),
                {"paper_id": "p2", "findings": ["a"]},
                {"paper_id": "p2", "findings": ["b"]},
            ]
        },
        exact=False,
    )
    assert normalized["cards"] == [card]
    assert diagnostics["identical_paper_ids"] == ["p1"]
    assert diagnostics["conflicting_paper_ids"] == ["p2"]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        response_validation.reconcile_evidence_card_duplicates(
            {
                "cards": [
                    {"paper_id": "p2", "findings": ["a"]},
                    {"paper_id": "p2", "findings": ["b"]},
                ]
            },
            exact=True,
        )
