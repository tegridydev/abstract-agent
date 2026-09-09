from __future__ import annotations

import json
import math

import pytest

from agent.agent import PipelineManager
from agent.knowledge.embeddings import (
    EmbeddingUnavailableError,
    OllamaEmbeddingService,
    cosine,
    pack_vector,
    unpack_vector,
)
from agent.knowledge.evidence import build_evidence_ledger, canonical_source
from agent.knowledge.store import KnowledgeStore
from agent.schemas import PipelineConfig


def paper(
    paper_id: str = "doi:10.1/example",
    summary: str = (
        "Quantization reduces model memory while preserving accuracy. "
        "A controlled benchmark compares four compression methods."
    ),
) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "provider": "crossref",
        "provider_id": paper_id,
        "title": "Model compression benchmark",
        "summary": summary,
        "source": "crossref",
        "sources": ["crossref"],
        "relevance_judgement": {
            "score": 5,
            "role": "direct_evidence",
            "rationale": "The paper directly measures model compression.",
        },
    }


class UsefulEmbeddings:
    model = "embedding:test"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = [[1.0, 0.0]]
        vectors.extend([1.0, 0.0] if "Quantization" in text else [0.0, 1.0] for text in texts[1:])
        return vectors


class BrokenEmbeddings:
    model = "embedding:broken"

    def embed(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise EmbeddingUnavailableError("unavailable")


def test_extracts_exact_source_spans_with_embedding_provenance() -> None:
    ledger, metrics = build_evidence_ledger(
        [paper()], "model compression", embedding_service=UsefulEmbeddings()
    )

    card = ledger["cards"][0]
    source = canonical_source(str(paper()["summary"]))
    assert metrics["modes"] == {"embedding_extractive": 1}
    assert card["provenance"]["embedding_model"] == "embedding:test"
    for finding in card["findings"]:
        assert finding["claim"] == finding["source_passage"]
        assert source[finding["source_start"] : finding["source_end"]] == finding["source_passage"]


def test_embedding_failure_falls_back_to_lexical_evidence() -> None:
    ledger, metrics = build_evidence_ledger(
        [paper()], "model compression", embedding_service=BrokenEmbeddings()
    )

    assert metrics["modes"] == {"lexical_extractive": 1}
    assert ledger["cards"][0]["provenance"]["embedding_model"] is None


def test_exact_hash_evidence_is_reused_but_changed_source_is_not(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "knowledge") as store:
        first, _ = build_evidence_ledger([paper()], "compression", knowledge_store=store)
        second, metrics = build_evidence_ledger(
            [paper()],
            "compression",
            embedding_service=BrokenEmbeddings(),
            knowledge_store=store,
        )
        changed, changed_metrics = build_evidence_ledger(
            [paper(summary="A newly corrected abstract about compression systems.")],
            "compression",
            knowledge_store=store,
        )

    assert second["cards"][0]["findings"] == first["cards"][0]["findings"]
    assert second["cards"][0]["provenance"]["reused"] is True
    assert metrics["reused_count"] == 1
    assert changed_metrics["reused_count"] == 0
    assert (
        changed["cards"][0]["provenance"]["source_content_hash"]
        != first["cards"][0]["provenance"]["source_content_hash"]
    )


class EmbedClient:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings
        self.calls: list[dict[str, object]] = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        return {"embeddings": self.embeddings}


def test_embedding_adapter_batches_disables_truncation_and_normalizes() -> None:
    client = EmbedClient([[3.0, 4.0], [0.0, 2.0]])
    service = OllamaEmbeddingService("embed:test", "http://localhost:11434", client=client)

    vectors = service.embed(["one", "two"])

    assert client.calls == [{"model": "embed:test", "input": ["one", "two"], "truncate": False}]
    assert all(math.isclose(sum(value * value for value in vector), 1.0) for vector in vectors)


def test_embedding_vector_storage_and_cosine_are_dimension_safe() -> None:
    packed = pack_vector([3.0, 4.0])
    restored = unpack_vector(packed, 2)

    assert restored == pytest.approx([0.6, 0.8])
    assert cosine(restored, restored) == pytest.approx(1.0)
    assert cosine([1.0], [1.0, 2.0]) == 0.0
    with pytest.raises(ValueError, match="dimensions"):
        unpack_vector(packed, 3)


@pytest.mark.parametrize(
    "vectors",
    [[], [[0.0, 0.0]], [[1.0], [1.0, 2.0]], [[float("nan")]]],
)
def test_embedding_adapter_rejects_unsafe_vectors(vectors) -> None:
    service = OllamaEmbeddingService(
        "embed:test", "http://localhost:11434", client=EmbedClient(vectors)
    )

    with pytest.raises(EmbeddingUnavailableError):
        service.embed(["one"] if len(vectors) != 2 else ["one", "two"])


def test_knowledge_store_search_projection_and_generated_idea_trust(tmp_path) -> None:
    root = tmp_path / "knowledge"
    with KnowledgeStore(root) as store:
        store.ingest_papers([paper()])
        results = store.search_papers("quantization compression")
        run_id = store.ingest_run(
            "compression",
            {
                "top_papers": [paper()],
                "evidence_ledger": {"cards": []},
                "hypothesis_candidates": [
                    {
                        "candidate_id": "idea-1",
                        "hypothesis": "Compression sparsity can improve quantization stability.",
                    }
                ],
            },
        )
        related = store.related_idea_queries("compression")
        status = store.status()

    assert results[0]["paper_id"] == "doi:10.1/example"
    assert status["counts"]["papers"] == 1
    insight = json.loads((root / "records" / "insights.jsonl").read_text().strip())
    assert insight["trust_tier"] == "generated_idea"
    assert "external_evidence" in insight["prohibited_uses"]
    assert related and "compression" in related[0]
    assert "quantization" in related[0]
    assert (root / "findings" / f"{run_id}.md").exists()
    assert (root / "ideas" / f"{run_id}.md").exists()


def test_projection_import_preserves_only_external_source_records(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    with KnowledgeStore(source) as store:
        store.ingest_papers([paper()])
        store.export_projections()
    with KnowledgeStore(destination) as store:
        counts = store.import_projections(source)
        result = store.search_papers("compression")

    assert counts["papers"] == 1
    assert result[0]["paper_id"] == "doi:10.1/example"


def test_store_falls_back_to_like_search_and_validates_limits(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "knowledge") as store:
        assert store.ingest_papers([{}, paper()]) == 1
        store.fts_available = False
        assert store.search_papers("compression")[0]["paper_id"] == "doi:10.1/example"
        assert store.reindex() == 0
        with pytest.raises(ValueError, match="between 1 and 100"):
            store.search_papers("compression", 0)


def test_legacy_evidence_is_not_reused_as_extractive_source_evidence(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "knowledge") as store:
        store.ingest_papers([paper()])
        count = store.ingest_evidence(
            {
                "cards": [
                    {
                        "paper_id": "doi:10.1/example",
                        "findings": [{"claim": "Generated", "source_passage": "Generated"}],
                        "relevance": "Legacy model output",
                    }
                ]
            }
        )

        assert count == 0
        assert store.status()["counts"]["evidence"] == 0


def test_long_or_missing_abstracts_still_produce_bounded_exact_cards() -> None:
    long_summary = " ".join(["compression"] * 500)
    ledger, _ = build_evidence_ledger(
        [paper(summary=long_summary), paper("paper:title", summary="")], "compression"
    )

    assert len(ledger["cards"]) == 2
    assert all(
        len(finding["source_passage"]) <= 360
        for card in ledger["cards"]
        for finding in card["findings"]
    )
    assert ledger["cards"][1]["findings"][0]["source_passage"] == ("Model compression benchmark")
    assert ledger["cards"][1]["provenance"]["source_field"] == "title"


def test_evidence_builder_rejects_papers_without_canonical_ids() -> None:
    invalid = paper()
    invalid.pop("paper_id")

    with pytest.raises(ValueError, match="canonical paper_id"):
        build_evidence_ledger([invalid], "compression")


def test_pipeline_integrates_evidence_metrics_and_persists_completed_run(tmp_path) -> None:
    config = PipelineConfig.model_validate(
        {
            "context_init": ["topic", "top_papers", "research_scope"],
            "steps": [
                {
                    "name": "Build Evidence Ledger",
                    "type": "tool_call",
                    "function": "knowledge.build_evidence_ledger",
                    "inputs": ["top_papers", "topic", "research_scope"],
                    "outputs": ["evidence_ledger", "evidence_extraction_metrics"],
                }
            ],
            "final_outputs": ["evidence_ledger"],
        }
    )
    root = tmp_path / "knowledge"
    manager = PipelineManager(config, knowledge_enabled=True, knowledge_dir=root)

    result = manager.run_pipeline(
        {
            "topic": "compression",
            "top_papers": [paper()],
            "research_scope": {"mode": "broad", "label": "Broad topic"},
        }
    )
    status = result["_knowledge"]["status"]
    manager.close()

    assert result["evidence_extraction_metrics"]["paper_count"] == 1
    assert status["counts"]["papers"] == 1
    assert status["counts"]["runs"] == 1
    assert list((root / "findings").glob("*.md"))
