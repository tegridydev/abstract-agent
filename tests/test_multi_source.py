import pytest

from agent import multi_source


def paper(**overrides):
    value = {
        "title": "Local language model compression",
        "summary": "A sufficiently long abstract about local language model compression methods.",
        "year": 2025,
        "citations": 10,
        "source": "Example",
        "url": "https://doi.org/10.1/example",
        "doi": "10.1/example",
    }
    value.update(overrides)
    return value


def test_canonical_identifier_normalizes_doi() -> None:
    assert multi_source.canonical_identifier(paper(doi="https://doi.org/10.1/EXAMPLE")) == (
        "doi:10.1/example"
    )


def test_canonical_identifier_strips_doi_url_noise() -> None:
    assert (
        multi_source.canonical_identifier(
            paper(doi="https://doi.org/10.1/EXAMPLE?source=x#fragment")
        )
        == "doi:10.1/example"
    )


def test_merge_duplicates_preserves_sources_and_highest_citations() -> None:
    merged = multi_source._merge_duplicates(
        [paper(source="Crossref", citations=2), paper(source="OpenAlex", citations=20)]
    )

    assert len(merged) == 1
    assert merged[0]["citations"] == 20
    assert merged[0]["sources"] == ["Crossref", "OpenAlex"]


def test_ranking_rejects_irrelevant_papers() -> None:
    merged = multi_source._merge_duplicates(
        [paper(), paper(title="Marine biology", summary="z" * 80, doi="10.1/other")]
    )
    ranked = multi_source.rank_papers(
        "language model compression",
        merged,
    )

    assert len(ranked) == 1
    assert ranked[0]["title"] == "Local language model compression"


def test_ranking_rejects_recent_highly_cited_generic_data_paper() -> None:
    papers = multi_source._merge_duplicates(
        [
            paper(
                title="Adaptive data compression algorithms",
                summary="A sufficiently long evaluation of adaptive data compression algorithms.",
            ),
            paper(
                title="Social protection data systems",
                summary="A contemporary evaluation of welfare services and public policy. " * 3,
                year=2026,
                citations=100_000,
                doi="10.1/unrelated",
            ),
        ]
    )

    ranked = multi_source.rank_papers("novel data compression algorithms", papers)

    assert [item["doi"] for item in ranked] == ["10.1/example"]


def test_ranking_scores_relevant_merged_paper() -> None:
    merged = multi_source._merge_duplicates([paper()])
    ranked = multi_source.rank_papers("language model compression", merged)

    assert len(ranked) == 1
    assert ranked[0]["composite_score"] > 0


def test_reconstruct_openalex_abstract() -> None:
    assert multi_source.reconstruct_abstract({"hello": [0], "world": [1]}) == "hello world"


def test_aggregate_records_provider_errors(monkeypatch) -> None:
    def successful(topic, max_results, client):
        return [paper()]

    def failing(topic, max_results, client):
        raise RuntimeError("offline")

    monkeypatch.setattr(multi_source, "FETCHERS", {"ok": successful, "bad": failing})
    ranked, fetched, statuses = multi_source.aggregate_sources(
        "language model compression", max_results=3, use_cache=False
    )

    assert fetched == 1
    assert len(ranked) == 1
    assert {item["provider"]: item["status"] for item in statuses} == {
        "bad": "error",
        "ok": "ok",
    }


def test_aggregate_skips_same_run_disabled_provider_without_fetching(monkeypatch) -> None:
    def must_not_fetch(*_args):
        raise AssertionError("disabled provider must not be contacted")

    monkeypatch.setattr(multi_source, "FETCHERS", {"failed": must_not_fetch})
    ranked, fetched, statuses = multi_source.aggregate_sources(
        "language model compression",
        max_results=3,
        use_cache=False,
        disabled_providers={"failed"},
    )

    assert ranked == []
    assert fetched == 0
    assert statuses == [
        {
            "provider": "failed",
            "status": "skipped",
            "result_count": 0,
            "duration_seconds": 0.0,
            "error_category": "circuit_open",
            "message": "provider skipped after a hard failure earlier in this run",
            "rate_limit": {},
            "query_count": 0,
            "planned_query_count": 1,
            "attempted_query_count": 0,
            "succeeded_query_count": 0,
            "skipped_query_count": 1,
            "circuit_state": "open",
            "circuit_reason": "prior_run_stage_failure",
            "failed_query_count": 0,
            "cached_query_count": 0,
            "warning_categories": [],
            "warning_query_count": 0,
            "retry_count": 0,
            "retry_categories": [],
        }
    ]


def test_multi_query_retrieval_tracks_facet_coverage(monkeypatch) -> None:
    def successful(topic, max_results, client):
        return [
            paper(
                title=f"Language model compression {topic}",
                summary=f"A sufficiently long study of language model compression and {topic}.",
                doi=f"10.1/{topic}",
            )
        ]

    monkeypatch.setattr(multi_source, "FETCHERS", {"ok": successful})
    ranked, fetched, statuses = multi_source.aggregate_sources(
        ["quantization", "pruning"], max_results=3, use_cache=False
    )

    assert fetched == 2
    assert {query for item in ranked for query in item["matched_queries"]} == {
        "quantization",
        "pruning",
    }
    assert len(statuses) == 1
    assert statuses[0]["query_count"] == 2


def test_offline_aggregation_uses_only_local_knowledge(monkeypatch) -> None:
    class Store:
        def search_papers(self, query, limit):
            assert query == "language model compression"
            assert limit == 3
            return [
                paper(
                    provider="crossref",
                    provider_id="10.1/example",
                    paper_id="doi:10.1/example",
                    sources=["crossref"],
                    knowledge={"trust_tier": "external_source"},
                )
            ]

    def must_not_fetch(*unused):
        raise AssertionError("offline refresh must not call providers")

    monkeypatch.setattr(multi_source, "FETCHERS", {"network": must_not_fetch})

    ranked, fetched, statuses = multi_source.aggregate_sources(
        "language model compression",
        max_results=3,
        knowledge_store=Store(),
        offline=True,
    )

    assert fetched == 1
    assert ranked[0]["paper_id"] == "doi:10.1/example"
    assert statuses[0]["provider"] == "local_knowledge"
    assert statuses[0]["status"] == "cached"


def test_generated_ideas_expand_queries_but_do_not_become_evidence(monkeypatch) -> None:
    requested = []

    class Store:
        def related_idea_queries(self, query, limit):
            assert limit == 1
            return [f"{query} sparsity quantization"]

        def search_papers(self, query, limit):
            return []

        def ingest_papers(self, papers):
            return len(papers)

    def fetcher(topic, max_results, client):
        requested.append(topic)
        return [paper(doi=f"10.1/{len(requested)}")]

    monkeypatch.setattr(multi_source, "FETCHERS", {"source": fetcher})

    ranked, _, _ = multi_source.aggregate_sources(
        "language model compression",
        max_results=3,
        knowledge_store=Store(),
        use_cache=False,
    )

    assert requested == ["language model compression"]
    assert ranked
    assert all(paper_record["source"] == "Example" for paper_record in ranked)


def test_arxiv_adapter_uses_hardened_http_client() -> None:
    class Response:
        text = """<feed><entry><id>https://arxiv.org/abs/1234.5678v2</id>
        <title>Test paper</title><summary>A sufficiently long abstract for testing.</summary>
        <published>2026-01-01T00:00:00Z</published><author><name>A. Author</name></author>
        </entry></feed>"""

    class Client:
        def get(self, url, **kwargs):
            assert url == "https://export.arxiv.org/api/query"
            return Response()

    papers = multi_source.fetch_arxiv("test", 1, Client())

    assert papers[0]["arxiv_id"] == "1234.5678"
    assert papers[0]["authors"] == "A. Author"


def test_error_sanitizer_redacts_api_keys() -> None:
    settings = multi_source.SourceSettings(
        contact_email=None,
        semantic_scholar_api_key="semantic-secret",
        openalex_api_key="openalex-secret",
        ncbi_api_key="ncbi-secret",
    )

    sanitized = multi_source._sanitize_error(
        "semantic-secret openalex-secret ncbi-secret", settings
    )

    assert sanitized == "[REDACTED] [REDACTED] [REDACTED]"


def test_hal_adapter_normalizes_abstract_records() -> None:
    class Response:
        def json(self):
            return {
                "response": {
                    "docs": [
                        {
                            "docid": 7,
                            "title_s": ["Compression paper"],
                            "abstract_s": ["A complete abstract about compression algorithms."],
                            "authFullName_s": ["A. Author"],
                            "producedDateY_i": 2025,
                            "doiId_s": "10.1/hal",
                            "uri_s": "https://hal.science/hal-7",
                        }
                    ]
                }
            }

    class Client:
        def get(self, url, **kwargs):
            assert url == "https://api.archives-ouvertes.fr/search/"
            return Response()

    assert multi_source.fetch_hal("compression", 3, Client())[0]["doi"] == "10.1/hal"


def test_openaire_and_core_adapters_normalize_abstract_records() -> None:
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class OpenAireClient:
        def get(self, url, **kwargs):
            return Response(
                {
                    "results": [
                        {
                            "id": "oa:1",
                            "mainTitle": "Compression study",
                            "description": "A full description of a compression algorithm.",
                            "publicationDate": "2025-01-01",
                            "authors": [{"fullName": "A. Author"}],
                            "pids": [{"scheme": "doi", "value": "10.1/openaire"}],
                        }
                    ]
                }
            )

    class CoreClient:
        def get(self, url, **kwargs):
            return Response(
                {
                    "results": [
                        {
                            "id": 2,
                            "title": "CORE compression study",
                            "abstract": "A full abstract of a lossless compression algorithm.",
                            "authors": [{"name": "B. Author"}],
                            "yearPublished": 2024,
                            "doi": "10.1/core",
                        }
                    ]
                }
            )

    assert multi_source.fetch_openaire("compression", 3, OpenAireClient())[0]["doi"] == (
        "10.1/openaire"
    )
    assert multi_source.fetch_core("compression", 3, CoreClient())[0]["doi"] == "10.1/core"


def test_provider_queries_are_serialized_and_statuses_consolidated(monkeypatch) -> None:
    sleeps = []

    def fake_fetch(provider, fetcher, topic, max_results, cache):
        del fetcher, max_results, cache
        return [paper(doi=f"10.1/{topic}")], multi_source.SourceStatus(
            provider=provider,
            status="ok" if topic == "first" else "error",
            result_count=1 if topic == "first" else 0,
            duration_seconds=0.1,
            error_category=None if topic == "first" else "rate_limit",
        )

    monkeypatch.setattr(multi_source, "_fetch_provider", fake_fetch)
    monkeypatch.setattr(multi_source.time, "sleep", sleeps.append)
    results = multi_source._fetch_provider_queries(
        "semantic_scholar", lambda *args: [], ["first", "second"], 3, None
    )
    statuses = multi_source._consolidate_statuses([item[2] for item in results])

    assert sleeps == [1.1]
    assert statuses[0].status == "partial"
    assert statuses[0].failed_query_count == 1
    assert statuses[0].message == "1/2 attempted queries failed (rate_limit)"


def test_stale_cache_fallback_is_usable_but_reported_as_degraded() -> None:
    status = multi_source._consolidate_statuses(
        [
            multi_source.SourceStatus(
                provider="example",
                status="cached",
                result_count=2,
                duration_seconds=0.1,
                error_category="timeout",
            )
        ]
    )[0]

    assert status.status == "cached"
    assert status.failed_query_count == 1
    assert status.cached_query_count == 1


def test_ranking_weights_must_sum_to_one(monkeypatch) -> None:
    monkeypatch.setattr(multi_source, "FETCHERS", {})

    with pytest.raises(ValueError, match="sum to 1.0"):
        multi_source.aggregate_sources(
            "topic",
            use_cache=False,
            relevance_weight=1,
            recency_weight=1,
            citation_weight=1,
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [("max_results", 0), ("max_workers", 0), ("cache_ttl_seconds", -1), ("minimum_relevance", 2)],
)
def test_aggregate_rejects_unsafe_bounds(monkeypatch, argument, value) -> None:
    monkeypatch.setattr(multi_source, "FETCHERS", {})

    with pytest.raises(ValueError):
        multi_source.aggregate_sources("topic", use_cache=False, **{argument: value})


def test_cache_write_failure_keeps_live_results() -> None:
    class BrokenCache:
        def get(self, *args):
            return None

        def put(self, *args):
            raise OSError("read-only cache")

    def fetcher(topic, max_results, client):
        return [paper(title="Live paper", summary="A " * 40, doi="10.1/live")]

    papers, status = multi_source._fetch_provider("example", fetcher, "topic", 1, BrokenCache())

    assert len(papers) == 1
    assert status.status == "partial"
    assert status.warning_categories == ["cache_write_error"]
    assert status.error_category is None


def test_provider_fallback_ids_are_namespaced() -> None:
    first = {"provider": "hal", "provider_id": "123", "title": "A"}
    second = {"provider": "core", "provider_id": "123", "title": "B"}

    assert multi_source.canonical_identifier(first) == "provider_id:hal:123"
    assert multi_source.canonical_identifier(second) == "provider_id:core:123"


def test_arxiv_identifier_preserves_legacy_archive() -> None:
    assert (
        multi_source.parse_arxiv_identifier("https://arxiv.org/abs/hep-th/9901001v2")
        == "hep-th/9901001"
    )
    assert multi_source.parse_arxiv_identifier("https://arxiv.org/abs/1501.00001v3") == (
        "1501.00001"
    )


def test_openalex_abstract_reconstruction_rejects_unsafe_positions() -> None:
    assert multi_source.reconstruct_abstract({"hello": [0], "world": [2]}) == "hello world"
    assert multi_source.reconstruct_abstract({"boom": [10**9]}) == ""
    assert multi_source.reconstruct_abstract({"bad": [True]}) == ""


def test_aggregation_is_stable_across_provider_completion_order(monkeypatch) -> None:
    def one(topic, max_results, client):
        return [paper(title="Same", summary="Stable summary " * 8, doi="10.1/same")]

    def two(topic, max_results, client):
        return [
            paper(
                title="Same",
                summary="Alternative summary " * 8,
                doi="10.1/same",
                source="Second",
            )
        ]

    monkeypatch.setattr(multi_source, "FETCHERS", {"one": one, "two": two})
    first = multi_source.aggregate_sources("same", use_cache=False)[0]
    monkeypatch.setattr(multi_source, "FETCHERS", {"one": one, "two": two})
    second = multi_source.aggregate_sources("same", use_cache=False)[0]

    assert first == second
