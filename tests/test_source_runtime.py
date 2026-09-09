from __future__ import annotations

from typing import Any

import httpx
import pytest

from agent import multi_source
from agent.literature_types import SearchQuery
from agent.schemas import SourceStatus


def paper(**overrides: Any) -> dict[str, Any]:
    value = {
        "title": "Compression algorithm study",
        "summary": "A sufficiently detailed study of compression algorithm performance. " * 2,
        "year": 2025,
        "citations": 4,
        "source": "Example",
        "provider": "example",
        "provider_id": "1",
    }
    value.update(overrides)
    return value


class FakeClient:
    exception: Exception | None = None
    rate_limit = {"x-ratelimit-remaining": "1"}

    def __init__(self) -> None:
        self.settings = multi_source.SourceSettings(None, "secret", None, None)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FreshCache:
    def get(self, *args: Any) -> tuple[list[dict[str, Any]], bool]:
        del args
        return [paper()], True


class StaleCache:
    def get(self, *args: Any) -> tuple[list[dict[str, Any]], bool]:
        del args
        return [paper()], False

    def put(self, *args: Any) -> None:
        del args


class BrokenReadCache:
    def get(self, *args: Any) -> None:
        del args
        raise OSError("unavailable")

    def put(self, *args: Any) -> None:
        del args


def test_fetch_provider_returns_fresh_cache_without_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        multi_source,
        "ScholarlyClient",
        lambda: (_ for _ in ()).throw(AssertionError("client created")),
    )
    papers, status = multi_source._fetch_provider(
        "example",
        lambda *args: [],
        "topic",
        2,
        FreshCache(),  # type: ignore[arg-type]
    )
    assert len(papers) == 1
    assert status.status == "cached"


def test_fetch_provider_cache_read_failure_reports_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr(multi_source, "ScholarlyClient", lambda: client)
    papers, status = multi_source._fetch_provider(
        "example",
        lambda *args: [paper()],
        "topic",
        2,
        BrokenReadCache(),  # type: ignore[arg-type]
    )
    assert len(papers) == 1
    assert status.status == "partial"
    assert status.warning_categories == ["cache_read_error"]
    assert client.closed


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (
            httpx.ReadTimeout(
                "secret timeout", request=httpx.Request("GET", "https://example.test")
            ),
            "timeout",
        ),
        (
            httpx.ConnectError(
                "secret network", request=httpx.Request("GET", "https://example.test")
            ),
            "network_error",
        ),
        (RuntimeError("secret parser"), "provider_error"),
    ],
)
def test_fetch_provider_maps_errors_and_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch, error: Exception, category: str
) -> None:
    client = FakeClient()
    monkeypatch.setattr(multi_source, "ScholarlyClient", lambda: client)

    def fail(*args: Any) -> list[dict[str, Any]]:
        del args
        raise error

    papers, status = multi_source._fetch_provider("example", fail, "topic", 2, None)
    assert papers == []
    assert status.status == "error"
    assert status.error_category == category
    assert "secret" not in (status.message or "")
    assert "[REDACTED]" in (status.message or "")
    assert client.closed


def test_fetch_provider_maps_http_status_and_uses_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    monkeypatch.setattr(multi_source, "ScholarlyClient", lambda: client)
    response = httpx.Response(
        429,
        request=httpx.Request("GET", "https://example.test"),
    )
    error = httpx.HTTPStatusError("limited", request=response.request, response=response)

    def fail(*args: Any) -> list[dict[str, Any]]:
        del args
        raise error

    papers, status = multi_source._fetch_provider(
        "example",
        fail,
        "topic",
        2,
        StaleCache(),  # type: ignore[arg-type]
    )
    assert len(papers) == 1
    assert status.status == "cached"
    assert status.error_category == "http_429"
    assert "stale cache" in (status.message or "")


def test_provider_query_circuit_skips_remaining_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        multi_source,
        "_fetch_provider",
        lambda *args: (
            [],
            SourceStatus(
                provider="core",
                status="error",
                result_count=0,
                duration_seconds=0,
                error_category="network_error",
            ),
        ),
    )
    results = multi_source._fetch_provider_queries(
        "core", lambda *args: [], ["one", "two", "three"], 2, None
    )
    assert [item[0] for item in results] == ["one", "two", "three"]
    assert results[0][2].error_category == "network_error"
    assert results[1][2].error_category == "network_error"
    assert results[2][2].error_category == "circuit_open"


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([SourceStatus(provider="p", status="error", result_count=0, duration_seconds=0)], "error"),
        (
            [SourceStatus(provider="p", status="cached", result_count=1, duration_seconds=0)],
            "cached",
        ),
        ([SourceStatus(provider="p", status="ok", result_count=1, duration_seconds=0)], "ok"),
        ([SourceStatus(provider="p", status="empty", result_count=0, duration_seconds=0)], "empty"),
        (
            [
                SourceStatus(
                    provider="p",
                    status="partial",
                    result_count=1,
                    duration_seconds=0,
                    warning_categories=["cache_read_error"],
                    warning_query_count=1,
                )
            ],
            "partial",
        ),
    ],
)
def test_consolidate_status_value_matrix(statuses: list[SourceStatus], expected: str) -> None:
    result = multi_source._consolidate_statuses(statuses)[0]
    assert result.status == expected


def test_consolidate_combines_failure_warning_message_and_rate_limits() -> None:
    result = multi_source._consolidate_statuses(
        [
            SourceStatus(
                provider="p",
                status="error",
                result_count=0,
                duration_seconds=0,
                error_category="timeout",
                rate_limit={"a": "1"},
                retry_count=3,
                retry_categories=["timeout"],
            ),
            SourceStatus(
                provider="p",
                status="partial",
                result_count=2,
                duration_seconds=0,
                warning_categories=["cache_write_error"],
                warning_query_count=1,
                rate_limit={"b": "2"},
                retry_count=1,
                retry_categories=["http_503"],
            ),
        ]
    )[0]
    assert result.status == "partial"
    assert "queries failed" in result.message
    assert "queries had warnings" in result.message
    assert result.rate_limit == {"a": "1", "b": "2"}
    assert result.retry_count == 4
    assert result.retry_categories == ["http_503", "timeout"]


def test_normalize_queries_accepts_models_dicts_and_rejects_empty() -> None:
    queries = multi_source._normalize_search_queries(
        [
            SearchQuery(query_id="q1", text="one"),
            {
                "query_id": "q2",
                "text": "two",
                "phase": "novelty",
                "candidate_id": "c2",
            },
            " three ",
        ]
    )
    assert [query.text for query in queries] == ["one", "two", "three"]
    with pytest.raises(ValueError, match="at least one"):
        multi_source._normalize_search_queries([])


def test_aggregate_worker_failure_becomes_provider_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(multi_source, "FETCHERS", {"broken": lambda *args: []})
    monkeypatch.setattr(
        multi_source,
        "_fetch_provider_queries",
        lambda *args: (_ for _ in ()).throw(RuntimeError("worker")),
    )
    ranked, fetched, statuses = multi_source.aggregate_sources(
        "compression algorithm", use_cache=False
    )
    assert ranked == [] and fetched == 0
    assert statuses[0]["error_category"] == "provider_worker_error"


def test_check_sources_sorts_health_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        multi_source,
        "FETCHERS",
        {"z": lambda *args: [], "a": lambda *args: []},
    )

    def fake(provider: str, *args: Any) -> tuple[list[Any], SourceStatus]:
        del args
        return [], SourceStatus(
            provider=provider, status="empty", result_count=0, duration_seconds=0
        )

    monkeypatch.setattr(multi_source, "_fetch_provider", fake)
    assert [item["provider"] for item in multi_source.check_sources()] == ["a", "z"]


def test_identifier_relevance_year_and_integer_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    assert multi_source.canonical_identifier({"pmid": "ABC"}) == "pmid:abc"
    assert multi_source.canonical_identifier({"title": "Only Title"}) == "title:only title"
    assert multi_source.simple_relevance_score("the and", "anything", "anything") == 0
    assert multi_source.normalize_year("bad") == 0.5
    monkeypatch.setattr(
        multi_source.dt, "datetime", type("D", (), {"now": lambda: type("N", (), {"year": 2025})()})
    )
    assert multi_source.normalize_year(2030) == 0
    assert multi_source.normalize_year(2000) == 0
    assert multi_source._as_int(-2) == 0
    assert multi_source._as_int("bad") == 0


def test_duplicate_merge_promotes_strong_identifier_and_metadata() -> None:
    merged = multi_source._merge_duplicates(
        [
            paper(provider_id=None, doi=None, title="Same Title", pmid=None),
            paper(provider_id=None, doi="10.1/strong", title="Same Title", pmid="123"),
            paper(provider_id=None, doi="10.1/strong", title="Same Title", pmcid="PMC1"),
        ]
    )
    assert len(merged) == 1
    assert merged[0]["paper_id"] == "doi:10.1/strong"
    assert merged[0]["pmid"] == "123"
    assert merged[0]["pmcid"] == "PMC1"


def test_rank_and_diversify_edge_paths() -> None:
    assert multi_source.rank_papers("topic", []) == []
    merged = multi_source._merge_duplicates(
        [
            paper(provider_id="1", title="Compression algorithm one", citations="bad"),
            paper(provider_id="2", title="Compression algorithm two", citations=0),
        ]
    )
    ranked = multi_source.rank_papers("compression algorithm", merged)
    assert len(ranked) == 2
    selected = multi_source.diversify_papers(ranked, 1)
    assert len(selected) == 1
