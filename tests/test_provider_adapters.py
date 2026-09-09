from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent import multi_source
from agent.literature.client import _retry_delay


def response(
    status: int = 200,
    *,
    payload: Any = None,
    text: str | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    request = httpx.Request("GET", "https://provider.test/resource")
    if text is not None:
        return httpx.Response(status, text=text, headers=headers, request=request)
    return httpx.Response(
        status, json=payload if payload is not None else {}, headers=headers, request=request
    )


class ScriptedSession:
    def __init__(self, events: list[httpx.Response | Exception]) -> None:
        self.events = list(events)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append((url, kwargs))
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event

    def close(self) -> None:
        self.closed = True


class AdapterClient:
    def __init__(
        self,
        payloads: list[httpx.Response],
        settings: multi_source.SourceSettings | None = None,
    ) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.settings = settings or multi_source.SourceSettings(None, None, None, None)

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append((url, kwargs))
        return self.payloads.pop(0)


def test_source_settings_loads_optional_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ABSTRACT_AGENT_CONTACT_EMAIL", "person@example.test")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "semantic")
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex")
    monkeypatch.setenv("NCBI_API_KEY", "ncbi")

    settings = multi_source.SourceSettings.from_env()

    assert settings.contact_email == "person@example.test"
    assert settings.semantic_scholar_api_key == "semantic"
    assert settings.openalex_api_key == "openalex"
    assert settings.ncbi_api_key == "ncbi"


def test_scholarly_client_retries_server_response_and_records_rate_limit() -> None:
    session = ScriptedSession(
        [
            response(503, headers={"retry-after": "2", "x-ratelimit-remaining": "0"}),
            response(200, payload={"ok": True}, headers={"x-rate-limit-limit": "10"}),
        ]
    )
    delays: list[float] = []
    client = multi_source.ScholarlyClient(session=session, sleep=delays.append)  # type: ignore[arg-type]

    result = client.get("https://provider.test", params={"q": "safe"})
    client.close()

    assert result.json() == {"ok": True}
    assert delays == [2.0]
    assert client.rate_limit == {
        "retry-after": "2",
        "x-ratelimit-remaining": "0",
        "x-rate-limit-limit": "10",
    }
    assert client.retry_events == [{"category": "http_503", "delay_seconds": 2.0}]
    assert session.closed


def test_scholarly_client_invalid_retry_after_uses_backoff() -> None:
    session = ScriptedSession([response(429, headers={"retry-after": "tomorrow"}), response(200)])
    delays: list[float] = []
    client = multi_source.ScholarlyClient(session=session, sleep=delays.append)  # type: ignore[arg-type]

    client.get("https://provider.test")

    assert delays == [0.5]
    assert client.retry_events == [{"category": "http_429", "delay_seconds": 0.5}]


def test_retry_delay_accepts_provider_reset_formats_and_bounds_waits() -> None:
    assert _retry_delay("1.25", 0.5) == 1.25
    assert _retry_delay("-2", 0.5) == 0
    assert _retry_delay("999", 0.5) == 30
    assert _retry_delay("not-a-reset", 0.75) == 0.75
    future = datetime.now(UTC) + timedelta(seconds=5)
    assert 3 <= _retry_delay(format_datetime(future), 0.5) <= 5
    assert 3 <= _retry_delay(future.isoformat(), 0.5) <= 5
    assert 3 <= _retry_delay(future.replace(tzinfo=None).isoformat(), 0.5) <= 5


def test_scholarly_client_does_not_retry_client_error_or_disabled_retry() -> None:
    first = ScriptedSession([response(400)])
    client = multi_source.ScholarlyClient(session=first, sleep=lambda _: None)  # type: ignore[arg-type]
    with pytest.raises(httpx.HTTPStatusError):
        client.get("https://provider.test")
    assert len(first.calls) == 1

    timeout = httpx.ReadTimeout("slow", request=httpx.Request("GET", "https://provider.test"))
    second = ScriptedSession([timeout])
    client = multi_source.ScholarlyClient(session=second, sleep=lambda _: None)  # type: ignore[arg-type]
    with pytest.raises(httpx.ReadTimeout):
        client.get("https://provider.test", retry=False)
    assert len(second.calls) == 1


def test_scholarly_client_exhausts_transient_network_retries() -> None:
    events = [
        httpx.ConnectError("offline", request=httpx.Request("GET", "https://provider.test"))
        for _ in range(4)
    ]
    session = ScriptedSession(events)
    delays: list[float] = []
    client = multi_source.ScholarlyClient(session=session, sleep=delays.append)  # type: ignore[arg-type]

    with pytest.raises(httpx.ConnectError):
        client.get("https://provider.test")

    assert delays == [0.5, 1.0, 2.0]


def test_source_cache_round_trip_stale_and_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "sources.sqlite3"
    cache = multi_source.SourceCache(path, ttl_seconds=5)
    assert cache.get("provider", "Topic", 2) is None
    cache.put("provider", " Topic  ", 2, [{"title": "paper"}])
    assert cache.get("provider", "topic", 2) == ([{"title": "paper"}], True)

    key = cache.query_key("topic", 2)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "UPDATE source_cache SET fetched_at=? WHERE provider=? AND query_key=?",
            (time.time() - 30, "provider", key),
        )
        connection.commit()
    assert cache.get("provider", "topic", 2) == ([{"title": "paper"}], False)

    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "UPDATE source_cache SET payload=? WHERE provider=? AND query_key=?",
            ("not json", "provider", key),
        )
        connection.commit()
    assert cache.get("provider", "topic", 2) is None

    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "UPDATE source_cache SET payload=? WHERE provider=? AND query_key=?",
            (json.dumps({"not": "a list"}), "provider", key),
        )
        connection.commit()
    assert cache.get("provider", "topic", 2) is None


def test_source_cache_validation_and_key_normalization(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non negative"):
        multi_source.SourceCache(tmp_path / "cache.sqlite3", ttl_seconds=-1)
    assert multi_source.SourceCache.query_key("  Mixed   CASE ", 3) == (
        multi_source.SourceCache.query_key("mixed case", 3)
    )


def test_semantic_scholar_and_europe_pmc_normalize_and_filter() -> None:
    settings = multi_source.SourceSettings(None, "secret", None, None)
    semantic = AdapterClient(
        [
            response(
                payload={
                    "data": [
                        {"title": "missing abstract"},
                        {
                            "paperId": "s1",
                            "title": "Semantic paper",
                            "abstract": "Useful abstract",
                            "authors": [{"name": "A"}, {"name": "B"}],
                            "externalIds": {"DOI": "10.1/s", "PubMed": "1", "ArXiv": "2"},
                            "citationCount": 7,
                            "year": 2024,
                        },
                    ]
                }
            )
        ],
        settings,
    )
    papers = multi_source.fetch_semanticscholar("topic", 2, semantic)  # type: ignore[arg-type]
    assert papers[0]["authors"] == "A, B"
    assert papers[0]["doi"] == "10.1/s"
    assert semantic.calls[0][1]["headers"] == {"x-api-key": "secret"}

    europe = AdapterClient(
        [
            response(
                payload={
                    "resultList": {
                        "result": [
                            {"title": "skip"},
                            {
                                "id": "e1",
                                "title": "PMC paper",
                                "abstractText": "Full abstract",
                                "authorString": "C Author",
                                "fullTextUrl": "https://full.test",
                            },
                            {
                                "id": "e2",
                                "title": "DOI paper",
                                "abstractText": "Full abstract",
                                "doi": "10.1/e",
                            },
                        ]
                    }
                }
            )
        ]
    )
    papers = multi_source.fetch_europepmc("topic", 3, europe)  # type: ignore[arg-type]
    assert [paper["url"] for paper in papers] == ["https://full.test", "https://doi.org/10.1/e"]


def test_crossref_and_doaj_normalize_fallbacks() -> None:
    settings = multi_source.SourceSettings("person@example.test", None, None, None)
    crossref = AdapterClient(
        [
            response(
                payload={
                    "message": {
                        "items": [
                            {"title": []},
                            {
                                "title": ["Crossref paper"],
                                "abstract": ["<jats:p>First</jats:p>", "second"],
                                "published-online": {"date-parts": [[2023, 1, 1]]},
                                "author": [{"given": "Ada", "family": "Lovelace"}],
                                "DOI": "10.1/crossref",
                            },
                        ]
                    }
                }
            )
        ],
        settings,
    )
    papers = multi_source.fetch_crossref("topic", 2, crossref)  # type: ignore[arg-type]
    assert papers[0]["summary"] == "First second"
    assert papers[0]["year"] == 2023
    assert crossref.calls[0][1]["params"]["mailto"] == "person@example.test"

    doaj = AdapterClient(
        [
            response(
                payload={
                    "results": [
                        {"bibjson": {"title": "skip"}},
                        {
                            "id": "d1",
                            "bibjson": {
                                "title": "DOAJ paper",
                                "abstract": "DOAJ abstract",
                                "identifier": [{"type": "doi", "id": "10.1/doaj"}],
                                "author": [{"name": "Author"}],
                                "link": [
                                    {"type": "landing", "url": "https://landing.test"},
                                    {"type": "fulltext", "url": "https://full.test"},
                                ],
                            },
                        },
                    ]
                }
            )
        ]
    )
    papers = multi_source.fetch_doaj("a/b", 2, doaj)  # type: ignore[arg-type]
    assert papers[0]["url"] == "https://full.test"
    assert "a%2Fb" in doaj.calls[0][0]


def test_pubmed_handles_empty_search_and_parses_article_xml() -> None:
    empty = AdapterClient([response(payload={"esearchresult": {"idlist": []}})])
    assert multi_source.fetch_pubmed("topic", 2, empty) == []  # type: ignore[arg-type]

    xml = """
    <PubmedArticleSet>
      <PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
        <ArticleTitle>A <i>nested</i> title</ArticleTitle>
        <Abstract><AbstractText>First.</AbstractText><AbstractText>Second.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Lovelace</LastName><Initials>A</Initials></Author></AuthorList>
        <Journal><JournalIssue><PubDate>
          <MedlineDate>2022 Winter</MedlineDate>
        </PubDate></JournalIssue></Journal>
      </Article></MedlineCitation></PubmedArticle>
      <PubmedArticle><MedlineCitation /></PubmedArticle>
    </PubmedArticleSet>
    """
    settings = multi_source.SourceSettings("person@example.test", None, None, "key")
    client = AdapterClient(
        [response(payload={"esearchresult": {"idlist": ["123"]}}), response(text=xml)],
        settings,
    )
    papers = multi_source.fetch_pubmed("topic", 2, client)  # type: ignore[arg-type]
    assert papers[0]["title"] == "A nested title"
    assert papers[0]["summary"] == "First. Second."
    assert papers[0]["authors"] == "Lovelace A"
    assert papers[0]["year"] == "2022"
    assert client.calls[0][1]["params"]["api_key"] == "key"


def test_openalex_and_arxiv_filter_incomplete_entries() -> None:
    settings = multi_source.SourceSettings(None, None, "open-key", None)
    openalex = AdapterClient(
        [
            response(
                payload={
                    "results": [
                        {"title": "skip", "abstract_inverted_index": None},
                        {
                            "id": "https://openalex.org/W1",
                            "doi": "https://doi.org/10.1/open",
                            "title": "OpenAlex paper",
                            "abstract_inverted_index": {"Complete": [0], "abstract": [1]},
                            "authorships": [{"author": {"display_name": "Author"}}],
                            "ids": {
                                "pmid": "https://pubmed.test/123/",
                                "pmcid": "https://pmc.test/PMC1",
                            },
                        },
                    ]
                }
            )
        ],
        settings,
    )
    papers = multi_source.fetch_openalex("topic", 2, openalex)  # type: ignore[arg-type]
    assert papers[0]["summary"] == "Complete abstract"
    assert papers[0]["pmid"] == "123"
    assert openalex.calls[0][1]["params"]["api_key"] == "open-key"

    feed = """
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>skip</title></entry>
      <entry><id>arXiv:hep-th/9901001v3</id><title> Arxiv  paper </title>
      <summary> Complete   summary </summary><published></published>
      <author><name>Author</name></author></entry>
    </feed>
    """
    arxiv = AdapterClient([response(text=feed)])
    papers = multi_source.fetch_arxiv("topic", 2, arxiv)  # type: ignore[arg-type]
    assert papers[0]["arxiv_id"] == "hep-th/9901001"
    assert papers[0]["year"] is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([], ""),
        ([{"value": "first"}], "first"),
        ({"name": "Ada"}, "Ada"),
        ({"unknown": "value"}, ""),
        (None, ""),
        (42, "42"),
    ],
)
def test_first_text_variants(value: Any, expected: str) -> None:
    assert multi_source._first_text(value) == expected


def test_small_normalization_helpers_cover_fallbacks() -> None:
    assert multi_source._listify(None) == []
    assert multi_source._listify("one") == ["one"]
    assert multi_source._listify(["one"]) == ["one"]
    assert multi_source._last_path(None) is None
    assert multi_source._last_path("https://example.test/path/") == "path"
    assert multi_source._strip_markup(None) == ""
    assert multi_source.parse_arxiv_identifier("arXiv:1234.5678v2") == "1234.5678"


def test_reconstruct_abstract_rejects_malformed_and_conflicting_indexes() -> None:
    assert multi_source.reconstruct_abstract([]) == ""
    assert multi_source.reconstruct_abstract({1: [0]}) == ""
    assert multi_source.reconstruct_abstract({"word": "zero"}) == ""
    assert multi_source.reconstruct_abstract({"one": [0], "two": [0]}) == ""
    assert multi_source.reconstruct_abstract({"negative": [-1]}) == ""
