"""Literature source adapters, caching, deduplication, and ranking."""

from __future__ import annotations

import datetime as dt
import math
import re
import sqlite3
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree

import httpx

try:
    from .literature import ScholarlyClient, SourceCache, SourceSettings
    from .literature_types import SearchQuery
    from .schemas import SourceStatus
except ImportError:  # pragma: no cover - supports direct script execution
    from literature import ScholarlyClient, SourceCache, SourceSettings
    from literature_types import SearchQuery
    from schemas import SourceStatus

Paper = dict[str, Any]
Fetcher = Callable[[str, int, "ScholarlyClient"], list[Paper]]


def fetch_semanticscholar(topic: str, max_results: int, client: ScholarlyClient) -> list[Paper]:
    headers = {}
    if client.settings.semantic_scholar_api_key:
        headers["x-api-key"] = client.settings.semantic_scholar_api_key
    response = client.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": topic,
            "limit": max_results,
            "fields": "paperId,title,authors,year,abstract,url,citationCount,externalIds",
        },
        headers=headers,
    )
    papers = []
    for item in response.json().get("data", []):
        if not item.get("title") or not item.get("abstract"):
            continue
        external = item.get("externalIds") or {}
        papers.append(
            {
                "provider_id": item.get("paperId"),
                "doi": external.get("DOI"),
                "pmid": external.get("PubMed"),
                "arxiv_id": external.get("ArXiv"),
                "title": item["title"],
                "authors": ", ".join(a.get("name", "") for a in item.get("authors", [])),
                "year": item.get("year"),
                "summary": item["abstract"],
                "url": item.get("url"),
                "citations": item.get("citationCount", 0),
                "source": "Semantic Scholar",
            }
        )
    return papers


def fetch_europepmc(topic: str, max_results: int, client: ScholarlyClient) -> list[Paper]:
    response = client.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": topic, "format": "json", "pageSize": max_results, "resultType": "core"},
    )
    papers = []
    for item in response.json().get("resultList", {}).get("result", []):
        if not item.get("title") or not item.get("abstractText"):
            continue
        doi = item.get("doi")
        papers.append(
            {
                "provider_id": item.get("id"),
                "doi": doi,
                "pmid": item.get("pmid"),
                "pmcid": item.get("pmcid"),
                "title": item["title"],
                "authors": item.get("authorString", ""),
                "year": item.get("pubYear"),
                "summary": item["abstractText"],
                "url": f"https://doi.org/{doi}" if doi else item.get("fullTextUrl", ""),
                "citations": item.get("citedByCount", 0),
                "source": "Europe PMC",
            }
        )
    return papers


def _strip_markup(value: Any) -> str:
    text = " ".join(str(part) for part in value) if isinstance(value, list) else str(value or "")
    return re.sub(r"<[^>]*>", "", text).strip()


def fetch_crossref(topic: str, max_results: int, client: ScholarlyClient) -> list[Paper]:
    params: dict[str, Any] = {
        "query": topic,
        "rows": max_results,
        "filter": "has-abstract:true",
    }
    if client.settings.contact_email:
        params["mailto"] = client.settings.contact_email
    response = client.get("https://api.crossref.org/works", params=params)
    papers = []
    for item in response.json().get("message", {}).get("items", []):
        title = (item.get("title") or [None])[0]
        abstract = _strip_markup(item.get("abstract"))
        if not title or not abstract:
            continue
        dates = (
            item.get("published-print") or item.get("published-online") or item.get("created") or {}
        )
        parts = dates.get("date-parts") or [[None]]
        papers.append(
            {
                "doi": item.get("DOI"),
                "title": title,
                "authors": ", ".join(
                    f"{author.get('given', '')} {author.get('family', '')}".strip()
                    for author in item.get("author", [])
                ),
                "year": parts[0][0],
                "summary": abstract,
                "url": item.get("URL", ""),
                "citations": item.get("is-referenced-by-count", 0),
                "source": "Crossref",
            }
        )
    return papers


def fetch_doaj(topic: str, max_results: int, client: ScholarlyClient) -> list[Paper]:
    response = client.get(
        f"https://doaj.org/api/v2/search/articles/{quote(topic, safe='')}",
        params={"page": 1, "pageSize": max_results},
    )
    papers = []
    for item in response.json().get("results", []):
        bib = item.get("bibjson", {})
        if not bib.get("title") or not bib.get("abstract"):
            continue
        identifiers = {i.get("type"): i.get("id") for i in bib.get("identifier", [])}
        links = bib.get("link", [])
        papers.append(
            {
                "provider_id": item.get("id"),
                "doi": identifiers.get("doi"),
                "title": bib["title"],
                "authors": ", ".join(a.get("name", "") for a in bib.get("author", [])),
                "year": bib.get("year"),
                "summary": bib["abstract"],
                "url": next(
                    (link.get("url") for link in links if link.get("type") == "fulltext"),
                    links[0].get("url", "") if links else "",
                ),
                "citations": 0,
                "source": "DOAJ",
            }
        )
    return papers


def fetch_pubmed(topic: str, max_results: int, client: ScholarlyClient) -> list[Paper]:
    common: dict[str, Any] = {"tool": "abstract_agent"}
    if client.settings.contact_email:
        common["email"] = client.settings.contact_email
    if client.settings.ncbi_api_key:
        common["api_key"] = client.settings.ncbi_api_key
    search = client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"db": "pubmed", "retmax": max_results, "term": topic, "retmode": "json", **common},
    )
    ids = search.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    fetched = client.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", **common},
    )
    documents = ElementTree.fromstring(getattr(fetched, "content", fetched.text)).findall(
        ".//PubmedArticle"
    )
    papers = []
    for document in documents:
        citation = document.find("MedlineCitation")
        article = citation.find("Article") if citation is not None else None
        if citation is None or article is None:
            continue
        pmid = citation.findtext("PMID")
        title = (
            "".join(article.find("ArticleTitle").itertext()).strip()
            if article.find("ArticleTitle") is not None
            else ""
        )
        abstract = " ".join(
            "".join(part.itertext()).strip() for part in article.findall("./Abstract/AbstractText")
        )
        if not title or not abstract:
            continue
        authors = article.findall("./AuthorList/Author")
        pub_date = article.find("./Journal/JournalIssue/PubDate")
        year = ""
        if pub_date is not None:
            year = (
                pub_date.findtext("Year") or (pub_date.findtext("MedlineDate") or "").split(" ")[0]
            )
        papers.append(
            {
                "provider_id": pmid,
                "pmid": pmid,
                "title": title,
                "authors": ", ".join(
                    f"{author.findtext('LastName', '')} {author.findtext('Initials', '')}".strip()
                    for author in authors
                ),
                "year": year,
                "summary": abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "citations": 0,
                "source": "PubMed",
            }
        )
    return papers


def _xml_text(element: ElementTree.Element, name: str, default: str = "") -> str:
    child = element.find(name)
    if child is None:
        child = element.find(f"{{*}}{name}")
    return "".join(child.itertext()).strip() if child is not None else default


def fetch_openalex(topic: str, max_results: int, client: ScholarlyClient) -> list[Paper]:
    params: dict[str, Any] = {
        "search": topic,
        "per_page": max_results,
        "filter": "has_abstract:true",
        "select": (
            "id,doi,title,publication_year,abstract_inverted_index,authorships,cited_by_count,ids"
        ),
    }
    if client.settings.openalex_api_key:
        params["api_key"] = client.settings.openalex_api_key
    response = client.get("https://api.openalex.org/works", params=params)
    papers = []
    for item in response.json().get("results", []):
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
        if not item.get("title") or not abstract:
            continue
        ids = item.get("ids") or {}
        papers.append(
            {
                "provider_id": item.get("id"),
                "doi": item.get("doi"),
                "pmid": _last_path(ids.get("pmid")),
                "pmcid": _last_path(ids.get("pmcid")),
                "title": item["title"],
                "authors": ", ".join(
                    authorship.get("author", {}).get("display_name", "")
                    for authorship in item.get("authorships", [])
                ),
                "year": item.get("publication_year"),
                "summary": abstract,
                "url": item.get("doi") or item.get("id"),
                "citations": item.get("cited_by_count", 0),
                "source": "OpenAlex",
            }
        )
    return papers


MAX_ABSTRACT_POSITIONS = 20_000


def reconstruct_abstract(index: Any) -> str:
    if not isinstance(index, dict) or not index:
        return ""
    positioned: dict[int, str] = {}
    total = 0
    for word, indexes in index.items():
        if not isinstance(word, str) or not isinstance(indexes, list):
            return ""
        for position in indexes:
            total += 1
            if (
                total > MAX_ABSTRACT_POSITIONS
                or isinstance(position, bool)
                or not isinstance(position, int)
                or not 0 <= position < MAX_ABSTRACT_POSITIONS
            ):
                return ""
            existing = positioned.get(position)
            if existing is not None and existing != word:
                return ""
            positioned[position] = word
    return " ".join(positioned[position] for position in sorted(positioned))


def parse_arxiv_identifier(entry_id: str) -> str:
    """Return a versionless modern or legacy arXiv identifier."""
    parsed = urlsplit(entry_id)
    raw = parsed.path.split("/abs/", 1)[1] if "/abs/" in parsed.path else entry_id
    raw = unquote(raw).removeprefix("arXiv:").strip().strip("/")
    return re.sub(r"v\d+$", "", raw, flags=re.IGNORECASE)


def fetch_arxiv(topic: str, max_results: int, client: ScholarlyClient) -> list[Paper]:
    response = client.get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": f"all:{topic}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
    )
    root = ElementTree.fromstring(getattr(response, "content", response.text))
    entries = root.findall("entry") or root.findall("{*}entry")
    papers = []
    for item in entries:
        title = " ".join(_xml_text(item, "title").split())
        summary = " ".join(_xml_text(item, "summary").split())
        if not title or not summary:
            continue
        entry_id = _xml_text(item, "id")
        arxiv_id = parse_arxiv_identifier(entry_id)
        authors = item.findall("author") or item.findall("{*}author")
        published = _xml_text(item, "published")
        papers.append(
            {
                "provider_id": arxiv_id,
                "arxiv_id": arxiv_id,
                "doi": _xml_text(item, "doi") or None,
                "title": title,
                "authors": ", ".join(_xml_text(author, "name") for author in authors),
                "year": published[:4] or None,
                "summary": summary,
                "url": entry_id,
                "citations": 0,
                "source": "arXiv",
            }
        )
    return papers


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return _first_text(value[0]) if value else ""
    if isinstance(value, dict):
        for key in ("value", "name", "fullName", "#text"):
            if value.get(key):
                return str(value[key])
        return ""
    return str(value or "")


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def fetch_hal(topic: str, max_results: int, client: ScholarlyClient) -> list[Paper]:
    """Search the anonymous HAL scholarly archive API."""
    response = client.get(
        "https://api.archives-ouvertes.fr/search/",
        params={
            "q": topic,
            "wt": "json",
            "rows": max_results,
            "fl": (
                "docid,title_s,abstract_s,authFullName_s,producedDateY_i,"
                "doiId_s,uri_s,citationFull_s"
            ),
        },
    )
    papers = []
    for item in response.json().get("response", {}).get("docs", []):
        title = _first_text(item.get("title_s"))
        abstract = _first_text(item.get("abstract_s"))
        if not title or not abstract:
            continue
        authors = _listify(item.get("authFullName_s"))
        papers.append(
            {
                "provider_id": str(item.get("docid") or ""),
                "doi": _first_text(item.get("doiId_s")),
                "title": title,
                "authors": ", ".join(str(author) for author in authors),
                "year": item.get("producedDateY_i"),
                "summary": abstract,
                "url": _first_text(item.get("uri_s")),
                "citations": 0,
                "source": "HAL",
            }
        )
    return papers


def fetch_openaire(topic: str, max_results: int, client: ScholarlyClient) -> list[Paper]:
    """Search publication metadata through OpenAIRE Graph V3."""
    response = client.get(
        "https://api.openaire.eu/graph/v3/research-products",
        params={"search": topic, "type": "publication", "pageSize": max_results},
    )
    papers = []
    for item in response.json().get("results", []):
        title = _first_text(item.get("mainTitle") or item.get("title"))
        abstract = _first_text(
            item.get("descriptions") or item.get("description") or item.get("abstract")
        )
        if not title or not abstract:
            continue
        pids = _listify(item.get("pids"))
        doi = next(
            (
                _first_text(pid)
                for pid in pids
                if isinstance(pid, dict) and str(pid.get("scheme", "")).casefold() == "doi"
            ),
            "",
        )
        authors = _listify(item.get("authors") or item.get("creators"))
        papers.append(
            {
                "provider_id": _first_text(item.get("id")),
                "doi": doi,
                "title": title,
                "authors": ", ".join(_first_text(author) for author in authors),
                "year": str(item.get("publicationDate") or item.get("dateOfAcceptance") or "")[:4],
                "summary": abstract,
                "url": doi and f"https://doi.org/{doi}" or _first_text(item.get("id")),
                "citations": int(
                    item.get("citationCount")
                    or item.get("indicators", {}).get("citationImpact", {}).get("citationCount", 0)
                ),
                "source": "OpenAIRE",
            }
        )
    return papers


def fetch_core(topic: str, max_results: int, client: ScholarlyClient) -> list[Paper]:
    """Search CORE's anonymous open access works endpoint."""
    response = client.get(
        "https://api.core.ac.uk/v3/search/works",
        params={"q": topic, "limit": max_results},
        retry=False,
        timeout=(3.05, 15),
    )
    papers = []
    for item in response.json().get("results", []):
        title = _first_text(item.get("title"))
        abstract = _first_text(item.get("abstract"))
        if not title or not abstract:
            continue
        authors = _listify(item.get("authors"))
        papers.append(
            {
                "provider_id": str(item.get("id") or ""),
                "doi": _first_text(item.get("doi")),
                "title": title,
                "authors": ", ".join(_first_text(author) for author in authors),
                "year": item.get("yearPublished") or item.get("publishedDate"),
                "summary": abstract,
                "url": _first_text(item.get("downloadUrl") or item.get("sourceFulltextUrls")),
                "citations": int(item.get("citationCount") or 0),
                "source": "CORE",
            }
        )
    return papers


def _last_path(value: Any) -> str | None:
    return str(value).rstrip("/").rsplit("/", 1)[-1] if value else None


FETCHERS: dict[str, Fetcher] = {
    "arxiv": fetch_arxiv,
    "semantic_scholar": fetch_semanticscholar,
    "pubmed": fetch_pubmed,
    "openalex": fetch_openalex,
    "europe_pmc": fetch_europepmc,
    "crossref": fetch_crossref,
    "doaj": fetch_doaj,
    "hal": fetch_hal,
    "openaire": fetch_openaire,
    "core": fetch_core,
}

PROVIDER_INTERVAL_SECONDS = {
    "arxiv": 3.0,
    "semantic_scholar": 1.1,
    "core": 2.1,
    "crossref": 1.1,
    "pubmed": 0.34,
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[\w]+", text))


def canonical_identifier(paper: Paper) -> str:
    doi = paper.get("doi")
    if not doi and "doi.org/" in str(paper.get("url", "")).casefold():
        doi = str(paper["url"]).casefold().split("doi.org/", 1)[1]
    if doi:
        normalized = str(doi).casefold().removeprefix("https://doi.org/").removeprefix("doi:")
        normalized = normalized.split("?", 1)[0].split("#", 1)[0].strip().rstrip(".,;)")
        if normalized:
            return f"doi:{normalized}"
    for key in ("pmcid", "pmid", "arxiv_id"):
        if paper.get(key):
            return f"{key}:{str(paper[key]).casefold()}"
    if paper.get("provider_id") and (paper.get("provider") or paper.get("source")):
        provider = normalize_text(paper.get("provider") or paper["source"]).replace(" ", "_")
        return f"provider_id:{provider}:{str(paper['provider_id']).casefold()}"
    return f"title:{normalize_text(paper.get('title'))}"


_RELEVANCE_STOP_WORDS = {
    "a",
    "an",
    "and",
    "approach",
    "based",
    "for",
    "in",
    "method",
    "methods",
    "new",
    "novel",
    "of",
    "on",
    "research",
    "study",
    "the",
    "to",
    "using",
    "with",
}


def _term_key(token: str) -> str:
    if token.startswith("compress"):
        return "compress"
    if token.startswith("algorithm"):
        return "algorithm"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _meaningful_terms(value: str) -> set[str]:
    return {
        _term_key(token)
        for token in normalize_text(value).split()
        if len(token) > 2 and token not in _RELEVANCE_STOP_WORDS
    }


def simple_relevance_score(topic: str, title: str, summary: str) -> float:
    """Score topic anchor coverage while giving title matches extra weight."""
    topic_terms = _meaningful_terms(topic)
    if not topic_terms:
        return 0.0
    title_terms = _meaningful_terms(title)
    text_terms = title_terms | _meaningful_terms(summary)
    overlap = topic_terms & text_terms
    minimum_overlap = 1 if len(topic_terms) <= 2 else 2
    if len(overlap) < minimum_overlap:
        return 0.0
    coverage = len(overlap) / len(topic_terms)
    title_coverage = len(topic_terms & title_terms) / len(topic_terms)
    return min(1.0, 0.7 * coverage + 0.3 * title_coverage)


def normalize_year(year: Any) -> float:
    try:
        parsed = int(str(year)[:4])
    except (TypeError, ValueError):
        return 0.5
    current = dt.datetime.now().year
    if parsed > current + 1:
        return 0.0
    return max(0.0, 1.0 - min(max(0, current - parsed), 20) / 20)


def _merge_duplicates(papers: list[Paper]) -> list[Paper]:
    merged: dict[str, Paper] = {}
    title_aliases: dict[str, str] = {}
    provider_order = {provider: index for index, provider in enumerate(FETCHERS)}
    ordered_papers = sorted(
        papers,
        key=lambda paper: (
            provider_order.get(str(paper.get("provider")), len(provider_order)),
            canonical_identifier(paper),
            normalize_text(paper.get("title")),
        ),
    )
    for paper in ordered_papers:
        identifier = canonical_identifier(paper)
        normalized_title = normalize_text(paper.get("title"))
        title_key = title_aliases.get(normalized_title) if normalized_title else None
        identifier_is_strong = not identifier.startswith("title:")
        existing_key = identifier if identifier in merged else title_key or identifier
        if (
            existing_key in merged
            and title_key == existing_key
            and identifier_is_strong
            and not existing_key.startswith("title:")
            and existing_key != identifier
        ):
            existing_key = identifier
        if existing_key not in merged:
            paper = paper.copy()
            paper["paper_id"] = identifier
            paper["sources"] = [paper["source"]] if paper.get("source") else []
            for key in ("matched_queries", "matched_query_ids", "matched_candidate_ids"):
                paper[key] = list(dict.fromkeys(paper.get(key, [])))
            merged[identifier] = paper
            if normalized_title:
                title_aliases[normalized_title] = identifier
            continue
        if existing_key.startswith("title:") and identifier_is_strong:
            existing = merged.pop(existing_key)
            existing["paper_id"] = identifier
            merged[identifier] = existing
            existing_key = identifier
            if normalized_title:
                title_aliases[normalized_title] = identifier
        existing = merged[existing_key]
        source = paper.get("source")
        if source and source not in existing["sources"]:
            existing["sources"].append(source)
        existing["matched_queries"] = list(
            dict.fromkeys(existing.get("matched_queries", []) + paper.get("matched_queries", []))
        )
        existing["matched_query_ids"] = list(
            dict.fromkeys(
                existing.get("matched_query_ids", []) + paper.get("matched_query_ids", [])
            )
        )
        existing["matched_candidate_ids"] = list(
            dict.fromkeys(
                existing.get("matched_candidate_ids", []) + paper.get("matched_candidate_ids", [])
            )
        )
        existing["citations"] = max(
            _as_int(existing.get("citations")), _as_int(paper.get("citations"))
        )
        for key in ("doi", "pmid", "pmcid", "arxiv_id"):
            if not existing.get(key) and paper.get(key):
                existing[key] = paper[key]
    return list(merged.values())


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def rank_papers(
    topic: str | list[SearchQuery],
    papers: list[Paper],
    *,
    relevance_weight: float = 0.5,
    recency_weight: float = 0.3,
    citation_weight: float = 0.2,
    minimum_relevance: float = 0.3,
) -> list[Paper]:
    relevant = []
    maximum_log_citations = max(
        (math.log1p(_as_int(p.get("citations"))) for p in papers), default=0
    )
    queries = (
        [SearchQuery(query_id="query:0", text=topic)] if isinstance(topic, str) else list(topic)
    )
    for paper in papers:
        query_scores = [
            (
                query,
                simple_relevance_score(
                    query.text, paper.get("title", ""), paper.get("summary", "")
                ),
            )
            for query in queries
        ]
        best_query, relevance = max(query_scores, key=lambda item: (item[1], item[0].query_id))
        if relevance < minimum_relevance or len(str(paper.get("summary", ""))) <= 50:
            continue
        recency = normalize_year(paper.get("year"))
        citation = (
            math.log1p(_as_int(paper.get("citations"))) / maximum_log_citations
            if maximum_log_citations
            else 0.0
        )
        composite = (
            relevance_weight * relevance + recency_weight * recency + citation_weight * citation
        )
        paper.update(
            {
                "citations": _as_int(paper.get("citations")),
                "relevance_score": round(relevance, 3),
                "recency_score": round(recency, 3),
                "citation_score": round(citation, 3),
                "composite_score": round(composite, 3),
                "best_matching_query_id": best_query.query_id,
            }
        )
        relevant.append(paper)
    return sorted(
        relevant,
        key=lambda item: (
            -item["composite_score"],
            -int(str(item.get("year") or "0")[:4])
            if str(item.get("year") or "")[:4].isdigit()
            else 0,
            -item["citations"],
            item["paper_id"],
        ),
    )


def diversify_papers(papers: list[Paper], max_results: int) -> list[Paper]:
    """Use a small MMR style penalty and coverage bonuses to avoid redundant shortlists."""
    selected: list[Paper] = []
    remaining = list(papers)
    covered_queries: set[str] = set()
    covered_sources: set[str] = set()
    while remaining and len(selected) < max_results:
        best: Paper | None = None
        best_score = -math.inf
        for paper in remaining:
            words = set(normalize_text(f"{paper.get('title')} {paper.get('summary')}").split())
            redundancy = 0.0
            for chosen in selected:
                chosen_words = set(
                    normalize_text(f"{chosen.get('title')} {chosen.get('summary')}").split()
                )
                union = words | chosen_words
                redundancy = max(
                    redundancy, len(words & chosen_words) / len(union) if union else 0.0
                )
            queries = set(paper.get("matched_queries", []))
            sources = set(paper.get("sources", []))
            coverage_bonus = 0.04 * len(queries - covered_queries) + 0.02 * bool(
                sources - covered_sources
            )
            score = float(paper.get("composite_score", 0)) - 0.15 * redundancy + coverage_bonus
            if score > best_score:
                best, best_score = paper, score
        if best is None:
            raise RuntimeError("paper diversification could not select a candidate")
        selected.append(best)
        covered_queries.update(best.get("matched_queries", []))
        covered_sources.update(best.get("sources", []))
        remaining.remove(best)
    return selected


def _fetch_provider(
    provider: str,
    fetcher: Fetcher,
    topic: str,
    max_results: int,
    cache: SourceCache | None,
) -> tuple[list[Paper], SourceStatus]:
    started = time.monotonic()
    cache_warnings: list[str] = []
    try:
        cached = cache.get(provider, topic, max_results) if cache else None
    except (OSError, sqlite3.Error):
        cached = None
        cache_warnings.append("cache_read_error")
    if cached and cached[1]:
        papers = cached[0]
        return papers, SourceStatus(
            provider=provider,
            status="cached",
            result_count=len(papers),
            duration_seconds=round(time.monotonic() - started, 3),
        )
    client = ScholarlyClient()
    try:
        papers = fetcher(topic, max_results, client)
        for paper in papers:
            paper.setdefault("provider", provider)
        if cache:
            try:
                cache.put(provider, topic, max_results, papers)
            except (OSError, sqlite3.Error):
                cache_warnings.append("cache_write_error")
        retry_events = getattr(client, "retry_events", [])
        retry_categories = sorted(
            {str(event.get("category")) for event in retry_events if event.get("category")}
        )
        if retry_events:
            cache_warnings.append(
                "rate_limit_retry" if "http_429" in retry_categories else "provider_retry"
            )
        return papers, SourceStatus(
            provider=provider,
            status="partial" if cache_warnings else ("ok" if papers else "empty"),
            result_count=len(papers),
            duration_seconds=round(time.monotonic() - started, 3),
            rate_limit=client.rate_limit,
            warning_categories=cache_warnings,
            warning_query_count=1 if cache_warnings else 0,
            message=", ".join(cache_warnings) or None,
            retry_count=len(retry_events),
            retry_categories=retry_categories,
        )
    except httpx.TimeoutException as error:
        category = "timeout"
        caught: Exception = error
    except httpx.HTTPStatusError as error:
        category = f"http_{error.response.status_code}"
        caught = error
    except httpx.NetworkError as error:
        category = "network_error"
        caught = error
    except Exception as error:  # provider parsers and arxiv expose varied exception types
        category = "provider_error"
        caught = error
    finally:
        client.close()
    if cached:
        papers = cached[0]
        return papers, SourceStatus(
            provider=provider,
            status="cached",
            result_count=len(papers),
            duration_seconds=round(time.monotonic() - started, 3),
            error_category=category,
            message=f"stale cache used after {type(caught).__name__}",
            rate_limit=client.rate_limit,
            retry_count=len(getattr(client, "retry_events", [])),
            retry_categories=sorted(
                {
                    str(event.get("category"))
                    for event in getattr(client, "retry_events", [])
                    if event.get("category")
                }
            ),
        )
    return [], SourceStatus(
        provider=provider,
        status="error",
        result_count=0,
        duration_seconds=round(time.monotonic() - started, 3),
        error_category=category,
        message=_sanitize_error(str(caught), client.settings),
        rate_limit=client.rate_limit,
        retry_count=len(getattr(client, "retry_events", [])),
        retry_categories=sorted(
            {
                str(event.get("category"))
                for event in getattr(client, "retry_events", [])
                if event.get("category")
            }
        ),
    )


def _fetch_provider_queries(
    provider: str,
    fetcher: Fetcher,
    topics: list[str],
    max_results: int,
    cache: SourceCache | None,
) -> list[tuple[str, list[Paper], SourceStatus]]:
    """Run one provider's queries serially so facets cannot rate limit each other."""
    results = []
    interval = PROVIDER_INTERVAL_SECONDS.get(provider, 0.0)
    previous_used_network = False
    consecutive_transient_failures = 0
    for index, query in enumerate(topics):
        if index and interval and previous_used_network:
            time.sleep(interval)
        papers, status = _fetch_provider(provider, fetcher, query, max_results, cache)
        permanent_failure = status.error_category in {
            "http_401",
            "http_403",
            "http_429",
            "provider_error",
        }
        transient_failure = status.status == "error" and not permanent_failure
        consecutive_transient_failures = (
            consecutive_transient_failures + 1 if transient_failure else 0
        )
        open_circuit = permanent_failure or consecutive_transient_failures >= 2
        if open_circuit:
            status = status.model_copy(
                update={
                    "circuit_state": "open",
                    "circuit_reason": status.error_category or status.status,
                }
            )
        results.append((query, papers, status))
        previous_used_network = status.status != "cached"
        if open_circuit:
            for skipped_query in topics[index + 1 :]:
                results.append(
                    (
                        skipped_query,
                        [],
                        SourceStatus(
                            provider=provider,
                            status="skipped",
                            result_count=0,
                            duration_seconds=0,
                            error_category="circuit_open",
                            message="query skipped after an earlier provider failure",
                            query_count=0,
                            planned_query_count=1,
                            attempted_query_count=0,
                            skipped_query_count=1,
                            circuit_state="open",
                            circuit_reason=status.error_category or status.status,
                        ),
                    )
                )
            break
    return results


def _consolidate_statuses(statuses: list[SourceStatus]) -> list[SourceStatus]:
    grouped: dict[str, list[SourceStatus]] = {}
    for status in statuses:
        grouped.setdefault(status.provider, []).append(status)
    consolidated = []
    for provider, items in grouped.items():
        hard_failures = [item for item in items if item.status == "error"]
        failures = [
            item
            for item in items
            if item.status != "skipped" and (item.status == "error" or item.error_category)
        ]
        cached = [item for item in items if item.status == "cached"]
        skipped = [item for item in items if item.status == "skipped"]
        warnings = [item for item in items if item.warning_categories]
        succeeded = [item for item in items if item.status in {"ok", "empty", "cached", "partial"}]
        if len(skipped) == len(items):
            status_value = "skipped"
        elif hard_failures and not succeeded:
            status_value = "error"
        elif len(cached) == len(items):
            status_value = "cached"
        elif (failures or warnings) and any(item.result_count for item in items):
            status_value = "partial"
        elif any(item.status == "ok" for item in items):
            status_value = "ok"
        else:
            status_value = "empty"
        categories = sorted({item.error_category for item in failures if item.error_category})
        warning_categories = sorted(
            {category for item in warnings for category in item.warning_categories}
        )
        message = None
        error_category = categories[0] if len(categories) == 1 else None
        if skipped and len(skipped) == len(items):
            error_category = "circuit_open"
            message = (
                skipped[0].message or "provider skipped because its run scoped circuit is open"
            )
        elif failures:
            planned = sum(item.planned_query_count for item in items)
            skipped_count = sum(item.skipped_query_count for item in items)
            total_label = "planned" if skipped_count else "attempted"
            message = (
                f"{len(failures)}/{planned} {total_label} queries failed"
                + (f"; {skipped_count} skipped" if skipped_count else "")
                + (f" ({', '.join(categories)})" if categories else "")
            )
        if warnings:
            warning_message = (
                f"{len(warnings)}/{len(items)} queries had warnings"
                f" ({', '.join(warning_categories)})"
            )
            message = f"{message}; {warning_message}" if message else warning_message
        rate_limit = {}
        for item in items:
            rate_limit.update(item.rate_limit)
        retry_categories = sorted(
            {category for item in items for category in item.retry_categories}
        )
        consolidated.append(
            SourceStatus(
                provider=provider,
                status=status_value,
                result_count=sum(item.result_count for item in items),
                duration_seconds=round(sum(item.duration_seconds for item in items), 3),
                error_category=error_category,
                message=message,
                rate_limit=rate_limit,
                # ``query_count`` is the backwards-compatible attempted-query
                # counter.  The explicit planned/skipped fields carry the
                # scheduling totals without making a skipped request look like
                # a network attempt.
                query_count=sum(item.attempted_query_count for item in items),
                planned_query_count=sum(item.planned_query_count for item in items),
                attempted_query_count=sum(item.attempted_query_count for item in items),
                succeeded_query_count=sum(
                    item.succeeded_query_count
                    or int(
                        item.status in {"ok", "empty", "partial"}
                        or (item.status == "cached" and not item.error_category)
                    )
                    for item in items
                ),
                skipped_query_count=sum(item.skipped_query_count for item in items),
                circuit_state="open"
                if any(item.circuit_state == "open" for item in items)
                else "closed",
                circuit_reason=next(
                    (item.circuit_reason for item in items if item.circuit_reason), None
                ),
                failed_query_count=sum(
                    item.failed_query_count
                    or int(
                        item.status != "skipped"
                        and (item.status == "error" or bool(item.error_category))
                    )
                    for item in items
                ),
                cached_query_count=len(cached),
                warning_categories=warning_categories,
                warning_query_count=len(warnings),
                retry_count=sum(item.retry_count for item in items),
                retry_categories=retry_categories,
            )
        )
    return sorted(consolidated, key=lambda item: item.provider)


def _normalize_search_queries(
    topic: str | list[str] | list[dict[str, Any]] | list[SearchQuery],
) -> list[SearchQuery]:
    raw_items: list[Any] = [topic] if isinstance(topic, str) else list(topic)
    queries = []
    for index, item in enumerate(raw_items):
        if isinstance(item, SearchQuery):
            query = item
        elif isinstance(item, str):
            query = SearchQuery(query_id=f"query:{index}", text=item.strip())
        else:
            query = SearchQuery.model_validate(item)
        queries.append(query)
    if not queries:
        raise ValueError("at least one non empty literature query is required")
    return queries


def aggregate_sources(
    topic: str | list[str] | list[dict[str, Any]] | list[SearchQuery],
    max_results: int = 5,
    *,
    cache_path: str | Path | None = None,
    cache_ttl_seconds: int = 86_400,
    use_cache: bool = True,
    max_workers: int = 4,
    relevance_weight: float = 0.5,
    recency_weight: float = 0.3,
    citation_weight: float = 0.2,
    minimum_relevance: float = 0.3,
    knowledge_store: Any | None = None,
    offline: bool = False,
    disabled_providers: list[str] | set[str] | None = None,
    external_expansion_limit: int = 0,
) -> tuple[list[Paper], int, list[dict[str, Any]]]:
    """Fetch, deduplicate, score, and rank literature from all configured sources."""
    weights = (relevance_weight, recency_weight, citation_weight)
    if not 1 <= max_results <= 100:
        raise ValueError("max_results must be between 1 and 100")
    if not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be between 1 and 32")
    if cache_ttl_seconds < 0:
        raise ValueError("cache_ttl_seconds must be non negative")
    if not 0 <= minimum_relevance <= 1:
        raise ValueError("minimum_relevance must be between 0 and 1")
    if not 0 <= external_expansion_limit <= 2:
        raise ValueError("external_expansion_limit must be between 0 and 2")
    if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("ranking weights must be non negative and sum to 1.0")
    disabled = set(disabled_providers or [])
    unknown_disabled = disabled - set(FETCHERS)
    if unknown_disabled:
        raise ValueError(f"unknown disabled providers: {sorted(unknown_disabled)}")
    queries = _normalize_search_queries(topic)
    base_queries = list(queries)
    knowledge_queries: list[SearchQuery] = []
    if knowledge_store is not None and not offline:
        with suppress(OSError, ValueError):
            for query in base_queries:
                for index, related in enumerate(
                    knowledge_store.related_idea_queries(query.text, limit=1), start=1
                ):
                    knowledge_queries.append(
                        SearchQuery(
                            query_id=f"knowledge:{query.query_id}:{index}",
                            text=related,
                            phase=query.phase,
                            candidate_id=query.candidate_id,
                        )
                    )
    queries.extend(knowledge_queries)
    remote_queries = [*base_queries, *knowledge_queries[:external_expansion_limit]]
    topics = list(dict.fromkeys(query.text for query in remote_queries))
    query_metadata: dict[str, list[SearchQuery]] = {}
    for query in queries:
        query_metadata.setdefault(query.text, []).append(query)
    cache = SourceCache(cache_path, cache_ttl_seconds) if use_cache else None
    fetched: list[Paper] = []
    statuses: list[SourceStatus] = []
    recalled_count = 0
    if knowledge_store is not None:
        recalled: dict[str, Paper] = {}
        try:
            for query in queries:
                for stored in knowledge_store.search_papers(query.text, limit=max_results):
                    paper = dict(stored)
                    paper.pop("knowledge", None)
                    paper.setdefault("matched_queries", []).append(query.text)
                    paper.setdefault("matched_query_ids", []).append(query.query_id)
                    if query.candidate_id:
                        paper.setdefault("matched_candidate_ids", []).append(query.candidate_id)
                    paper_id = str(paper.get("paper_id") or "")
                    if paper_id:
                        recalled[paper_id] = paper
            fetched.extend(recalled.values())
            recalled_count = len(recalled)
            if recalled:
                statuses.append(
                    SourceStatus(
                        provider="local_knowledge",
                        status="cached",
                        result_count=len(recalled),
                        duration_seconds=0,
                        query_count=len(queries),
                        planned_query_count=len(queries),
                        attempted_query_count=len(queries),
                        succeeded_query_count=len(queries),
                        cached_query_count=len(queries),
                    )
                )
        except (OSError, ValueError):
            pass
    if offline:
        if not statuses:
            statuses.append(
                SourceStatus(
                    provider="local_knowledge",
                    status="empty",
                    result_count=0,
                    duration_seconds=0,
                    query_count=len(queries),
                    planned_query_count=len(queries),
                    attempted_query_count=len(queries),
                    succeeded_query_count=len(queries),
                )
            )
        ranked = rank_papers(
            queries,
            _merge_duplicates(fetched),
            relevance_weight=relevance_weight,
            recency_weight=recency_weight,
            citation_weight=citation_weight,
            minimum_relevance=minimum_relevance,
        )
        return (
            diversify_papers(ranked, max_results),
            recalled_count,
            [status.model_dump() for status in _consolidate_statuses(statuses)],
        )
    per_source_limit = max(5, max_results // len(topics) + 3)
    enabled_fetchers = {
        provider: fetcher for provider, fetcher in FETCHERS.items() if provider not in disabled
    }
    for provider in sorted(disabled):
        statuses.append(
            SourceStatus(
                provider=provider,
                status="skipped",
                result_count=0,
                duration_seconds=0,
                error_category="circuit_open",
                message="provider skipped after a hard failure earlier in this run",
                query_count=len(topics),
                planned_query_count=len(topics),
                attempted_query_count=0,
                skipped_query_count=len(topics),
                circuit_state="open",
                circuit_reason="prior_run_stage_failure",
            )
        )
    with ThreadPoolExecutor(
        max_workers=max(1, min(max_workers, len(enabled_fetchers), 8))
    ) as executor:
        futures = {
            executor.submit(
                _fetch_provider_queries,
                provider,
                fetcher,
                topics,
                per_source_limit,
                cache,
            ): provider
            for provider, fetcher in enabled_fetchers.items()
        }
        completed: dict[str, list[tuple[str, list[Paper], SourceStatus]]] = {}
        for future in as_completed(futures):
            provider = futures[future]
            try:
                completed[provider] = future.result()
            except Exception as error:
                completed[provider] = [
                    (
                        query,
                        [],
                        SourceStatus(
                            provider=provider,
                            status="error",
                            result_count=0,
                            duration_seconds=0,
                            error_category="provider_worker_error",
                            message=type(error).__name__,
                        ),
                    )
                    for query in topics
                ]
        for provider in enabled_fetchers:
            provider_results = completed[provider]
            for query, papers, status in provider_results:
                metadata = query_metadata[query]
                for paper in papers:
                    paper.setdefault("provider", provider)
                    paper.setdefault("matched_queries", []).append(query)
                    paper.setdefault("matched_query_ids", []).extend(
                        item.query_id for item in metadata
                    )
                    paper.setdefault("matched_candidate_ids", []).extend(
                        item.candidate_id for item in metadata if item.candidate_id
                    )
                fetched.extend(papers)
                statuses.append(status)
    unique = _merge_duplicates(fetched)
    ranked = rank_papers(
        queries,
        unique,
        relevance_weight=relevance_weight,
        recency_weight=recency_weight,
        citation_weight=citation_weight,
        minimum_relevance=minimum_relevance,
    )
    ranked = diversify_papers(ranked, max_results)
    if knowledge_store is not None:
        with suppress(OSError, ValueError):
            knowledge_store.ingest_papers(ranked)
    return ranked, len(fetched), [status.model_dump() for status in _consolidate_statuses(statuses)]


def check_sources(topic: str = "machine learning") -> list[dict[str, Any]]:
    """Perform a minimal uncached health check against each source."""
    statuses = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_fetch_provider, provider, fetcher, topic, 1, None): provider
            for provider, fetcher in FETCHERS.items()
        }
        for future in as_completed(futures):
            _, status = future.result()
            statuses.append(status.model_dump())
    return sorted(statuses, key=lambda item: item["provider"])


def _sanitize_error(message: str, settings: SourceSettings) -> str:
    sanitized = message
    for secret in (
        settings.semantic_scholar_api_key,
        settings.openalex_api_key,
        settings.ncbi_api_key,
    ):
        if secret:
            sanitized = sanitized.replace(secret, "[REDACTED]")
    return sanitized
