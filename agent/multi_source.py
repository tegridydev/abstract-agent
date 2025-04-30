# multi_source.py
# abstract-agent
# Author: tegridydev
# Repo: https://github.com/tegridydev/abstract-agent
# License: MIT
# Year: 2025

import os
import datetime
from typing import List, Tuple, Dict, Any
import arxiv
import requests
import xmltodict
from ollama import Client

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

def fetch_semanticscholar(topic, max_results=3) -> List[Dict[str, Any]]:
    """Fetches papers from Semantic Scholar API."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={topic}&limit={max_results}&fields=title,authors,year,abstract,url,citationCount" # Added citationCount
    results = []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for paper in data.get('data', []):
            if paper.get('title') and paper.get('abstract'):
                results.append({
                    'title': paper.get('title'),
                    'authors': ', '.join([a['name'] for a in paper.get('authors', []) if a.get('name')]),
                    'year': paper.get('year'),
                    'summary': paper.get('abstract', ''),
                    'url': paper.get('url'),
                    'citations': paper.get('citationCount', 0), # Get citation count
                    'source': 'Semantic Scholar',
                })
    except requests.exceptions.RequestException as e:
        print(f"[Error] Semantic Scholar request failed: {e}")
    except Exception as e:
        print(f"[Error] Semantic Scholar processing failed: {e}")
    return results

def fetch_europepmc(topic, max_results=3) -> List[Dict[str, Any]]:
    """Fetches papers from Europe PMC API."""
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={topic}&format=json&pageSize={max_results}&resultType=core" # Ensure resultType=core for citations
    results = []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for paper in data.get('resultList', {}).get('result', []):
            if paper.get('title') and paper.get('abstractText'):
                results.append({
                    'title': paper.get('title'),
                    'authors': paper.get('authorString', ''),
                    'year': paper.get('pubYear', ''),
                    'summary': paper.get('abstractText', ''),
                    'url': paper.get('doi', '') and f"https://doi.org/{paper['doi']}" or paper.get('fullTextUrlList', {}).get('fullTextUrl', [{}])[0].get('url', ''),
                    'citations': paper.get('citedByCount', 0), # Get citation count
                    'source': 'EuropePMC',
                })
    except requests.exceptions.RequestException as e:
        print(f"[Error] EuropePMC request failed: {e}")
    except Exception as e:
        print(f"[Error] EuropePMC processing failed: {e}")
    return results

def fetch_crossref(topic, max_results=3) -> List[Dict[str, Any]]:
    """Fetches papers from Crossref API."""
    url = f"https://api.crossref.org/works?query={topic}&rows={max_results}&filter=has-abstract:true" # Filter for abstracts
    results = []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get('message', {}).get('items', []):
            abstract = item.get('abstract', '')
            if isinstance(abstract, str) and abstract.strip().startswith('<'):
                try:
                   import re
                   abstract = re.sub('<[^>]*>', '', abstract).strip()
                except Exception:
                    pass # Keep original if regex fails somehow
            elif isinstance(abstract, list):
                abstract = ' '.join(str(a) for a in abstract)
            elif not isinstance(abstract, str):
                abstract = str(abstract)

            title_list = item.get('title', [])
            title = title_list[0] if title_list else None
            if title and abstract:
                results.append({
                    'title': title,
                    'authors': ', '.join([f"{a.get('given','')} {a.get('family','')}".strip() for a in item.get('author', []) if isinstance(a, dict)]) if 'author' in item else '',
                    'year': item.get('published-print', {}).get('date-parts', [[None]])[0][0] or item.get('created', {}).get('date-parts', [[None]])[0][0],
                    'summary': abstract,
                    'url': item.get('URL', ''),
                    'citations': item.get('is-referenced-by-count', 0), # Get citation count
                    'source': 'Crossref',
                })
    except requests.exceptions.RequestException as e:
        print(f"[Error] Crossref request failed: {e}")
    except Exception as e:
        print(f"[Error] Crossref processing failed: {e}")
    return results

def fetch_doaj(topic, max_results=3) -> List[Dict[str, Any]]:
    """Fetches papers from DOAJ API."""
    url = f"https://doaj.org/api/v2/search/articles/{topic}?page=1&pageSize={max_results}"
    results = []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get('results', []):
            bib = item.get('bibjson', {})
            if bib.get('title') and bib.get('abstract'):
                results.append({
                    'title': bib.get('title', ''),
                    'authors': ', '.join([a.get('name', '') for a in bib.get('author', []) if isinstance(a, dict)]),
                    'year': bib.get('year', ''),
                    'summary': bib.get('abstract', ''),
                    'url': next((link.get('url') for link in bib.get('link', []) if link.get('type') == 'fulltext'), bib.get('link', [{}])[0].get('url', '')), # Prefer fulltext link
                    'citations': 0, # DOAJ API doesn't typically return citation counts
                    'source': 'DOAJ',
                })
    except requests.exceptions.RequestException as e:
        print(f"[Error] DOAJ request failed: {e}")
    except Exception as e:
        print(f"[Error] DOAJ processing failed: {e}")
    return results

def fetch_biorxiv(topic, max_results=3) -> List[Dict[str, Any]]:
     """Placeholder for bioRxiv/medRxiv fetch. Currently unreliable."""
     # Note: bioRxiv API is tricky for arbitrary topic searches. The previous example used details endpoint structure incorrectly.
     # A proper implementation might require specific API endpoints or libraries not used here.
     # For now, returning empty to avoid misleading results or errors from a likely incorrect URL structure/method.
     print(f"[Warning] bioRxiv/medRxiv search implementation is currently disabled due to API limitations/complexity.")
     return []

def fetch_pubmed(topic, max_results=3) -> List[Dict[str, Any]]:
    """Fetches papers from PubMed API using E-utilities."""
    try:
        # ESearch to get PMIDs
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmax={max_results}&term={topic}&retmode=json"
        search_resp = requests.get(search_url, timeout=10)
        search_resp.raise_for_status() # Raise error for bad responses
        id_list = search_resp.json().get('esearchresult', {}).get('idlist', [])
        if not id_list:
            return []

        ids = ','.join(id_list)
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
        fetch_resp = requests.get(fetch_url, timeout=15)
        fetch_resp.raise_for_status()

        results = []
        docs = xmltodict.parse(fetch_resp.text)
        articles_data = docs.get('PubmedArticleSet', {})
        if not articles_data:
            print(f"[Warning] PubMed EFetch returned unexpected XML structure. Top keys: {list(docs.keys())}")
            return []
        articles = articles_data.get('PubmedArticle', [])
        if not isinstance(articles, list):
            articles = [articles] if articles else []

        for art in articles:
            medline_citation = art.get('MedlineCitation', {})
            if not medline_citation: continue
            pmid_obj = medline_citation.get('PMID')
            pmid = pmid_obj.get('#text') if isinstance(pmid_obj, dict) else pmid_obj
            if not pmid: continue

            article_info = medline_citation.get('Article', {})
            if not article_info: continue

            title_obj = article_info.get('ArticleTitle', '')
            title = title_obj.get('#text', '') if isinstance(title_obj, dict) else str(title_obj)

            authors_list = article_info.get('AuthorList', {}).get('Author', [])
            if authors_list and not isinstance(authors_list, list):
                 authors_list = [authors_list]
            authors = ', '.join(
                f"{a.get('LastName', '')} {a.get('Initials', '')}".strip()
                for a in authors_list if isinstance(a, dict) and a.get('LastName')
            )

            pub_date = article_info.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year_obj = pub_date.get('Year', '')
            year = year_obj.get('#text') if isinstance(year_obj, dict) else str(year_obj)
            if not year:
                 medline_date = pub_date.get('MedlineDate', '')
                 if isinstance(medline_date, str):
                      year = medline_date.split(' ')[0]

            abstract_section = article_info.get('Abstract', {})
            if not abstract_section:
                 continue
            abstract_text_obj = abstract_section.get('AbstractText', '')
            abstract = ''
            if isinstance(abstract_text_obj, list):
                abstract_list = []
                for part in abstract_text_obj:
                    if isinstance(part, str):
                        abstract_list.append(part)
                    elif isinstance(part, dict):
                        label = part.get('@Label', '')
                        text = part.get('#text', '')
                        if label and text:
                            abstract_list.append(f"{label}: {text}")
                        elif text:
                            abstract_list.append(text)
                abstract = ' '.join(abstract_list).strip()
            elif isinstance(abstract_text_obj, dict):
                abstract = abstract_text_obj.get('#text', '')
            elif isinstance(abstract_text_obj, str):
                 abstract = abstract_text_obj
            elif abstract_text_obj is not None:
                 abstract = str(abstract_text_obj)
            if title and abstract:
                results.append({
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'summary': abstract.strip(),
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    'citations': 0,
                    'source': 'PubMed',
                })
        return results
    except requests.exceptions.RequestException as e:
        print(f"[Error] PubMed request failed: {e}")
        return []
    except Exception as e:
        print(f"[Error] PubMed parsing failed: {e}")
        import traceback
        print(traceback.format_exc())
        return []


def fetch_openalex(topic, max_results=3) -> List[Dict[str, Any]]:
    """Fetches papers from OpenAlex API."""
    my_email = os.environ.get("OPENALEX_EMAIL", "anonymous@example.com")
    headers = {'User-Agent': f'AbstractAgent/1.0 (mailto:{my_email})'}
    url = f"https://api.openalex.org/works?search={topic}&per-page={max_results}&filter=has_abstract:true"
    results = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for paper in data.get('results', []):
            summary = ''
            abs_inv = paper.get('abstract_inverted_index')
            if isinstance(abs_inv, dict):
                try:
                    if not abs_inv:
                         summary = ""
                    else:
                         max_len = max(max(positions) for positions in abs_inv.values() if positions)
                         ordered_words = [''] * (max_len + 1)
                         for word, positions in abs_inv.items():
                             if positions:
                                 for pos in positions:
                                     if 0 <= pos <= max_len:
                                         ordered_words[pos] = word
                         summary = ' '.join(filter(None, ordered_words))
                except (ValueError, TypeError) as e:
                    print(f"[Warning] OpenAlex abstract reconstruction failed for paper '{paper.get('id')}': {e}")
                    summary = "[Abstract not available or invalid format]"
            elif isinstance(abs_inv, str):
                 summary = abs_inv
            title = paper.get('title')
            if title and summary:
                 results.append({
                     'title': title,
                     'authors': ', '.join([a['author']['display_name'] for a in paper.get('authorships', []) if a.get('author') and a['author'].get('display_name')]),
                     'year': paper.get('publication_year'),
                     'summary': summary,
                     'url': paper.get('doi') or paper.get('id'),
                     'citations': paper.get('cited_by_count', 0),
                     'source': 'OpenAlex',
                 })
    except requests.exceptions.RequestException as e:
        print(f"[Error] OpenAlex request failed: {e}")
    except Exception as e:
        print(f"[Error] OpenAlex processing failed: {e}")
    return results


def fetch_arxiv(topic, max_results=3) -> List[Dict[str, Any]]:
    """Fetches papers from arXiv API using the official library."""
    results = []
    try:
        client = arxiv.Client(
             page_size = max_results,
             delay_seconds = 3,
             num_retries = 3
        )
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        for paper in client.results(search):
            if paper.title and paper.summary:
                 results.append({
                     'title': paper.title,
                     'authors': ', '.join([a.name for a in paper.authors if a.name]),
                     'year': paper.published.year if paper.published else None,
                     'summary': paper.summary.strip(),
                     'url': paper.entry_id,
                     'citations': 0,
                     'source': 'arXiv',
                 })
            if len(results) >= max_results:
                 break
    except Exception as e:
        print(f"[Error] arXiv fetch failed: {e}")
    return results


# --- Aggregation Logic ---

def normalize_year(year: Any) -> float:
    """Normalizes year to a recency score (0.0 to 1.0)."""
    try:
        year_int = int(year)
        this_year = datetime.datetime.now().year
        age = max(0, this_year - year_int)
        recency_score = max(0.0, 1.0 - min(age, 20) / 20.0)
        return recency_score
    except (ValueError, TypeError, AttributeError):
        return 0.5


def normalize_citations(cites: Any, max_cites: int) -> float:
    """Normalizes citation count to a score (0.0 to 1.0) based on max found."""
    try:
        cites_int = int(cites)
        if max_cites <= 0:
            return 0.0
        return min(1.0, cites_int / max_cites)
    except (ValueError, TypeError):
        return 0.0


def simple_relevance_score(topic: str, title: str, summary: str) -> float:
    """Calculates a basic relevance score based on keyword overlap."""
    try:
        topic_words = set(filter(None, topic.lower().split()))
        text = (str(title) or '') + ' ' + (str(summary) or '')
        text_words = set(filter(None, text.lower().split()))
        if not topic_words:
             return 0.0
        overlap = len(topic_words.intersection(text_words))
        return overlap / len(topic_words)
    except Exception as e:
        print(f"[Warning] Failed to calculate relevance score: {e}")
        return 0.0


def aggregate_sources(topic: str, max_results: int = 5, **kwargs) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetches papers from all sources, deduplicates, scores, ranks,
    and returns the top N papers and the total count before deduplication.

    Args:
        topic: The research topic query.
        max_results: The maximum number of *top* papers to return after ranking.
        **kwargs: Catches unused parameters from potential older calls or config.

    Returns:
        A tuple containing:
        - list: Top N ranked papers (dictionaries).
        - int: Total number of papers fetched across all sources before deduplication.
    """
    all_fetched_results: List[Dict[str, Any]] = []
    fetch_functions = [
        fetch_arxiv, fetch_semanticscholar, fetch_pubmed, fetch_openalex,
        fetch_europepmc, fetch_crossref, fetch_doaj
    ]
    for fetch_func in fetch_functions:
        print(f"Fetching from {fetch_func.__name__}...")
        try:
            results = fetch_func(topic, max_results=max_results + 5)
            print(f" -> Found {len(results)} results from {fetch_func.__name__}.")
            if results:
                all_fetched_results.extend(results)
        except Exception as e:
            print(f"[Warning] Failed to fetch or process from {fetch_func.__name__}: {e}")

    total_fetched_count = len(all_fetched_results)
    print(f"\nTotal fetched before deduplication: {total_fetched_count}")

    seen_identifiers = set()
    deduped = []
    for item in all_fetched_results:
        title = item.get('title','').lower().strip()
        import re
        title_norm = re.sub(r'[^\w\s]', '', title)
        title_norm = ' '.join(title_norm.split())

        url = item.get('url', '').lower().strip()
        identifier = url if 'doi.org' in url else title_norm

        if identifier and identifier not in seen_identifiers:
            if item.get('title') and len(str(item.get('summary', ''))) > 50:
                 item['citations'] = item.get('citations', 0)
                 item['year'] = item.get('year', None)

                 deduped.append(item)
                 seen_identifiers.add(identifier)

    print(f"Total after deduplication and quality filtering: {len(deduped)}")

    max_cites = 0
    for paper in deduped:
        try:
            cites = int(paper.get('citations', 0))
            paper['citations'] = cites
            if cites > max_cites:
                max_cites = cites
        except (ValueError, TypeError):
            paper['citations'] = 0

    print(f"Maximum citations found in batch: {max_cites}")

    scored_papers = []
    for paper in deduped:
        rel = simple_relevance_score(topic, paper.get('title',''), paper.get('summary',''))
        year_score = normalize_year(paper.get('year'))
        cite_score = normalize_citations(paper.get('citations', 0), max_cites)

        score = (0.5 * rel) + (0.3 * year_score) + (0.2 * cite_score)

        paper['relevance_score'] = round(rel, 3)
        paper['recency_score'] = round(year_score, 3)
        paper['citation_score'] = round(cite_score, 3)
        paper['composite_score'] = round(score, 3)
        scored_papers.append(paper)

    scored_papers.sort(key=lambda x: x['composite_score'], reverse=True)

    top_papers = scored_papers[:max_results]
    print(f"Returning top {len(top_papers)} papers.")

    return top_papers, total_fetched_count