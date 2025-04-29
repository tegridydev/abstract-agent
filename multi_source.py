# abstract-agent
# Author: tegridydev
# Repo: https://github.com/tegridydev/abstract-agent
# License: MIT
# Year: 2025

import os
import datetime
from typing import List

import arxiv
import requests
import xmltodict
from ollama import Client

OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

# --- Utility: Summarize papers using Ollama ---
def summarize_papers_via_ollama(papers: List[dict], topic: str, model: str = 'qwen3:0.6b') -> str:
    client = Client(host=OLLAMA_HOST)
    abstracts = '\n'.join([f"- {p['title']}: {p.get('summary','')[:500]}" for p in papers if p.get('summary')])
    prompt = f"You are a research assistant. Summarize the following academic papers related to '{topic}'. Highlight key findings, trends, and gaps.\n\n{abstracts}"
    try:
        response = client.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        return f"[Summarization failed: {e}]"

# --- Semantic Scholar ---
def fetch_semanticscholar(topic, max_results=3):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={topic}&limit={max_results}&fields=title,authors,year,abstract,url"
    resp = requests.get(url)
    results = []
    if resp.ok:
        data = resp.json()
        for paper in data.get('data', []):
            results.append({
                'title': paper.get('title'),
                'authors': ', '.join([a['name'] for a in paper.get('authors', [])]),
                'year': paper.get('year'),
                'summary': paper.get('abstract', ''),
                'url': paper.get('url'),
                'source': 'Semantic Scholar',
            })
    return results

# --- Europe PMC ---
def fetch_europepmc(topic, max_results=3):
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={topic}&format=json&pageSize={max_results}"
    try:
        resp = requests.get(url, timeout=10)
        results = []
        if resp.ok:
            data = resp.json()
            for paper in data.get('resultList', {}).get('result', []):
                results.append({
                    'title': paper.get('title'),
                    'authors': paper.get('authorString', ''),
                    'year': paper.get('pubYear', ''),
                    'summary': paper.get('abstractText', ''),
                    'url': paper.get('doi', '') and f"https://doi.org/{paper['doi']}" or paper.get('fullTextUrlList', {}).get('fullTextUrl', [{}])[0].get('url', ''),
                    'source': 'EuropePMC',
                })
        return results
    except Exception:
        return []

# --- Crossref ---
def fetch_crossref(topic, max_results=3):
    url = f"https://api.crossref.org/works?query={topic}&rows={max_results}"
    try:
        resp = requests.get(url, timeout=10)
        results = []
        if resp.ok:
            data = resp.json()
            for item in data.get('message', {}).get('items', []):
                abstract = item.get('abstract', '')
                # Sometimes abstract is a list or dict
                if isinstance(abstract, list):
                    abstract = ' '.join(str(a) for a in abstract)
                elif not isinstance(abstract, str):
                    abstract = str(abstract)
                results.append({
                    'title': item.get('title', [''])[0],
                    'authors': ', '.join([f"{a.get('given','')} {a.get('family','')}" for a in item.get('author', [])]) if 'author' in item else '',
                    'year': item.get('published-print', {}).get('date-parts', [[None]])[0][0] or item.get('created', {}).get('date-parts', [[None]])[0][0],
                    'summary': abstract,
                    'url': item.get('URL', ''),
                    'source': 'Crossref',
                })
        return results
    except Exception:
        return []

# --- DOAJ ---
def fetch_doaj(topic, max_results=3):
    url = f"https://doaj.org/api/v2/search/articles/{topic}?page=1&pageSize={max_results}"
    try:
        resp = requests.get(url, timeout=10)
        results = []
        if resp.ok:
            data = resp.json()
            for item in data.get('results', []):
                bib = item.get('bibjson', {})
                results.append({
                    'title': bib.get('title', ''),
                    'authors': ', '.join([a.get('name', '') for a in bib.get('author', [])]),
                    'year': bib.get('year', ''),
                    'summary': bib.get('abstract', ''),
                    'url': bib.get('link', [{}])[0].get('url', ''),
                    'source': 'DOAJ',
                })
        return results
    except Exception:
        return []

# --- bioRxiv/medRxiv ---
def fetch_biorxiv(topic, max_results=3):
    url = f"https://api.biorxiv.org/details/biorxiv/2020-01-01/3000-12-31/{topic}/{max_results}"
    try:
        resp = requests.get(url, timeout=10)
        results = []
        if resp.ok:
            data = resp.json()
            for paper in data.get('collection', []):
                results.append({
                    'title': paper.get('title', ''),
                    'authors': paper.get('authors', ''),
                    'year': paper.get('date', '')[:4],
                    'summary': paper.get('abstract', ''),
                    'url': paper.get('doi', '') and f"https://doi.org/{paper['doi']}" or '',
                    'source': 'bioRxiv/medRxiv',
                })
        return results
    except Exception:
        return []

# --- PubMed ---
def fetch_pubmed(topic, max_results=3):
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmax={max_results}&term={topic}&retmode=json"
    search_resp = requests.get(search_url)
    id_list = search_resp.json().get('esearchresult', {}).get('idlist', [])
    if not id_list:
        return []
    ids = ','.join(id_list)
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
    fetch_resp = requests.get(fetch_url)
    results = []
    if fetch_resp.ok:
        docs = xmltodict.parse(fetch_resp.text)
        articles = docs['PubmedArticleSet'].get('PubmedArticle', [])
        if not isinstance(articles, list):
            articles = [articles]
        for art in articles:
            info = art['MedlineCitation']['Article']
            title = info.get('ArticleTitle', '')
            authors = ', '.join([a['LastName'] for a in info.get('AuthorList', {}).get('Author', []) if 'LastName' in a])
            year = info.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}).get('Year', '')
            abstract = info.get('Abstract', {}).get('AbstractText', '')
            # Abstract can be a list (multiple sections), join if so
            if isinstance(abstract, list):
                abstract = ' '.join(str(a) for a in abstract)
            elif not isinstance(abstract, str):
                abstract = str(abstract)
            results.append({
                'title': title,
                'authors': authors,
                'year': year,
                'summary': abstract,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{art['MedlineCitation']['PMID']}/",
                'source': 'PubMed',
            })
    return results

# --- OpenAlex ---
def fetch_openalex(topic, max_results=3):
    url = f"https://api.openalex.org/works?search={topic}&per-page={max_results}"
    resp = requests.get(url)
    results = []
    if resp.ok:
        data = resp.json()
        for paper in data.get('results', []):
            # OpenAlex abstracts are dicts: {word: [positions]} -- reconstruct to string
            abs_inv = paper.get('abstract_inverted_index', '')
            if isinstance(abs_inv, dict):
                # reconstruct text by ordering words by position
                word_positions = []
                for word, poses in abs_inv.items():
                    for pos in poses:
                        word_positions.append((pos, word))
                word_positions.sort()
                summary = ' '.join(word for _, word in word_positions)
            elif isinstance(abs_inv, str):
                summary = abs_inv
            else:
                summary = ''
            results.append({
                'title': paper.get('title'),
                'authors': ', '.join([a['author']['display_name'] for a in paper.get('authorships', [])]),
                'year': paper.get('publication_year'),
                'summary': summary,
                'url': paper.get('id'),
                'source': 'OpenAlex',
            })
    return results

# --- arXiv ---
def fetch_arxiv(topic, max_results=3):
    search = arxiv.Search(query=topic, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results = []
    for paper in arxiv.Client().results(search):
        results.append({
            'title': paper.title,
            'authors': ', '.join([a.name for a in paper.authors]),
            'year': paper.published.year,
            'summary': paper.summary,
            'url': paper.entry_id,
            'source': 'arXiv',
        })
    return results

# --- Aggregator ---
def normalize_year(year):
    try:
        year = int(year)
        this_year = datetime.datetime.now().year
        # Recency: 1.0 for this year, decays to 0.0 for 20+ years old
        return max(0.0, 1.0 - min((this_year - year), 20) / 20)
    except Exception:
        return 0.5

def normalize_citations(cites, max_cites):
    try:
        cites = int(cites)
        if max_cites == 0:
            return 0.0
        return min(1.0, cites / max_cites)
    except Exception:
        return 0.0

def simple_relevance_score(topic, title, summary):
    # Fallback: count keyword overlap
    topic_words = set(topic.lower().split())
    text = (title or '') + ' ' + (summary or '')
    text_words = set(text.lower().split())
    overlap = topic_words & text_words
    return len(overlap) / (len(topic_words) + 1e-5)

def aggregate_sources(topic, max_results=3, summarize=False, ollama_model='qwen3:0.6b'):
    all_results = []
    all_results.extend(fetch_arxiv(topic, max_results))
    all_results.extend(fetch_semanticscholar(topic, max_results))
    all_results.extend(fetch_pubmed(topic, max_results))
    all_results.extend(fetch_openalex(topic, max_results))
    all_results.extend(fetch_europepmc(topic, max_results))
    all_results.extend(fetch_crossref(topic, max_results))
    all_results.extend(fetch_doaj(topic, max_results))
    all_results.extend(fetch_biorxiv(topic, max_results))
    # Deduplicate by title
    seen = set()
    deduped = []
    for item in all_results:
        if item['title'] and item['title'] not in seen:
            deduped.append(item)
            seen.add(item['title'])
    # Gather max citation count for normalization
    max_cites = 0
    for paper in deduped:
        cites = 0
        # Try to extract citation count from paper dict if present
        for k in ['citationCount', 'cited_by_count', 'cited_by', 'is-referenced-by-count']:
            if k in paper and paper[k]:
                try:
                    cites = int(paper[k])
                    break
                except Exception:
                    continue
        paper['citations'] = cites
        if cites > max_cites:
            max_cites = cites
    # Score and sort
    scored = []
    for paper in deduped:
        rel = simple_relevance_score(topic, paper.get('title',''), paper.get('summary',''))
        year_score = normalize_year(paper.get('year',''))
        cite_score = normalize_citations(paper.get('citations',0), max_cites)
        # Composite: weight relevance highest, then recency, then citations
        score = 0.5 * rel + 0.3 * year_score + 0.2 * cite_score
        paper['relevance_score'] = rel
        paper['recency_score'] = year_score
        paper['citation_score'] = cite_score
        paper['composite_score'] = score
        scored.append(paper)
    # Sort by composite score
    scored.sort(key=lambda x: x['composite_score'], reverse=True)
    top_papers = scored[:max_results]
    # Optionally summarize
    summary = None
    if summarize:
        summary = summarize_papers_via_ollama(top_papers, topic, model=ollama_model)
    return top_papers if not summarize else (top_papers, summary)
