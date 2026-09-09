# Privacy and external services

There are two separate places your research can go: the literature providers and the model
server. Choosing a local model only addresses the second one.

## Literature

Normal retrieval sends queries derived from your topic to enabled scholarly providers.
Caching, provider circuits and offline knowledge refresh can avoid some or all requests.
Retrieved titles and abstracts are treated as external data when included in prompts.

Provider documentation is available here:

* [arXiv](https://info.arxiv.org/help/api/index.html)
* [Semantic Scholar](https://www.semanticscholar.org/product/api)
* [NCBI](https://www.ncbi.nlm.nih.gov/books/NBK25497/)
* [OpenAlex](https://developers.openalex.org/)
* [Europe PMC](https://europepmc.org/RestfulWebService)
* [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/)
* [DOAJ](https://doaj.org/)

NCBI provides a [disclaimer and copyright notice](https://www.ncbi.nlm.nih.gov/About/disclaimer.html).
Abstracts may contain copyrighted material. These links are references; local checks do not
verify provider availability or their latest terms.

## Models

Inference is local when `OLLAMA_HOST` points to a local server and every selected model is
local. A remote host produces a warning. Cloud models require `--allow-cloud-models` in the
CLI, or consent in the interactive flow. Prompts sent to those services can contain research
topics and source material.

## Files on your machine

Run files can contain topics, abstracts, intermediate model content, warnings and metrics.
The knowledge store keeps source records and generated ideas in separate trust tiers.
Preferences contain settings rather than research content or API keys.

Keep keys in environment variables. Default output, knowledge, cache and settings locations
are ignored by Git. Custom locations need the same care. Review files before sharing them;
research records and local credentials belong outside the public repository.
