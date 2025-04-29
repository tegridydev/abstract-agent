# abstract-agent

No-BS, easily extendable, 100% local multi-agent system for generating novel research hypotheses, abstracts, and references. 

All powered by local Ollama LLMs. No API keys. No cloud. Just you, your GPU/CPU, and public sources.   

---

## Features
- Multi-agent pipeline: breakdown, critique, synthesize, innovate, and polish
- Pulls from public sources: arXiv, Semantic Scholar, EuropePMC, Crossref, DOAJ, bioRxiv, medRxiv, OpenAlex, PubMed
- Scores, ranks, and summarizes literature
- Uses Ollama for summarization and novelty checks
- Final output is a clean, human-readable panel with stats / insights

---

## Quickstart

1. Clone this repo:
   ```bash
   git clone https://github.com/tegridydev/abstract-agent
   cd abstract-agent
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install [Ollama](https://ollama.com/download) and pull a model (e.g. qwen3:0.6b):
   ```bash
   ollama pull qwen3:0.6b
   ```
4. Run the agent:
   ```bash
   python agent.py
   ```

---

## Agent Pipeline (Lego block style)
- **Agent A:** Breaks down your topic into all the core pieces
- **Agent B:** Roasts the literature, finds gaps and trends
- **Agent C:** Synthesizes new directions
- **Agent D:** Goes wild, generates bold hypotheses
- **Agent E:** Polishes, references, and scores the final abstract
- **Novelty Check:** Checks if it's actually new or just recycled

---

## Output
- Final hypothesis, novelty score, references, and run stats (references searched/used, time taken)

---

## Dependencies
- ollama
- rich
- arxiv
- requests
- xmltodict

No API keys. All sources are public.

---

## How to modify
- Edit `agents_config.yaml` to change the agent pipeline, prompts, or personas
- Add new sources in `multi_source.py`

---

## License / Citations
MIT. Use it, fork it, break it, share it. Just give a shoutout to tegridydev if you want <3

[![MIT License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Author:** tegridydev  
**Repo:** https://github.com/tegridydev/abstract-agent

