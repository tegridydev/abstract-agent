# abstract-agent

Easily extendable 100% local multi-agent system for generating novel research hypotheses, abstracts, and references. 

All powered by local Ollama LLMs. No API keys. No cloud. Just you, your GPU/CPU, and public sources.   

---

## Features
- Multi-agent pipeline: breakdown, critique, synthesize, innovate, and polish
- Pulls from public sources: arXiv, Semantic Scholar, EuropePMC, Crossref, DOAJ, bioRxiv, medRxiv, OpenAlex, PubMed
- Scores, ranks, and summarizes literature
- Uses Ollama for summarization and novelty checks
- Final output is a clean, human-readable panel with stats / insights

---

## Example Outputs

```bash
────────────────────────────────────────────── Pipeline 'Research Hypothesis Generation' Finished in 102.67s ───────────────────────────────────────────────
────────────────────────────────────────────────────────────────── Final Results Summary ───────────────────────────────────────────────────────────────────
╭────────────────────────────────────────────────────────────── Final Hypothesis Structured ───────────────────────────────────────────────────────────────╮
│ This research introduces a novel approach to Large Language Model (LLM) compression predicated on Neuro-Symbolic Contextual Compression. We propose a    │
│ system that translates LLM attention maps into a discrete, graph-based representation, subsequently employing a learned graph pruning algorithm to       │
│ remove irrelevant nodes while preserving critical semantic relationships. Unlike existing compression methods focused on direct neural manipulation,     │
│ this approach leverages the established techniques of graph pruning, offering potentially significant gains in model size and efficiency. The            │
│ integration of learned pruning, adapting to specific task and input characteristics, represents a fundamentally new paradigm for LLM compression, moving │
│ beyond purely neural optimizations.                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────────────────── Novelty Assessment ───────────────────────────────────────────────────────────────────╮
│                                                                                                                                                          │  
│                                                                                                                                                          │
│ **Novelty Score: 7/10**                                                                                                                                  │
│                                                                                                                                                          │
│ **Reasoning:**                                                                                                                                           │
│                                                                                                                                                          │
│ This hypothesis demonstrates a moderate level of novelty, primarily due to the specific combination of techniques and the integration of neuro-symbolic  │
│ approaches. Let's break down the assessment:                                                                                                             │
│                                                                                                                                                          │
│ * **Elements of Novelty (Strengths):**                                                                                                                   │
│     * **Neuro-Symbolic Contextual Compression:** The core idea of translating LLM attention maps into a discrete, graph-based representation *is* a      │
│ relatively new area of exploration. While graph pruning exists, applying it specifically to the output of LLM attention maps – and framing it within a   │
│ neuro-symbolic context – is a distinctive aspect.                                                                                                        │
│     * **Learned Graph Pruning:** The explicit mention of a *learned* graph pruning algorithm elevates the novelty. Many pruning methods are static,      │
│ whereas learning the pruning criteria based on task and input characteristics is a significant step forward.                                             │
│     * **Integration of Graph Pruning with LLMs:** While graph pruning is used in other domains, its application to LLMs, particularly in this way, is    │
│ not widely established.                                                                                                                                  │
│                                                                                                                                                          │
│ * **Elements Limiting Novelty (Weaknesses):**                                                                                                            │
│     * **Graph Pruning is Not Entirely New:** As highlighted in Paper 1, graph pruning techniques exist in general. The core concept of pruning nodes     │
│ based on importance is well-established.                                                                                                                 │
│     * **Related Work Exists:** Several papers (Papers 2, 3, 4, 5, 6, 7) address aspects of model compression, including quantization, sparsity, and      │
│ dynamic budgets.  While the *combination* is novel, the individual components are not.  Paper 7's "thinking step-by-step compression" is particularly    │
│ relevant, even though it uses a different framing (dynamic compression of reasoning steps).                                                              │
│     * **Fine-grained vs. Coarse-grained:** The hypothesis positions itself against "coarse-grained" methods (Paper 1). However, many current compression │
│ techniques are moving towards finer-grained approaches.                                                                                                  │
│                                                                                                                                                          │
│                                                                                                                                                          │
│ **Justification for the Score:**                                                                                                                         │
│                                                                                                                                                          │
│ A score of 7 reflects that the hypothesis presents a novel *approach* rather than a completely new concept. The combination of learned graph pruning     │
│ with attention maps represents a worthwhile exploration. However, it's not a revolutionary breakthrough because graph pruning itself isn’t entirely      │
│ novel, and the field is already actively investigating various compression strategies.                                                                   │
│                                                                                                                                                          │
│ **Recommendations for Strengthening the Hypothesis:**                                                                                                    │
│                                                                                                                                                          │
│ * **Quantify the Expected Gains:**  Adding specific claims about the expected reduction in model size and efficiency would strengthen the hypothesis.    │
│ * **Elaborate on the "Neuro-Symbolic" Aspect:**  Provide more detail on how the discrete graph representation represents the underlying semantic         │
│ relationships within the LLM.                                                                                                                            │
│ * **Highlight the Advantage over Existing Methods:**  Clearly articulate *why* this approach is expected to be superior to existing techniques (e.g., in │
│ terms of accuracy, speed, or ease of implementation).                                                                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

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
3. Install [Ollama](https://ollama.com/download) and pull a model (e.g. gemma3:4b):
   ```bash
   ollama pull gemma3:4b
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
- pydantic
- pyyaml

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

