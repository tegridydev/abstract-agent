# Abstract Agent

research questions, local models and a record of the evidence

I wanted a way to work through research ideas without losing track of the papers behind them.
Abstract Agent searches scholarly sources, builds an evidence record and asks local models to
propose and critique possible directions. It checks the closest prior art before writing a proposal.

The interesting part is being able to look back through the work. What did the search find?
Which papers support the idea? Where did a model struggle, and what still needs checking?
Each run keeps those details alongside the result.

This is a tool for exploring questions. A finished proposal still needs someone to read the
sources, check the reasoning and decide whether the experiment is worth doing.

## Run locally

Use Python 3.11 to 3.14, Ollama and uv from the 0.11 series, starting at 0.11.32.
From this folder, with Ollama running:

```bash
uv sync --locked --all-groups
ollama pull gemma3:4b
uv run --locked abs.py
```

The menu walks through the topic, models and run settings. You can use one model throughout
or give planning, generation, critique and writing their own models. Opening the menu does
not contact Ollama or the literature providers.

For a run from the terminal:

```bash
uv run --locked abstract-agent run \
  --topic "Novel methods for language model compression" \
  --model gemma3:4b
```

The model above is a starting example. Available memory and model quality will affect what
works on your machine. The initial probe checks whether a model can respond in a usable format;
it cannot tell you whether that model will produce good science.

You can also install with `python -m pip install .` and use `abstract-agent`.
That resolves the dependency ranges in `pyproject.toml` rather than using `uv.lock`.
`python -m agent`, `python abs.py` and the older `python agent/agent.py` use the same CLI.

## What happens in a run

1. Search the topic and related facets across the scholarly providers.
2. Check relevance and build evidence cards from passages in the retrieved abstracts.
3. Develop candidate ideas, then review their evidence and proposed methods.
4. Revise a candidate and search for the closest existing work.
5. Save a proposal, or save the evidence with an explanation of why the run stopped there.

The default evidence requirement is six relevant papers across two facets, including three
with direct or transferable evidence. Surveys can help with context. The program records
any override of the evidence requirement.

Search adapters cover arXiv, Semantic Scholar, PubMed, OpenAlex, Europe PMC, Crossref, DOAJ,
HAL, OpenAIRE and CORE. There are no separate bioRxiv or medRxiv adapters. A provider can
return nothing, fail or reach a request limit; those outcomes stay in the run record.

## Results and privacy

Results go to `agent/output/` unless you set `--output-dir`. JSON keeps the evidence, model
metrics, warnings and intermediate context; the text file gives you a readable result.
Check `outcome` and the markers for human review, even when `status` is `complete`.

Literature queries leave your machine. Inference stays local when both the Ollama server
and the selected models are local. Cloud models require consent, and a remote host produces
a warning. Topics, abstracts and generated content can appear in saved results.

The local settings, caches, knowledge database and generated outputs are excluded from Git
by the default ignore rules. Keep API keys in environment variables. More detail is in
[privacy](docs/privacy.md) and [usage](docs/usage.md).

The package version is **5.0.0**. The workflow and result schema are **6.0**.
They describe different parts of the project and are versioned separately.

## Keep exploring

* [Usage and configuration](docs/usage.md)
* [How the pipeline fits together](docs/architecture.md)
* [Model output and recovery](docs/model-output.md)
* [Local research knowledge](docs/knowledge.md)
* [Contributing and checks](CONTRIBUTING.md)

Everything here is available under the MIT license. Feel free to read through, try things
and build on it. See [LICENSE](LICENSE) for the terms.
