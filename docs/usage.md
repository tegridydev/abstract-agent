# Usage and configuration

Start with the menu if you want to look through the settings before running anything:

```bash
uv run --locked abs.py
```

The research screen lets you choose a topic, review the settings and select models. The
models screen shows installed and running models, context size and capabilities. It can
pull a model after confirmation, assign roles or reset learned response compatibility.

## Run from the terminal

```bash
abstract-agent run --topic "Language model compression" --model gemma3:4b
```

Use `--planning-model`, `--creative-model`, `--critique-model` and `--writing-model` for
separate roles. Supply names that are installed in your Ollama server. Loading several
large models can take time and may cause Ollama to swap them as the run moves between roles.

Before research starts, the program checks each selected model for completion support and
probes its response format. Schema, JSON and text routes are supported. A successful probe
only shows that the transport works; later stages still validate the actual content.

## Evidence and scope

The default requirement is six relevant papers across two facets, including three direct
or transferable studies. Surveys add context but cannot establish a hypothesis or novelty
on their own.

`--minimum-papers N` changes the paper threshold. `--allow-insufficient-literature` bypasses
the admission gate and records the override. Use that deliberately when exploring sparse
literature, then read the evidence limitations in the result.

`--search-depth deep` increases retrieval effort. The interactive flow can offer this retry
when a standard search runs out of usable evidence. Research stays broad by default;
`--scope "Exact discovered scope"` selects a supported evidence domain.

## Resume a run

```bash
abstract-agent resume --diagnostic agent/output/failed_RUN.json
```

Replace the example path with your diagnostic. The file must pass checksum, workflow and
input validation. A diagnostic can be resumed up to three times. Completed mapped units
are kept so the remaining work can continue. The menu also offers resume after a recoverable
failure. See [model output and recovery](model-output.md) for what can be recovered.

## Connection and settings

CLI host and timeout arguments override their environment variables, which override the
workflow defaults. Saved preferences initialise interactive runs. Commands run directly
from the terminal do not load that preferences file. Resume restores selected checkpoint
settings unless a corresponding CLI override is supplied.

* `OLLAMA_HOST` sets the Ollama base URL.
* `OLLAMA_TIMEOUT_SECONDS` sets the inactivity timeout, which defaults to 600 seconds.
* `ABSTRACT_AGENT_CONTACT_EMAIL` supplies contact information to Crossref and NCBI.
* `SEMANTIC_SCHOLAR_API_KEY` supplies an optional Semantic Scholar key.
* `OPENALEX_API_KEY` supplies an optional OpenAlex key.
* `NCBI_API_KEY` supplies an optional NCBI key.
* `ABSTRACT_AGENT_CACHE_PATH` changes the SQLite literature cache location.

`--ollama-inactivity-timeout SECONDS` and its older alias `--ollama-timeout` set the same
inactivity limit. Generation can run longer than this while response data keeps arriving.

Preferences live in `.abstract-agent.toml`. The application stores settings there, not
research topics, model responses or API keys. Relative paths for output, knowledge, cache
and preferences are resolved from the current working directory.

## Check a service or inspect local knowledge

```bash
abstract-agent sources check
abstract-agent knowledge status
abstract-agent knowledge search "model compression" --limit 10
```

Source checks contact the providers. Knowledge commands use the local store and can create
its directories and database. See [local knowledge](knowledge.md) for refresh and import details.

`abstract-agent --help`, `abstract-agent run --help` and `abstract-agent resume --help`
show the command options without contacting services.
