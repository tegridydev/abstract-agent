# How the pipeline fits together

The workflow lives in `agent/pipeline/default.yaml`. Each stage declares its inputs,
outputs and handler. Prompts live beside it in `prompts/`. The config loader resolves
those files within the pipeline directory and validates the result before a run starts.

The useful boundary is between work owned by the application and content proposed by a
model. Python owns identifiers, evidence links, scope, status and validation. Models supply
content for specific contracts. That keeps a fluent response from quietly changing the
meaning of a record.

## Where things live

* `agent.agent` provides the CLI and orchestration, with compatibility exports for older callers.
* `agent.contracts` holds the typed research, pipeline, model and provider records.
* `agent.llm` defines generation contracts and response policies.
* `agent.literature` handles provider transport, retry rules, caching and relevance.
* `agent.knowledge` stores source records, extracts evidence and handles embeddings.
* `agent.application` provides file persistence helpers.
* `agent.presentation` loads settings and discovers saved runs.

Within `agent.pipeline`, `context.py` and `state.py` manage execution state. The generic
executor invokes trusted tools through the registry. `stages/runtime.py` supplies run
settings, provider circuits and embedding services to those tools. Scientific decisions
belong in `stages/`, rather than in the generic executor.

`model_io/` reads streamed responses and records transport capabilities. `prompting/`
estimates the complete request budget. `output/` parses and validates responses, records
compilation outcomes and manages compatibility profiles. Its `validation.py` contains
response checks and normalisation for metadata owned by the application.

`recovery/` prepares bounded packets for repair. `persistence.py` writes results and checks
diagnostics, while `migrations.py` maps compatible older diagnostics into the current graph.
`ports.py` describes the interfaces for replacement adapters.

## Evidence and selection

Relevance starts with the user's topic. Planned concepts can strengthen a match but cannot
replace that anchor. Evidence cards use exact abstract passages and retain source offsets.
An installed embedding model can rank passages; lexical scoring is the fallback. The
program does not pull an embedding model automatically.

Critics review one candidate per request. The compiler attaches the candidate identity from
the packet, even if the model tries to supply another one. Fallback scores do not count
as critic coverage. One usable critic can support selection with low confidence if the
scientific and grounding checks pass; two usable critics provide fuller coverage.

Novelty remains `uncertain` or `prior_art_overlap` when that is what the evidence supports.
`novelty_supported` requires at least three admissible papers across two providers for each
candidate independently. That is an application threshold, not proof that an idea is new.

## Requests and limits

Requests account for system text, data, schema, output allowance, reasoning reserve and a
safety margin. Compact packets can shorten selected display text. Canonical records remain
in context, and protected scientific views keep their required fields complete. Serialised
JSON is never cut in the middle of an object.

Queries run serially within each provider and concurrently across providers. Retries cover
429 responses, server errors, timeouts and network failures with bounded delays.
`Retry-After` and supported reset timestamps are capped at 30 seconds. A provider circuit
opened during discovery stays open during prior art retrieval for that run. Deep search
allows at most two external expansions based on local ideas.

## Saved results

The package is version `5.0.0`; the workflow and JSON result schema are `6.0`.
A result records the pipeline, run settings, source health, model metrics, warnings,
compilation and generation events, and the final context.

The main context fields are `audited_research_proposal`, `evidence_audit`,
`novelty_verification`, `candidate_scorecard` and `evidence_papers`. Metrics include model
assignments, request counts, response routes, prompt hashes, output budgets, source offsets,
recovery events and checks that require human review.

Read `outcome` alongside `status`. A completed envelope can hold `proposal_complete`,
`evidence_only`, `insufficient_evidence` or `insufficient_valid_candidates`.
Transport failures, interruptions and configuration errors remain separate outcomes.

Writes are atomic per file. Step outputs are staged together, so an invalid result with
several outputs cannot partly update the context. Diagnostics from versions 4.7 through
5.0 are checked against their original manifest and checksum before migration. The loader
also checks typed inputs and the resume limit. Older results can still appear in history.

See [model recovery](model-output.md), [local knowledge](knowledge.md) and
[contributing](../CONTRIBUTING.md) for the next layer of detail.
