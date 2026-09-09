# Local research knowledge

The knowledge store keeps retrieved sources available for later work. It also records ideas,
but those ideas never become external evidence or proof of novelty.

`knowledge/index.sqlite3` is the authoritative database. The `records/`, `findings/` and
`ideas/` folders are readable projections. This is generated research data and stays out of
the public repository.

## Use the store

```bash
abstract-agent knowledge status
abstract-agent knowledge search "model compression" --limit 10
abstract-agent knowledge export
abstract-agent knowledge reindex
abstract-agent knowledge import /path/to/records
```

To use a different location, put the directory option before the operation:

```bash
abstract-agent knowledge --knowledge-dir /path/to/knowledge status
```

These commands open the store and can create directories and a database, including status
and search. Export writes the projections. Import reads paper and evidence records from
another local export.

## Reuse and refresh

Evidence is reusable only when paper identity, normalised content hash and evidence contract
version match. Generated ideas stay in their own trust tier. An installed embedding model
can rank passages; lexical scoring is available when embeddings are not.

Research enables knowledge storage by default. `--no-knowledge` disables it,
`--knowledge-dir PATH` changes its location and `--embedding-model NAME` selects embeddings.
The program does not pull embedding models automatically.

Balanced refresh keeps discovery queries for seven days and prior art queries for one day.
`--knowledge-refresh always` refreshes providers. `--knowledge-refresh offline` retrieves
literature only from the local store. Combining offline refresh with `--no-knowledge` is
rejected because there would be no local store to query.

Offline refresh does not disable inference. Use a local Ollama host and local models if you
want inference to stay on your machine too. See [privacy](privacy.md) and [usage](usage.md).
