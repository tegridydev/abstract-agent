# Contributing

The useful changes are usually the ones that make a run easier to understand or a failure
easier to reproduce. If you change behaviour, keep a small test that shows what was wrong
and what the result should be.

## Set up

Use Python 3.11 to 3.14 and the uv version range in `pyproject.toml`:

```bash
uv sync --locked --all-groups
```

Once the dependencies are installed, these checks run without fetching anything:

```bash
.venv/bin/ruff check --no-cache .
.venv/bin/ruff format --check --no-cache .
.venv/bin/pytest --cov=agent --cov-branch --cov-report=term-missing
```

Tests use fake model responses and provider fixtures. IP socket connections are blocked.
Keep the 90% coverage requirement, including branches. Evidence admission, identity,
checkpoint compatibility and recovery limits deserve particular care.

## Changing the code

Scientific policy belongs in the stage modules. Response parsing belongs in the output
modules, transport belongs in the model adapters, and file writes use the persistence helpers.
The [architecture notes](docs/architecture.md) show where those pieces meet.

Keep the existing commands and compatibility imports working. Update the docs when a flag,
default or output changes. Commands, identifiers and URLs keep their required punctuation;
ordinary prose uses short sentences without em dashes or hyphenated compounds.

CI checks Python versions, lint, formatting, coverage and package contents. Its dependency
audit needs network access. Passing local tests does not replace that audit.

## Sharing the source

Keep generated research, environments, caches, editor state and backups out of the repository.
Include the source, tests, documentation and CI configuration when sharing changes.
