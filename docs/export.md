# Making a public copy

The exporter builds a fresh source folder and a ZIP from the files listed in
`scripts/public_files.txt`. It does not use Git, contact services or remove anything from
your working project.

From the repository root, run:

```bash
python3 scripts/export_public.py
```

Use Python 3.11 or newer. The default output is a sibling folder called `public_repo`.
For another location:

```bash
python3 scripts/export_public.py --output ../public_repo_next
```

The output must be outside the source folder and must not already exist. Choose a new
folder for each export. The script will not overwrite an older copy.

## What you get

* `abstract-agent/` contains the repository files ready to upload.
* `abstract-agent.zip` contains the same folder, including `.github` and `.gitignore`.
* `scan_report.json` records the file inventory, hashes, checks and their limits.

Upload the contents of `abstract-agent/`, including the hidden files. If using the ZIP,
extract it first and upload the files inside its project folder. The ZIP itself is a
convenient copy of the source, not a replacement for the repository file tree.

The public source includes tests, CI, documentation and the export script. It excludes
Git history, environments, caches, editor state, generated results, knowledge databases,
local preferences and backup archives. New files must be added to the public file list
before they can be exported.

## Checks

The script rejects unsafe paths, symbolic links, missing files, binary files, large files
and known local artifact locations. It checks credential and personal path patterns,
Python and TOML syntax, documentation links and prose punctuation. It then compares hashes
between the source, copied folder and ZIP.

Run the checks in [Contributing](../CONTRIBUTING.md) before exporting. The exporter does not
execute tests or perform a dependency vulnerability audit. Pattern checks cannot recognise
every possible secret, so the explicit inventory is the first boundary.

Commands, identifiers, file names and URLs keep their required punctuation. Prose uses
ordinary sentences without em dashes or hyphenated compounds.
