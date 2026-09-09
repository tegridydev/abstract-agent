"""Authoritative local SQLite knowledge store with readable projections."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..application.io import atomic_write
from ..contracts.research import EvidenceCard
from .evidence import canonical_source, source_hash

SCHEMA_VERSION = 1
_FTS_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class KnowledgeStore:
    """Store reusable source records separately from untrusted generated insights."""

    def __init__(self, root: str | Path = "knowledge") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "records").mkdir(exist_ok=True)
        (self.root / "findings").mkdir(exist_ok=True)
        (self.root / "ideas").mkdir(exist_ok=True)
        self.path = self.root / "index.sqlite3"
        self.connection = sqlite3.connect(self.path, timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=10000")
        self.fts_available = True
        self._migrate()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                trust_tier TEXT NOT NULL CHECK (trust_tier = 'external_source'),
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evidence (
                paper_id TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                contract_version TEXT NOT NULL,
                extraction_mode TEXT NOT NULL,
                created_at TEXT NOT NULL,
                card_json TEXT NOT NULL,
                PRIMARY KEY (paper_id, source_hash, contract_version),
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                record_type TEXT NOT NULL,
                record_id TEXT NOT NULL,
                model TEXT NOT NULL,
                model_digest TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (record_type, record_id, model, model_digest)
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS insights (
                insight_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                trust_tier TEXT NOT NULL CHECK (trust_tier = 'generated_idea'),
                created_at TEXT NOT NULL,
                record_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts "
                "USING fts5(paper_id UNINDEXED, title, summary)"
            )
        except sqlite3.OperationalError:
            self.fts_available = False
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    def ingest_papers(self, papers: list[dict[str, Any]]) -> int:
        timestamp = _now()
        count = 0
        with self.connection:
            for raw in papers:
                paper_id = str(raw.get("paper_id") or "").strip()
                if not paper_id:
                    continue
                summary = canonical_source(str(raw.get("summary") or ""))
                title = canonical_source(str(raw.get("title") or "Untitled"))
                record = {
                    **raw,
                    "paper_id": paper_id,
                    "title": title,
                    "summary": summary,
                    "knowledge": {
                        "trust_tier": "external_source",
                        "source_content_hash": source_hash(summary),
                    },
                }
                existing = self.connection.execute(
                    "SELECT first_seen_at FROM papers WHERE paper_id = ?", (paper_id,)
                ).fetchone()
                self.connection.execute(
                    """
                    INSERT INTO papers(
                        paper_id, title, summary, source_hash, trust_tier,
                        first_seen_at, last_seen_at, record_json
                    ) VALUES (?, ?, ?, ?, 'external_source', ?, ?, ?)
                    ON CONFLICT(paper_id) DO UPDATE SET
                        title=excluded.title,
                        summary=excluded.summary,
                        source_hash=excluded.source_hash,
                        last_seen_at=excluded.last_seen_at,
                        record_json=excluded.record_json
                    """,
                    (
                        paper_id,
                        title,
                        summary,
                        source_hash(summary),
                        existing["first_seen_at"] if existing else timestamp,
                        timestamp,
                        _json(record),
                    ),
                )
                if self.fts_available:
                    self.connection.execute(
                        "DELETE FROM papers_fts WHERE paper_id = ?", (paper_id,)
                    )
                    self.connection.execute(
                        "INSERT INTO papers_fts(paper_id, title, summary) VALUES (?, ?, ?)",
                        (paper_id, title, summary),
                    )
                count += 1
        return count

    def ingest_evidence(self, ledger: dict[str, Any]) -> int:
        timestamp = _now()
        count = 0
        with self.connection:
            for raw in ledger.get("cards", []):
                card = EvidenceCard.model_validate(raw)
                provenance = card.provenance
                if (
                    provenance.extraction_mode == "legacy_model"
                    or not provenance.source_content_hash
                    or provenance.evidence_contract_version == "legacy"
                ):
                    continue
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO evidence(
                        paper_id, source_hash, contract_version, extraction_mode,
                        created_at, card_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card.paper_id,
                        provenance.source_content_hash,
                        provenance.evidence_contract_version,
                        provenance.extraction_mode,
                        timestamp,
                        _json(card.model_dump()),
                    ),
                )
                count += 1
        return count

    def get_reusable_evidence(
        self, paper_id: str, content_hash: str, contract_version: str
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT card_json FROM evidence
            WHERE paper_id = ? AND source_hash = ? AND contract_version = ?
            """,
            (paper_id, content_hash, contract_version),
        ).fetchone()
        return json.loads(row["card_json"]) if row else None

    def search_papers(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("knowledge search limit must be between 1 and 100")
        tokens = _FTS_TOKEN.findall(query)
        rows: list[sqlite3.Row]
        if self.fts_available and tokens:
            expression = " OR ".join(f'"{token}"' for token in tokens[:20])
            rows = self.connection.execute(
                """
                SELECT papers.record_json
                FROM papers_fts JOIN papers USING(paper_id)
                WHERE papers_fts MATCH ? ORDER BY bm25(papers_fts) LIMIT ?
                """,
                (expression, limit),
            ).fetchall()
        else:
            pattern = f"%{query.strip()}%"
            rows = self.connection.execute(
                """
                SELECT record_json FROM papers
                WHERE title LIKE ? OR summary LIKE ?
                ORDER BY last_seen_at DESC LIMIT ?
                """,
                (pattern, pattern, limit),
            ).fetchall()
        return [json.loads(row["record_json"]) for row in rows]

    def related_idea_queries(self, query: str, limit: int = 2) -> list[str]:
        """Derive bounded discovery only query expansions from generated ideas."""
        query_tokens = [token.casefold() for token in _FTS_TOKEN.findall(query)]
        if not query_tokens or limit < 1:
            return []
        rows = self.connection.execute(
            "SELECT record_json FROM insights ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        stop_words = {
            "about",
            "after",
            "before",
            "could",
            "from",
            "into",
            "model",
            "paper",
            "research",
            "that",
            "their",
            "this",
            "using",
            "with",
        }
        candidates: list[tuple[int, str]] = []
        anchors = set(query_tokens)
        for row in rows:
            record = json.loads(row["record_json"])
            text = _json(record.get("content", ""))
            tokens = [token.casefold() for token in _FTS_TOKEN.findall(text)]
            overlap = len(anchors & set(tokens))
            if overlap < 1:
                continue
            additions = [
                token
                for token in tokens
                if token not in anchors and token not in stop_words and len(token) > 2
            ]
            expanded = list(dict.fromkeys([*query_tokens, *additions]))[:10]
            if len(expanded) > len(set(query_tokens)):
                candidates.append((overlap, " ".join(expanded)))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return list(dict.fromkeys(value for _, value in candidates))[:limit]

    def reindex(self) -> int:
        if not self.fts_available:
            return 0
        rows = self.connection.execute("SELECT paper_id, title, summary FROM papers").fetchall()
        with self.connection:
            self.connection.execute("DELETE FROM papers_fts")
            self.connection.executemany(
                "INSERT INTO papers_fts(paper_id, title, summary) VALUES (?, ?, ?)",
                [(row["paper_id"], row["title"], row["summary"]) for row in rows],
            )
        return len(rows)

    def ingest_run(
        self,
        topic: str,
        context: dict[str, Any],
        *,
        status: str = "complete",
        run_id: str | None = None,
    ) -> str:
        identifier = run_id or (
            f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )
        timestamp = _now()
        final_keys = (
            "selected_candidate",
            "research_proposal",
            "hypothesis_candidates",
            "novelty_verifications",
            "gap_map",
        )
        record = {
            "run_id": identifier,
            "topic": topic,
            "status": status,
            "created_at": timestamp,
            "paper_ids": [
                paper.get("paper_id")
                for paper in context.get("top_papers", [])
                if paper.get("paper_id")
            ],
            "outputs": {key: context[key] for key in final_keys if key in context},
        }
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
                (identifier, topic, status, timestamp, _json(record)),
            )
            for key, value in record["outputs"].items():
                insight_id = f"{identifier}:{key}"
                insight = {
                    "insight_id": insight_id,
                    "run_id": identifier,
                    "kind": key,
                    "trust_tier": "generated_idea",
                    "created_at": timestamp,
                    "content": value,
                    "allowed_uses": ["brainstorming", "query_expansion", "related_recall"],
                    "prohibited_uses": ["external_evidence", "novelty_proof"],
                }
                self.connection.execute(
                    "INSERT OR REPLACE INTO insights VALUES (?, ?, ?, 'generated_idea', ?, ?)",
                    (insight_id, identifier, key, timestamp, _json(insight)),
                )
        self._write_run_markdown(identifier, topic, context, status)
        self.export_projections()
        return identifier

    def _write_run_markdown(
        self, run_id: str, topic: str, context: dict[str, Any], status: str
    ) -> None:
        cards = context.get("evidence_ledger", {}).get("cards", [])
        finding_lines = [f"# Findings: {topic}", "", f"Status: `{status}`", ""]
        for card in cards:
            finding_lines.append(f"## {card.get('paper_id', 'unknown paper')}")
            finding_lines.append("")
            for finding in card.get("findings", []):
                finding_lines.append(f"- {finding.get('source_passage', '')}")
            finding_lines.append("")
        atomic_write(self.root / "findings" / f"{run_id}.md", "\n".join(finding_lines))

        idea_lines = [f"# Ideas: {topic}", "", "Trust tier: `generated_idea`", ""]
        for key in ("selected_candidate", "research_proposal", "hypothesis_candidates"):
            if key in context:
                idea_lines.extend([f"## {key.replace('_', ' ').title()}", "", "```json"])
                idea_lines.append(
                    json.dumps(context[key], ensure_ascii=False, indent=2, default=str)
                )
                idea_lines.extend(["```", ""])
        atomic_write(self.root / "ideas" / f"{run_id}.md", "\n".join(idea_lines))

    def export_projections(self) -> dict[str, int]:
        mappings = {
            "papers.jsonl": ("papers", "record_json", "paper_id"),
            "evidence.jsonl": ("evidence", "card_json", "paper_id"),
            "insights.jsonl": ("insights", "record_json", "insight_id"),
            "runs.jsonl": ("runs", "record_json", "run_id"),
        }
        counts: dict[str, int] = {}
        for filename, (table, column, order) in mappings.items():
            rows = self.connection.execute(
                f"SELECT {column} FROM {table} ORDER BY {order}"  # noqa: S608 - fixed mapping
            ).fetchall()
            content = "\n".join(row[column] for row in rows)
            if content:
                content += "\n"
            atomic_write(self.root / "records" / filename, content)
            counts[table] = len(rows)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _now(),
            "database": "index.sqlite3",
            "fts5": self.fts_available,
            "counts": counts,
            "authority": "SQLite is authoritative; JSONL and Markdown are projections.",
        }
        atomic_write(self.root / "manifest.json", json.dumps(manifest, indent=2))
        atomic_write(
            self.root / "README.md",
            "# Local research knowledge\n\n"
            "This directory is generated and local only. `index.sqlite3` is authoritative; "
            "files under `records/`, `findings/`, and `ideas/` are readable projections.\n\n"
            "Generated ideas are never accepted as external evidence or proof of novelty.\n",
        )
        return counts

    def import_projections(self, source: str | Path) -> dict[str, int]:
        """Import trusted paper/evidence JSONL exported by another local store."""
        source_path = Path(source)
        records = source_path / "records" if (source_path / "records").is_dir() else source_path
        papers: list[dict[str, Any]] = []
        paper_file = records / "papers.jsonl"
        if paper_file.exists():
            for line in paper_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    knowledge = record.pop("knowledge", {})
                    if knowledge.get("trust_tier", "external_source") != "external_source":
                        continue
                    papers.append(record)
        paper_count = self.ingest_papers(papers)
        cards = []
        evidence_file = records / "evidence.jsonl"
        if evidence_file.exists():
            for line in evidence_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    cards.append(json.loads(line))
        evidence_count = self.ingest_evidence({"cards": cards})
        self.export_projections()
        return {"papers": paper_count, "evidence": evidence_count}

    def status(self) -> dict[str, Any]:
        counts = {
            table: self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("papers", "evidence", "runs", "insights", "embeddings")
        }
        return {
            "path": str(self.path),
            "schema_version": SCHEMA_VERSION,
            "fts5": self.fts_available,
            "counts": counts,
        }

    def close(self) -> None:
        self.connection.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter cleanup safety net
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()

    def __enter__(self) -> KnowledgeStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
