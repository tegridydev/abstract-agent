"""SQLite backed literature response cache."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

Paper = dict[str, Any]


class SourceCache:
    VERSION = "4"

    def __init__(self, path: str | Path | None = None, ttl_seconds: int = 86_400) -> None:
        if ttl_seconds < 0:
            raise ValueError("cache TTL must be non negative")
        configured = path or os.getenv("ABSTRACT_AGENT_CACHE_PATH")
        self.path = Path(configured or ".cache/abstract-agent/sources.sqlite3")
        self.ttl_seconds = ttl_seconds

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS source_cache (
                provider TEXT NOT NULL,
                query_key TEXT NOT NULL,
                fetched_at REAL NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (provider, query_key)
            )
            """
        )
        return connection

    @staticmethod
    def query_key(topic: str, max_results: int) -> str:
        raw = f"{SourceCache.VERSION}|{' '.join(topic.casefold().split())}|{max_results}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, provider: str, topic: str, max_results: int) -> tuple[list[Paper], bool] | None:
        if not self.path.exists():
            return None
        key = self.query_key(topic, max_results)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT fetched_at, payload FROM source_cache WHERE provider=? AND query_key=?",
                (provider, key),
            ).fetchone()
        if not row:
            return None
        fetched_at, payload = row
        fresh = time.time() - float(fetched_at) <= self.ttl_seconds
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return (decoded, fresh) if isinstance(decoded, list) else None

    def put(self, provider: str, topic: str, max_results: int, papers: list[Paper]) -> None:
        key = self.query_key(topic, max_results)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO source_cache(provider, query_key, fetched_at, payload)
                VALUES (?, ?, ?, ?)
                """,
                (provider, key, time.time(), json.dumps(papers, default=str)),
            )
            connection.commit()
