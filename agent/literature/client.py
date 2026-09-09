"""Shared HTTP transport and provider aware retry handling."""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .. import __version__


@dataclass(frozen=True)
class SourceSettings:
    contact_email: str | None
    semantic_scholar_api_key: str | None
    openalex_api_key: str | None
    ncbi_api_key: str | None
    connect_timeout: float = 3.05
    read_timeout: float = 20.0

    @classmethod
    def from_env(cls) -> SourceSettings:
        return cls(
            contact_email=os.getenv("ABSTRACT_AGENT_CONTACT_EMAIL"),
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            openalex_api_key=os.getenv("OPENALEX_API_KEY"),
            ncbi_api_key=os.getenv("NCBI_API_KEY"),
        )


def _retry_delay(value: str | None, fallback: float, *, maximum: float = 30.0) -> float:
    """Parse delta seconds, epoch seconds, HTTP dates, and ISO provider reset timestamps."""
    if not value:
        return fallback
    now = dt.datetime.now(dt.UTC)
    try:
        numeric = float(value)
    except ValueError:
        parsed: dt.datetime | None = None
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            with_iso_zone = value[:-1] + "+00:00" if value.endswith("Z") else value
            with suppress(ValueError):
                parsed = dt.datetime.fromisoformat(with_iso_zone)
        if parsed is None:
            return fallback
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        delay = (parsed.astimezone(dt.UTC) - now).total_seconds()
    else:
        delay = numeric - now.timestamp() if numeric > 1_000_000_000 else numeric
    return min(max(delay, 0.0), maximum)


class ScholarlyClient:
    """Shared, retrying HTTP client for safe scholarly GET requests."""

    def __init__(
        self,
        settings: SourceSettings | None = None,
        *,
        session: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or SourceSettings.from_env()
        self.rate_limit: dict[str, str] = {}
        self.retry_events: list[dict[str, str | float]] = []
        self._sleep = sleep
        self.session = session or httpx.Client(
            headers={
                "User-Agent": f"AbstractAgent/{__version__}"
                + (
                    f" (mailto:{self.settings.contact_email})"
                    if self.settings.contact_email
                    else ""
                )
            },
            timeout=httpx.Timeout(
                connect=self.settings.connect_timeout,
                read=self.settings.read_timeout,
                write=self.settings.read_timeout,
                pool=self.settings.connect_timeout,
            ),
            follow_redirects=True,
        )

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        retry = kwargs.pop("retry", True)
        attempts = 4 if retry else 1
        for attempt in range(attempts):
            response: httpx.Response | None = None
            try:
                response = self.session.get(url, **kwargs)
                self.rate_limit.update(
                    {
                        key: value
                        for key, value in response.headers.items()
                        if key.casefold().startswith("x-ratelimit")
                        or key.casefold()
                        in {
                            "retry-after",
                            "x-rate-limit-limit",
                            "x-rate-limit-interval",
                            "x-rate-limit-reset",
                        }
                    }
                )
                response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                retryable_status = isinstance(error, httpx.HTTPStatusError) and (
                    error.response.status_code == 429 or error.response.status_code >= 500
                )
                if attempt == attempts - 1 or (
                    isinstance(error, httpx.HTTPStatusError) and not retryable_status
                ):
                    raise
                headers = response.headers if response else {}
                reset = next(
                    (
                        headers.get(name)
                        for name in (
                            "retry-after",
                            "x-ratelimit-retry-after",
                            "x-rate-limit-reset",
                            "x-ratelimit-reset",
                        )
                        if headers.get(name)
                    ),
                    None,
                )
                delay = _retry_delay(reset, 0.5 * 2**attempt)
                category = (
                    f"http_{error.response.status_code}"
                    if isinstance(error, httpx.HTTPStatusError)
                    else "timeout"
                    if isinstance(error, httpx.TimeoutException)
                    else "network_error"
                )
                self.retry_events.append({"category": category, "delay_seconds": round(delay, 3)})
                self._sleep(delay)
        raise RuntimeError("unreachable HTTP retry state")

    def close(self) -> None:
        self.session.close()
