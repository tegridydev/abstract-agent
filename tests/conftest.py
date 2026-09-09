"""Suite-wide safeguards for deterministic, offline tests."""

from __future__ import annotations

import socket
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast if a test accidentally attempts an IP network connection."""
    original_connect = socket.socket.connect

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        if sock.family == socket.AF_UNIX:
            return original_connect(sock, address)
        raise AssertionError(f"unexpected network connection attempted: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
