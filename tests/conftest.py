"""Shared test fixtures."""

import socket

import pytest


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch):
    """Force ``socket.getaddrinfo`` to resolve every hostname to a public IP.

    Keeps the test suite hermetic: SSRF host validation happens for every URL
    fetch, and we don't want tests reaching real DNS for ``example.com``.
    Tests that specifically exercise resolution behavior override this fixture
    via their own ``monkeypatch`` of ``socket.getaddrinfo``.
    """

    def fake_getaddrinfo(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
