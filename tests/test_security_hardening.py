import socket

import pytest

from src.ingestion.loader import validate_public_http_url
from src.api.routes.ingest import _safe_display_filename


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://user:password@example.com/",
        "http://example.com:8080/",
    ],
)
def test_url_guard_rejects_unsafe_destinations(url):
    with pytest.raises(ValueError):
        validate_public_http_url(url)


def test_url_guard_allows_public_https_host_with_public_dns(monkeypatch):
    def fake_getaddrinfo(host, port, type):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert validate_public_http_url("https://docs.example.com/reference") == "https://docs.example.com/reference"


def test_upload_filename_never_uses_client_path():
    assert _safe_display_filename("../../private/guide.md") == "guide.md"
    assert _safe_display_filename(r"C:\private\guide.md") == "guide.md"


def test_upload_filename_rejects_empty_path():
    with pytest.raises(Exception):
        _safe_display_filename("../../")
