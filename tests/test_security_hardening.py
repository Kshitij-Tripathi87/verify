import socket
import zipfile

import pytest

from src.ingestion.loader import validate_public_http_url
from src.api.routes.ingest import _safe_display_filename
from src.api.routes import download


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
    from fastapi import HTTPException\n\n    with pytest.raises(HTTPException):
        _safe_display_filename("../../")


def test_source_archive_excludes_local_secrets_and_user_data(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('demo')", encoding="utf-8")
    (root / "README.md").write_text("demo", encoding="utf-8")
    (root / ".env.example").write_text("SAFE=example", encoding="utf-8")
    (root / ".env.production").write_text("API_KEY=secret", encoding="utf-8")
    (root / "data" / "chroma_db").mkdir(parents=True)
    (root / "data" / "chroma_db" / "index.bin").write_bytes(b"private vectors")
    (root / "data" / "raw_documents" / "uploaded").mkdir(parents=True)
    (root / "data" / "raw_documents" / "uploaded" / "private.md").write_text("private upload", encoding="utf-8")
    (root / "data" / "app_storage.db").write_bytes(b"private db")

    archive_dir = root / "dist"
    archive_path = archive_dir / "DocuGraphAI_Project_Source.zip"
    monkeypatch.setattr(download, "ROOT_DIR", root)
    monkeypatch.setattr(download, "DIST_DIR", archive_dir)
    monkeypatch.setattr(download, "SOURCE_ZIP_PATH", archive_path)

    result = download.build_project_source_archive(force=True)

    with zipfile.ZipFile(result) as bundle:
        names = set(bundle.namelist())

    assert "DocuGraphAI/README.md" in names
    assert "DocuGraphAI/.env.example" in names
    assert "DocuGraphAI/src/app.py" in names
    assert not any(".env.production" in name for name in names)
    assert not any("chroma_db" in name for name in names)
    assert not any("uploaded/private.md" in name for name in names)
    assert not any(name.endswith(".db") for name in names)
