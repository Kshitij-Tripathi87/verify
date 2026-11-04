import ipaddress
import socket
from pathlib import Path
from typing import List
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document


MAX_REMOTE_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 3
ALLOWED_REMOTE_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
}


def validate_public_http_url(url: str) -> str:
    """
    Validate a public HTTP(S) destination before fetching it.

    This blocks common SSRF targets (loopback, private, link-local, multicast,
    reserved addresses, non-web ports, local hostnames and URL credentials).
    Production deployment must additionally enforce outbound network egress
    policy because DNS can change between validation and connection.
    """
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise ValueError("URL must be a non-empty string no longer than 2048 characters.")

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are allowed.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL must contain a hostname and must not contain credentials.")

    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise ValueError("URL contains an invalid port.") from exc

    if port not in {80, 443}:
        raise ValueError("Only standard HTTP and HTTPS ports (80 and 443) are allowed.")

    hostname = parsed.hostname.rstrip(".").lower()
    if (
        hostname in {"localhost", "metadata.google.internal"}
        or hostname.endswith((".localhost", ".local", ".internal", ".test"))
    ):
        raise ValueError("Local and internal hostnames are not allowed.")

    try:
        literal_ip = ipaddress.ip_address(hostname)
        addresses = [literal_ip]
    except ValueError:
        try:
            records = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
            addresses = [ipaddress.ip_address(record[4][0]) for record in records]
        except (OSError, ValueError) as exc:
            raise ValueError("URL hostname could not be safely resolved.") from exc

    if not addresses:
        raise ValueError("URL hostname did not resolve to an IP address.")

    if any(not address.is_global for address in addresses):
        raise ValueError("URL resolves to a non-public IP address and was blocked.")

    return url


async def _fetch_remote_text(url: str) -> tuple[str, str]:
    """Fetch a bounded text document while validating every redirect target."""
    current_url = url
    headers = {
        "User-Agent": "DocuGraphAIDemo/2.1 (+documentation importer)",
        "Accept": "text/html,application/xhtml+xml,text/plain",
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        follow_redirects=False,
        trust_env=False,
    ) as client:
        for redirect_count in range(MAX_REDIRECTS + 1):
            validate_public_http_url(current_url)

            async with client.stream("GET", current_url, headers=headers) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location or redirect_count >= MAX_REDIRECTS:
                        raise ValueError("Remote document exceeded the redirect limit.")
                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type not in ALLOWED_REMOTE_CONTENT_TYPES:
                    raise ValueError(f"Unsupported remote content type: {content_type or 'unknown'}.")

                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > MAX_REMOTE_DOCUMENT_BYTES:
                            raise ValueError("Remote document exceeds the 2 MiB size limit.")
                    except ValueError as exc:
                        if "size limit" in str(exc):
                            raise
                        raise ValueError("Remote document has an invalid Content-Length header.") from exc

                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_REMOTE_DOCUMENT_BYTES:
                        raise ValueError("Remote document exceeds the 2 MiB size limit.")

                encoding = response.encoding or "utf-8"
                return bytes(body).decode(encoding, errors="replace"), current_url

    raise ValueError("Remote document could not be fetched safely.")


class DocumentLoader:
    """
    Handles loading technical documentation from local files (.md, .txt, .html)
    or bounded, public HTTP(S) URLs.
    """

    @staticmethod
    def load_file(file_path: str | Path) -> Document:
        """Load content from a local file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix in [".md", ".markdown", ".txt", ".html", ".htm"]:
            text = path.read_text(encoding="utf-8", errors="replace")
            first_line = text.strip().split("\n")[0] if text else ""
            title = first_line.lstrip("#").strip() if first_line.startswith("#") else path.stem
            return Document(
                page_content=text,
                metadata={
                    "source": str(path.resolve()),
                    "title": title or path.name,
                    "filename": path.name,
                    "source_type": "file",
                },
            )
        raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    async def load_url(url: str) -> Document:
        """Fetch a public documentation URL with SSRF and response-size controls."""
        validate_public_http_url(url)
        html, final_url = await _fetch_remote_text(url)
        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            element.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else final_url
        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", {"role": "main"})
            or soup.body
        )
        content = main_content.get_text(separator="\n\n", strip=True) if main_content else ""
        if not content.strip():
            raise ValueError("No readable documentation text was found at this URL.")

        # Do not expose URL query strings or fragments in stored citations.
        parsed = urlsplit(final_url)
        safe_source = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))

        return Document(
            page_content=content,
            metadata={
                "source": safe_source,
                "title": title,
                "filename": Path(parsed.path).name or "webpage",
                "source_type": "url",
            },
        )

    @staticmethod
    def load_directory(directory_path: str | Path) -> List[Document]:
        """Load supported seed files from a directory."""
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return []

        docs = []
        for file_path in dir_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in [".md", ".markdown", ".txt"]:
                try:
                    docs.append(DocumentLoader.load_file(file_path))
                except Exception as exc:
                    print(f"Error loading {file_path}: {exc}")
        return docs
