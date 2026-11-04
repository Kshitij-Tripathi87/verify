import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Request
from langchain_core.documents import Document
from src.api.schemas import IngestRequest, IngestResponse
from src.ingestion.loader import DocumentLoader
from src.ingestion.indexer import index_documents
from src.storage.repository import storage_repo
from src.config import settings

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

async def _process_urls(urls: List[str]) -> IngestResponse:
    if not urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No URLs provided for ingestion."
        )

    loaded_docs: List[Document] = []
    for url in urls:
        try:
            doc = await DocumentLoader.load_url(url)
            loaded_docs.append(doc)
        except Exception as e:
            print(f"[Ingest URL] Failed to load {url}: {e}")

    if not loaded_docs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract content from the provided URLs."
        )

    total_chunks = index_documents(loaded_docs)

    for doc in loaded_docs:
        storage_repo.record_document(
            source=doc.metadata.get("source", ""),
            title=doc.metadata.get("title", "Web Document"),
            chunk_count=total_chunks // len(loaded_docs),
            source_type="url",
        )

    return IngestResponse(
        status="success",
        indexed_documents=len(loaded_docs),
        total_chunks=total_chunks,
        message=f"Successfully indexed {len(loaded_docs)} document(s) into {total_chunks} chunks."
    )

@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest new documents from URLs or file uploads",
    description="Crawls and indexes documents from JSON URLs or multipart form-data file uploads."
)
async def ingest_documents_endpoint(request: Request) -> IngestResponse:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_files = form.getlist("files") or form.getlist("file")
        valid_files = [f for f in uploaded_files if hasattr(f, "filename") and f.filename]
        if valid_files:
            return await ingest_files(valid_files)
        
        url_vals = form.getlist("urls") or form.getlist("url")
        extracted_urls = []
        for val in url_vals:
            if isinstance(val, str):
                extracted_urls.extend([u.strip() for u in val.split(",") if u.strip()])
        if extracted_urls:
            return await _process_urls(extracted_urls)
            
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files or URLs found in multipart form data."
        )

    # Standard JSON body
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload provided."
        )

    try:
        payload = IngestRequest(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {e}"
        )

    if not payload.urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No URLs provided for ingestion."
        )

    return await _process_urls(payload.urls)

ALLOWED_UPLOAD_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_FILES = 10


def _safe_display_filename(raw_filename: str) -> str:
    """Keep only a display basename; never use the client path on disk."""
    normalized = (raw_filename or "").replace("\\", "/")
    display_name = Path(normalized).name.strip()
    if not display_name or display_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Uploaded file has an invalid filename.")
    return display_name[:180]


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest technical documentation files",
    description="Upload bounded Markdown, text, or HTML documentation files.",
)
async def ingest_files(files: List[UploadFile] = File(...)) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail=f"A maximum of {MAX_UPLOAD_FILES} files may be uploaded at once.")

    upload_dir = Path(settings.DOCS_DIRECTORY) / "uploaded"
    upload_dir.mkdir(parents=True, exist_ok=True)

    loaded_docs: List[Document] = []
    stored_paths: List[Path] = []
    total_bytes = 0

    try:
        for uploaded_file in files:
            display_name = _safe_display_filename(uploaded_file.filename or "")
            extension = Path(display_name).suffix.lower()
            if extension not in ALLOWED_UPLOAD_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported file type for '{display_name}'. Allowed: .md, .markdown, .txt, .html, .htm.",
                )

            storage_name = f"{uuid.uuid4().hex}{extension}"
            file_path = upload_dir / storage_name
            file_size = 0

            with file_path.open("wb") as output:
                while True:
                    chunk = await uploaded_file.read(64 * 1024)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    total_bytes += len(chunk)
                    if file_size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"'{display_name}' exceeds the 5 MiB per-file limit.",
                        )
                    if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="The combined upload exceeds the 10 MiB request limit.",
                        )
                    output.write(chunk)

            if file_size == 0:
                raise HTTPException(status_code=400, detail=f"'{display_name}' is empty.")

            stored_paths.append(file_path)
            doc = DocumentLoader.load_file(file_path)
            doc.metadata["source"] = f"upload:{storage_name}"
            doc.metadata["filename"] = display_name
            if doc.metadata.get("title") == file_path.stem:
                doc.metadata["title"] = Path(display_name).stem
            loaded_docs.append(doc)

        if not loaded_docs:
            raise HTTPException(status_code=422, detail="No supported documentation files were provided.")

        total_chunks = index_documents(loaded_docs)

        for doc in loaded_docs:
            storage_repo.record_document(
                source=doc.metadata.get("source", ""),
                title=doc.metadata.get("title", "Uploaded Document"),
                chunk_count=total_chunks // len(loaded_docs),
                source_type="file",
            )

        return IngestResponse(
            status="success",
            indexed_documents=len(loaded_docs),
            total_chunks=total_chunks,
            message=f"Successfully uploaded {len(loaded_docs)} document(s) into {total_chunks} chunks.",
        )
    except HTTPException:
        for path in stored_paths:
            path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        for path in stored_paths:
            path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not ingest uploaded documents: {type(exc).__name__}.") from exc
    finally:
        for uploaded_file in files:
            await uploaded_file.close()
