import hashlib
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, List
from threading import Lock
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import FileResponse, JSONResponse
from src.config import settings

router = APIRouter(tags=["Downloads"])

# Resolve repository paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DIST_DIR = ROOT_DIR / "dist"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

EXE_PATH = DIST_DIR / "DocuGraphAI_Setup.exe"
MSI_PATH = DIST_DIR / "DocuGraphAI_Setup.msi"
ZIP_PATH = DIST_DIR / "DocuGraphAI_bundle.zip"
BAT_PATH = ROOT_DIR / "Install_DocuGraphAI.bat"
SOURCE_ZIP_PATH = DIST_DIR / "DocuGraphAI_Project_Source.zip"

_HASH_CACHE: Dict[str, str] = {}
_TELEMETRY_LOCK = Lock()

# In-memory Production Download Telemetry
_DOWNLOAD_STATS = {
    "server_start_time": datetime.now(timezone.utc).isoformat(),
    "total_downloads": 0,
    "total_bytes_served": 0,
    "downloads_by_package": {
        "project": 0,
        "exe": 0,
        "msi": 0,
        "zip": 0,
        "bat": 0,
    },
    "recent_events": []  # List of latest 25 download events
}


def record_download(package_key: str, file_size: int, client_host: str = "anonymous"):
    """Thread-safe download metrics logging."""
    with _TELEMETRY_LOCK:
        _DOWNLOAD_STATS["total_downloads"] += 1
        _DOWNLOAD_STATS["total_bytes_served"] += file_size
        if package_key in _DOWNLOAD_STATS["downloads_by_package"]:
            _DOWNLOAD_STATS["downloads_by_package"][package_key] += 1
        else:
            _DOWNLOAD_STATS["downloads_by_package"][package_key] = 1

        event = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "package": package_key,
            "size_bytes": file_size,
            "client_ip_masked": f"{client_host.split('.')[0]}.*.*.*" if "." in client_host else "client",
        }
        _DOWNLOAD_STATS["recent_events"].insert(0, event)
        if len(_DOWNLOAD_STATS["recent_events"]) > 25:
            _DOWNLOAD_STATS["recent_events"].pop()


def get_file_sha256(path: Path) -> str:
    """Computes and caches the SHA256 checksum of a file."""
    if not path.exists():
        return ""
    str_path = str(path)
    if str_path in _HASH_CACHE:
        return _HASH_CACHE[str_path]
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest().upper()
    _HASH_CACHE[str_path] = digest
    return digest


def build_project_source_archive(force: bool = False) -> Path:
    """
    Creates a clean, production-ready zip archive containing the entire project code,
    documentation, tests, and configuration (excluding secrets, venvs, and build clutter).
    """
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_ZIP_PATH.exists() and not force:
        # Check if archive was created in the last 10 minutes
        mtime = SOURCE_ZIP_PATH.stat().st_mtime
        if (time.time() - mtime) < 600:
            return SOURCE_ZIP_PATH

    ignored_dirs = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
        "dist", "build", ".gemini", ".agents", "chroma_db_store",
        "chroma_db", "uploaded", ".idea", ".vscode"
    }
    ignored_extensions = {".pyc", ".pyo", ".pyd", ".log", ".tmp"}
    ignored_files = {".env", "app_storage.db", "app_storage.db-wal", "app_storage.db-shm"}

    with zipfile.ZipFile(SOURCE_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in ROOT_DIR.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT_DIR)
            if any(part in ignored_dirs for part in rel.parts):
                continue
            if path.name in ignored_files or path.suffix.lower() in ignored_extensions:
                continue
            # Keep secrets, environment overrides, local databases and user uploads
            # out of the downloadable source bundle. Preserve the safe example file.
            if path.name.startswith(".env") and path.name != ".env.example":
                continue
            if path.name.endswith((".db", ".db-wal", ".db-shm")):
                continue
            zf.write(path, arcname=f"DocuGraphAI/{rel.as_posix()}")

    # Invalidate hash cache for source zip
    _HASH_CACHE.pop(str(SOURCE_ZIP_PATH), None)
    return SOURCE_ZIP_PATH


@router.get(
    "/download",
    include_in_schema=False,
    summary="Download portal landing page",
)
async def serve_download_portal():
    download_html = STATIC_DIR / "download.html"
    if download_html.exists():
        return FileResponse(
            path=str(download_html),
            media_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )
    return {"message": "Download portal is available. Visit /download/project, /download/exe, or /download/msi directly."}


@router.get(
    "/download/project",
    summary="Download Complete Project Source Code (.zip)",
    description="Downloads a clean, production-ready ZIP archive of the entire DocuGraph AI repository ready to run.",
)
@router.get("/download/source", include_in_schema=False)
async def download_project_source(request: Request):
    zip_file = build_project_source_archive()
    if not zip_file.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to package project source archive on server.",
        )
    file_size = zip_file.stat().st_size
    record_download("project", file_size, request.client.host if request.client else "anonymous")
    etag = f'"{get_file_sha256(zip_file)[:16]}"'

    return FileResponse(
        path=str(zip_file),
        filename="DocuGraphAI_Project_Source.zip",
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="DocuGraphAI_Project_Source.zip"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=1800",
            "ETag": etag,
        },
    )


@router.get(
    "/download/latest",
    summary="Download recommended application package",
    description="Serves the best available installer package or complete project archive.",
)
@router.get("/download/app", include_in_schema=False)
async def download_latest(request: Request):
    """
    Universal 1-click download handler:
    Serves the Windows Setup Wizard (.exe) if available, falling back to MSI or project archive.
    """
    if EXE_PATH.exists():
        return await download_exe(request)
    if MSI_PATH.exists():
        return await download_msi(request)
    return await download_project_source(request)


@router.get(
    "/download/exe",
    summary="Download Windows Setup Wizard (.exe)",
    description="Downloads the standalone Windows Setup Wizard executable for DocuGraph AI.",
)
async def download_exe(request: Request):
    if not EXE_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Windows Setup executable (.exe) is currently not available on this server.",
        )
    file_size = EXE_PATH.stat().st_size
    record_download("exe", file_size, request.client.host if request.client else "anonymous")
    etag = f'"{get_file_sha256(EXE_PATH)[:16]}"'

    return FileResponse(
        path=str(EXE_PATH),
        filename="DocuGraphAI_Setup.exe",
        media_type="application/vnd.microsoft.portable-executable",
        headers={
            "Content-Disposition": 'attachment; filename="DocuGraphAI_Setup.exe"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
            "ETag": etag,
        },
    )


@router.get(
    "/download/msi",
    summary="Download Windows Installer Package (.msi)",
    description="Downloads the standard Windows Installer MSI package for DocuGraph AI.",
)
async def download_msi(request: Request):
    if not MSI_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Windows Installer package (.msi) is currently not available on this server.",
        )
    file_size = MSI_PATH.stat().st_size
    record_download("msi", file_size, request.client.host if request.client else "anonymous")
    etag = f'"{get_file_sha256(MSI_PATH)[:16]}"'

    return FileResponse(
        path=str(MSI_PATH),
        filename="DocuGraphAI_Setup.msi",
        media_type="application/x-msi",
        headers={
            "Content-Disposition": 'attachment; filename="DocuGraphAI_Setup.msi"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
            "ETag": etag,
        },
    )


@router.get(
    "/download/zip",
    summary="Download Portable Application Archive (.zip)",
    description="Downloads the portable zip package of DocuGraph AI (no installation required).",
)
async def download_zip(request: Request):
    if not ZIP_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portable zip archive is currently not available on this server.",
        )
    file_size = ZIP_PATH.stat().st_size
    record_download("zip", file_size, request.client.host if request.client else "anonymous")
    etag = f'"{get_file_sha256(ZIP_PATH)[:16]}"'

    return FileResponse(
        path=str(ZIP_PATH),
        filename="DocuGraphAI_bundle.zip",
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="DocuGraphAI_bundle.zip"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
            "ETag": etag,
        },
    )


@router.get(
    "/download/script",
    summary="Download 1-Click Batch Installer (.bat)",
    description="Downloads the automated Windows batch setup script.",
)
async def download_script(request: Request):
    if not BAT_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Setup batch script is not available on this server.",
        )
    file_size = BAT_PATH.stat().st_size
    record_download("bat", file_size, request.client.host if request.client else "anonymous")

    return FileResponse(
        path=str(BAT_PATH),
        filename="Install_DocuGraphAI.bat",
        media_type="application/x-bat",
        headers={
            "Content-Disposition": 'attachment; filename="Install_DocuGraphAI.bat"',
            "Cache-Control": "public, max-age=3600",
        },
    )


@router.get(
    "/api/v1/download/info",
    summary="Get installer file metadata and SHA256 checksums",
    description="Returns detailed file sizes, cryptographic hashes, and download URLs for all releases.",
)
async def get_download_info() -> Dict[str, Any]:
    # Ensure project source archive exists
    build_project_source_archive()

    def file_info(path: Path, url: str, name: str, description: str, is_recommended: bool = False):
        if not path.exists():
            return {
                "name": name,
                "description": description,
                "available": False,
                "download_url": url,
            }
        size_bytes = path.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 2)
        return {
            "name": name,
            "filename": path.name,
            "description": description,
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "sha256": get_file_sha256(path),
            "download_url": url,
            "recommended": is_recommended,
            "available": True,
        }

    return {
        "app_name": "DocuGraph AI",
        "tagline": "Self-Corrective Technical Documentation Assistant",
        "version": "2.0.0",
        "release_date": "October 2026",
        "architecture": "x86_64",
        "supported_os": "Windows 10 / Windows 11 (64-bit) & Linux/macOS for Python Source",
        "packages": {
            "project": file_info(
                SOURCE_ZIP_PATH,
                "/download/project",
                "Full Project Source Code (.zip)",
                "Complete repository source code with FastAPI, LangGraph workflow, ChromaDB, and docs",
                is_recommended=True,
            ),
            "exe": file_info(
                EXE_PATH,
                "/download/exe",
                "Windows Setup Wizard (.exe)",
                "Standalone GUI installation wizard with desktop shortcut & uninstaller",
                is_recommended=False,
            ),
            "msi": file_info(
                MSI_PATH,
                "/download/msi",
                "Windows Installer Package (.msi)",
                "Standard Windows Installer MSI for enterprise & silent deployments",
                is_recommended=False,
            ),
            "zip": file_info(
                ZIP_PATH,
                "/download/zip",
                "Portable Application Archive (.zip)",
                "Self-contained desktop application bundle without installation",
                is_recommended=False,
            ),
            "bat": file_info(
                BAT_PATH,
                "/download/script",
                "1-Click Batch Installer Script (.bat)",
                "Automated PowerShell/Batch installer script",
                is_recommended=False,
            ),
        },
    }


@router.get(
    "/api/v1/download/stats",
    summary="Get live production download telemetry",
    description="Returns real-time download counts, bandwidth served, and recent distribution events.",
)
async def get_download_stats() -> Dict[str, Any]:
    with _TELEMETRY_LOCK:
        mb_served = round(_DOWNLOAD_STATS["total_bytes_served"] / (1024 * 1024), 2)
        return {
            "status": "operational",
            "server_start_time": _DOWNLOAD_STATS["server_start_time"],
            "total_downloads": _DOWNLOAD_STATS["total_downloads"],
            "total_megabytes_served": mb_served,
            "downloads_by_package": _DOWNLOAD_STATS["downloads_by_package"],
            "recent_events": _DOWNLOAD_STATS["recent_events"][:10],
        }
