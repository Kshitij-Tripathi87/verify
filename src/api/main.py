from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.config import settings
from src.storage.database import init_db
from src.core.vector_store import vector_store_manager
from src.ingestion.indexer import seed_default_corpus
from src.api.routes import query, ingest, documents, feedback, health, download

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing SQLite database...")
    init_db()

    current_count = vector_store_manager.count()
    print(f"[Startup] Vector store contains {current_count} chunk(s).")
    if current_count == 0:
        print("[Startup] Vector store is empty. Ingesting seed documentation...")
        try:
            seed_default_corpus()
        except Exception as e:
            print(f"[Startup] Warning: Automatic seed ingestion error: {e}")

    yield
    print("[Shutdown] Cleaning up resources...")

app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    description="Self-Corrective RAG Technical Documentation Assistant using LangGraph and FastAPI.",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_demo_security_headers(request, call_next):
    """Add low-risk browser security headers to API and static responses."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


# Include API Routers with /api/v1 prefix (standard versioned endpoints)
app.include_router(query.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
app.include_router(download.router, prefix="/api/v1")

# Also include API Routers at root level (as specified in assignment PRD: /query, /ingest, /documents, /feedback, /health, /download)
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(documents.router)
app.include_router(feedback.router)
app.include_router(health.router)
app.include_router(download.router)

# Mount Static UI directory
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Welcome to RAG Technical Documentation Assistant API. Visit /docs for Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.HOST, port=settings.PORT, reload=True)
