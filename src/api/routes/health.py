from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from src.api.schemas import HealthResponse
from src.config import settings
from src.core.vector_store import vector_store_manager

router = APIRouter(tags=["Health"])

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System health and vector index readiness",
)
async def health_check() -> HealthResponse:
    vector_count = vector_store_manager.count()

    models = {
        "gemini": settings.GEMINI_MODEL,
        "groq": settings.GROQ_MODEL,
        "openai": settings.OPENAI_MODEL,
    }
    model = models.get(settings.DEFAULT_LLM_PROVIDER.lower(), "unknown")

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        llm_provider=settings.DEFAULT_LLM_PROVIDER,
        model_name=model,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        vector_store_count=vector_count,
    )




@router.get(
    "/ready",
    summary="Check local demo readiness",
    description="Checks whether the vector index and configured inference credentials are available.",
)
async def readiness_check():
    """Report readiness separately from process liveness."""
    vector_count = vector_store_manager.count()
    provider = settings.DEFAULT_LLM_PROVIDER.lower()
    credentials = {
        "gemini": settings.GEMINI_API_KEY,
        "groq": settings.GROQ_API_KEY,
        "openai": settings.OPENAI_API_KEY,
    }
    inference_configured = bool(credentials.get(provider))
    index_ready = vector_count > 0
    ready = index_ready and inference_configured

    payload = {
        "status": "ready" if ready else "degraded",
        "mode": "local-demo",
        "checks": {
            "vector_index": "ready" if index_ready else "empty",
            "inference_credentials": "configured" if inference_configured else "missing",
        },
        "vector_store_count": vector_count,
        "llm_provider": provider,
        "note": "Credential presence is checked here; live provider connectivity is verified by an actual query.",
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)
