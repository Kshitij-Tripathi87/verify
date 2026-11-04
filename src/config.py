import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "Technical Documentation RAG Assistant"
    APP_ENV: str = "development"
    PORT: int = 8000
    HOST: str = "127.0.0.1"
    DEBUG: bool = False
    CORS_ORIGINS: str = "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:3000,http://localhost:3000"

    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = Field(default="gemini", description="LLM Provider: gemini, groq, openai, anthropic")
    GROQ_API_KEY: str = Field(default="")
    OPENAI_API_KEY: str = Field(default="")
    GEMINI_API_KEY: str = Field(default="")
    ANTHROPIC_API_KEY: str = Field(default="")

    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    GEMINI_MODEL: str = Field(default="gemini-3.6-flash")

    # Embedding Settings
    EMBEDDING_PROVIDER: str = Field(default="fastembed", description="fastembed, huggingface, openai")
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Search Fallback Settings
    SEARCH_PROVIDER: str = Field(default="duckduckgo", description="duckduckgo, tavily")
    TAVILY_API_KEY: str = Field(default="")

    # Self-Correction Parameters
    TOP_K_RETRIEVAL: int = 4
    MAX_RETRIES: int = 2

    # Storage Paths
    CHROMA_PERSIST_DIRECTORY: str = str(BASE_DIR / "data" / "chroma_db")
    SQLITE_DB_PATH: str = str(BASE_DIR / "data" / "app_storage.db")
    DOCS_DIRECTORY: str = str(BASE_DIR / "data" / "raw_documents")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(Path(settings.SQLITE_DB_PATH).parent, exist_ok=True)
os.makedirs(settings.DOCS_DIRECTORY, exist_ok=True)
