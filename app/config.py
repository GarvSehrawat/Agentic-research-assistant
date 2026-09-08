"""Application configuration loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class Config:
    """Central configuration object. All values come from environment
    variables so no secrets are ever hardcoded in source."""

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    # Web search
    WEB_SEARCH_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "tavily")
    WEB_SEARCH_API_KEY: str | None = os.getenv("WEB_SEARCH_API_KEY")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/app.db")

    # Retrieval
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "indexes/faiss")
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "data/uploads")

    # Safety / agent limits
    MAX_AGENT_ITERATIONS: int = _int_env("MAX_AGENT_ITERATIONS", 3)
    MAX_TOOL_CALLS: int = _int_env("MAX_TOOL_CALLS", 20)
    MAX_TASKS: int = _int_env("MAX_TASKS", 15)
    MAX_FILE_SIZE_MB: int = _int_env("MAX_FILE_SIZE_MB", 20)
    MAX_QUERY_LENGTH: int = _int_env("MAX_QUERY_LENGTH", 2000)

    ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}

    TESTING: bool = False

    @classmethod
    def as_dict(cls) -> dict:
        return {
            k: v
            for k, v in vars(cls).items()
            if not k.startswith("_") and not callable(v)
        }
