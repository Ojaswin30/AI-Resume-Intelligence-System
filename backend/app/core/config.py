from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "Local SLM Resume Intelligence Engine"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "127.0.0.1"

    # LLM Settings (Ollama / Local OpenAI compatible)
    LLM_PROVIDER: str = "ollama"  # or "openai_compatible", "mock"
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:3b"

    # Embedding model name for local CPU execution
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
