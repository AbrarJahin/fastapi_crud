from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl, Field

class Settings(BaseSettings):
    """Centralized configuration loaded from environment variables and .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = Field(default="sqlite:///./app.sqlite3", validation_alias="DATABASE_URL")

    # Ollama
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_embed_model: str = Field(default="bge-m3:latest", validation_alias="OLLAMA_EMBED_MODEL")
    ollama_chat_model: str = Field(default="qwen2.5:14b-instruct", validation_alias="OLLAMA_CHAT_MODEL")
    ollama_translate_model: str = Field(default="qwen2.5:14b-instruct", validation_alias="OLLAMA_TRANSLATE_MODEL")

settings = Settings()
