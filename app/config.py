from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration loaded once from env + .env, with normalization."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------
    # Database
    # -------------------------
    database_url: str = Field(default="sqlite:///./app.sqlite3", validation_alias="DATABASE_URL")

    # -------------------------
    # Ollama (raw)
    # -------------------------
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", validation_alias="OLLAMA_BASE_URL")
    ollama_embed_model: str = Field(default="bge-m3:latest", validation_alias="OLLAMA_EMBED_MODEL")
    ollama_chat_model: str = Field(default="qwen2.5:14b-instruct", validation_alias="OLLAMA_CHAT_MODEL")
    ollama_translate_model: str = Field(default="qwen2.5:14b-instruct", validation_alias="OLLAMA_TRANSLATE_MODEL")

    # -------------------------
    # Ask-Web (DuckDuckGo + fetch) tuning
    # -------------------------
    ask_web_fetch_timeout_s: float = Field(default=8.0, validation_alias="ASK_WEB_FETCH_TIMEOUT_S")
    ask_web_user_agent: str = Field(default="fastapi-agent/1.0 (+local)", validation_alias="ASK_WEB_USER_AGENT")
    ask_web_max_page_bytes: int = Field(default=1_000_000, validation_alias="ASK_WEB_MAX_PAGE_BYTES")
    ask_web_fetch_concurrency: int = Field(default=3, validation_alias="ASK_WEB_FETCH_CONCURRENCY")

    # ---------- Normalized / derived ----------
    ollama_base_url_norm: str = ""  # computed after init

    def model_post_init(self, __context: Any) -> None:
        # Normalize URL (strip, remove trailing slash, remove accidental /api suffix)
        base = (self.ollama_base_url or "").strip().rstrip("/")
        if base.endswith("/api"):
            base = base[:-4].rstrip("/")
        self.ollama_base_url_norm = base

        # Normalize model names (strip)
        self.ollama_embed_model = (self.ollama_embed_model or "").strip()
        self.ollama_chat_model = (self.ollama_chat_model or "").strip()
        self.ollama_translate_model = (self.ollama_translate_model or "").strip()

        # Normalize ask-web settings
        self.ask_web_user_agent = (self.ask_web_user_agent or "").strip()

        # -------------------------
        # Basic validations (fail fast at startup)
        # -------------------------
        if not (self.database_url or "").strip():
            raise ValueError("DATABASE_URL is empty.")

        if not self.ollama_base_url_norm:
            raise ValueError("OLLAMA_BASE_URL is empty after normalization.")
        if not self.ollama_embed_model:
            raise ValueError("OLLAMA_EMBED_MODEL is empty.")
        if not self.ollama_chat_model:
            raise ValueError("OLLAMA_CHAT_MODEL is empty.")
        if not self.ollama_translate_model:
            raise ValueError("OLLAMA_TRANSLATE_MODEL is empty.")

        # Ask-web validations
        if self.ask_web_fetch_timeout_s <= 0:
            raise ValueError("ASK_WEB_FETCH_TIMEOUT_S must be > 0.")
        if not self.ask_web_user_agent:
            raise ValueError("ASK_WEB_USER_AGENT is empty.")
        if self.ask_web_max_page_bytes < 100_000:
            raise ValueError("ASK_WEB_MAX_PAGE_BYTES is too small (min recommended ~100000).")
        if not (1 <= self.ask_web_fetch_concurrency <= 10):
            raise ValueError("ASK_WEB_FETCH_CONCURRENCY must be between 1 and 10.")

    def ollama_url(self, path: str) -> str:
        p = path.strip()
        if not p.startswith("/"):
            p = "/" + p
        return urljoin(self.ollama_base_url_norm + "/", p.lstrip("/"))


settings = Settings()
