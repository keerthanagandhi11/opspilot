"""Application settings, read from environment variables (or a .env file)."""

import re
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (any OpenAI-compatible chat API; Groq free tier by default) ---
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 1500  # room for reasoning models (e.g. Gemini flash) that think before answering

    # --- Agent (agentic RAG) ---
    agent_enabled: bool = True  # false = single-shot RAG
    agent_max_steps: int = 3  # tool-calling rounds before the agent must answer (keeps free-tier token use low)
    agent_verify: bool = True  # self-check the answer against its sources

    # --- Retrieval ---
    enable_embeddings: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_chars: int = 900
    chunk_overlap: int = 150
    top_k: int = 5

    # --- Limits (keep the free host healthy) ---
    max_files: int = 10
    max_file_mb: float = 15.0
    max_pages_per_file: int = 300
    max_question_chars: int = 500
    session_ttl_minutes: int = 60
    max_sessions: int = 200
    rate_limit_per_minute: int = 30
    trust_proxy: bool = False  # true behind Hugging Face / Render, so X-Forwarded-For is used
    proxy_hops: int = 1  # how many trusted proxies append to X-Forwarded-For

    # --- Server ---
    cors_origins: str = ""  # comma separated; empty = same-origin only
    log_level: str = "INFO"

    @field_validator("llm_api_key", "llm_model", "llm_base_url", mode="before")
    @classmethod
    def _clean_pasted_value(cls, value, info):
        """Keys pasted into a host's dashboard often pick up a trailing space or newline,
        surrounding quotes, or the "NAME=" prefix copied from .env; any of these makes the
        provider reject the key (HTTP 401)."""
        if not isinstance(value, str):
            return value
        value = value.strip()
        value = re.sub(rf"^{info.field_name}\s*=\s*", "", value, flags=re.IGNORECASE)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()
        return value

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key.strip())

    @property
    def max_file_bytes(self) -> int:
        return int(self.max_file_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
