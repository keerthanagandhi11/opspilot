"""Request/response models (also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FileOut(BaseModel):
    name: str
    pages: int
    size_kb: int
    has_text: bool
    sample: bool


class SessionOut(BaseModel):
    session_id: str
    files: list[FileOut]
    search_mode: str = Field(description="'hybrid' (keywords + embeddings) or 'keyword'")
    warnings: list[str] = []


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceOut(BaseModel):
    file: str
    page: int
    label: str
    snippet: str


class StepOut(BaseModel):
    kind: str = Field(description="search | read | timeline | list | check | info")
    title: str
    detail: str = ""


class AnswerOut(BaseModel):
    answer: str
    found: bool
    mode: str = Field(description="'agent', 'llm' (single-shot RAG) or 'extractive' (quote mode)")
    sources: list[SourceOut]
    notice: str | None = None
    steps: list[StepOut] = Field(default=[], description="What the agent did, in order")


class SummaryOut(BaseModel):
    what_happened: str
    impact: str
    status: str
    missing: list[str]
    mode: str


class ConfigOut(BaseModel):
    llm_enabled: bool
    llm_model: str | None
    agent_enabled: bool
    embeddings_enabled: bool
    max_files: int
    max_file_mb: float
    version: str
