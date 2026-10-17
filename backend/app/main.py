"""OpsPilot: FastAPI REST API + Gradio UI (mounted at "/")."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .agent import ask_once
from .config import Settings, get_settings
from .llm import LLMClient
from .pipeline import InputError, build_session
from .rag import summarize
from .rate_limit import RateLimited, RateLimiter
from .retrieval import Embedder
from .schemas import AnswerOut, AskIn, ConfigOut, FileOut, SessionOut, SourceOut, StepOut, SummaryOut
from .sessions import SessionStore
from .ui import mount

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("opspilot")

store = SessionStore(settings.session_ttl_minutes * 60, settings.max_sessions)
limiter = RateLimiter(settings.rate_limit_per_minute)
llm = LLMClient(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info(
        "OpsPilot %s starting | LLM: %s | agent: %s | embeddings: %s",
        __version__,
        settings.llm_model if settings.llm_enabled else "off (quote mode)",
        settings.agent_enabled and settings.llm_enabled,
        settings.enable_embeddings,
    )
    if settings.enable_embeddings:  # warm the model in the background
        threading.Thread(target=Embedder.get, args=(settings.embedding_model,), daemon=True).start()
    yield


app = FastAPI(
    title="OpsPilot API",
    version=__version__,
    lifespan=lifespan,
    description="Agentic RAG over incident PDFs. The Gradio UI is served at `/`.",
)

if settings.cors_origins.strip():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def rate_limited(request: Request) -> None:
    try:
        limiter.check(request)
    except RateLimited as exc:
        raise HTTPException(429, exc.message) from exc


def _settings() -> Settings:
    return settings


# ------------------------------------------------------------------ REST API


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "sessions": len(store)}


@app.get("/api/config", response_model=ConfigOut)
def config(s: Settings = Depends(_settings)) -> ConfigOut:
    return ConfigOut(
        llm_enabled=s.llm_enabled,
        llm_model=s.llm_model if s.llm_enabled else None,
        agent_enabled=s.llm_enabled and s.agent_enabled,
        embeddings_enabled=s.enable_embeddings,
        max_files=s.max_files,
        max_file_mb=s.max_file_mb,
        version=__version__,
    )


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    buf = bytearray()
    while chunk := await upload.read(1024 * 1024):
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(413, f"{upload.filename} is larger than {settings.max_file_mb:g} MB.")
    return bytes(buf)


@app.post("/api/sessions", response_model=SessionOut, dependencies=[Depends(rate_limited)])
async def create_session(
    files: list[UploadFile] = File(default=[]),
    include_sample: bool = Form(default=False),
) -> SessionOut:
    files = [f for f in files if f.filename]
    if len(files) > settings.max_files:
        raise HTTPException(400, f"Upload at most {settings.max_files} PDFs at a time.")
    uploads = [(f.filename or "upload.pdf", await _read_limited(f, settings.max_file_bytes)) for f in files]
    try:
        session, warnings = await run_in_threadpool(build_session, uploads, include_sample, settings, store)
    except InputError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return SessionOut(
        session_id=session.id,
        files=[
            FileOut(
                name=i.name,
                pages=i.pages,
                size_kb=max(1, round(i.size_bytes / 1024)),
                has_text=i.has_text,
                sample=i.sample,
            )
            for i in session.files
        ],
        search_mode=session.index.mode,
        warnings=warnings,
    )


def _get_session(session_id: str):
    s = store.get(session_id)
    if not s:
        raise HTTPException(404, "This chat has expired. Please load your PDFs again.")
    return s


@app.get("/api/sessions/{session_id}/summary", response_model=SummaryOut)
async def session_summary(session_id: str) -> SummaryOut:
    s = _get_session(session_id)
    if s.summary is None:
        s.summary = await summarize(s.chunks, llm)
    return SummaryOut(**s.summary)


@app.post("/api/sessions/{session_id}/ask", response_model=AnswerOut, dependencies=[Depends(rate_limited)])
async def ask(session_id: str, body: AskIn) -> AnswerOut:
    s = _get_session(session_id)
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Type a question first.")
    if len(question) > settings.max_question_chars:
        raise HTTPException(400, f"Keep questions under {settings.max_question_chars} characters.")
    ans = await ask_once(question, s, llm, settings)
    return AnswerOut(
        answer=ans.text,
        found=ans.found,
        mode=ans.mode,
        notice=ans.notice,
        sources=[SourceOut(file=x.file, page=x.page, label=x.label, snippet=x.snippet) for x in ans.sources],
        steps=[StepOut(kind=st.kind, title=st.title, detail=st.detail) for st in ans.steps],
    )


@app.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    store.delete(session_id)


# ------------------------------------------------------------------ Gradio UI

app = mount(app, settings, store, llm, limiter)
