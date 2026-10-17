"""Turn uploaded PDFs into a searchable session. Shared by the REST API and the Gradio UI."""

from __future__ import annotations

import logging

from .chunking import chunk_documents
from .config import BASE_DIR, Settings
from .pdf_parser import ParsedPdf, PdfError, parse_pdf
from .retrieval import Index
from .sessions import FileInfo, Session, SessionStore

log = logging.getLogger(__name__)

SAMPLE_DIR = BASE_DIR / "sample_data"


class InputError(ValueError):
    """A problem with what the user uploaded. `status` is the matching HTTP code."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def load_samples(settings: Settings) -> list[ParsedPdf]:
    return [parse_pdf(p.name, p.read_bytes(), settings.max_pages_per_file) for p in sorted(SAMPLE_DIR.glob("*.pdf"))]


def clean_name(raw: str | None) -> str:
    return (raw or "upload.pdf").replace("\\", "/").rsplit("/", 1)[-1][:120]


def build_session(
    uploads: list[tuple[str, bytes]], include_sample: bool, settings: Settings, store: SessionStore
) -> tuple[Session, list[str]]:
    """Parse, chunk and index the PDFs. Blocking (CPU work): call from a worker thread."""
    if not uploads and not include_sample:
        raise InputError("Upload at least one PDF or choose the sample incident.")
    if len(uploads) > settings.max_files:
        raise InputError(f"Upload at most {settings.max_files} PDFs at a time.")

    docs: list[ParsedPdf] = []
    infos: list[FileInfo] = []
    warnings: list[str] = []

    if include_sample:
        for d in load_samples(settings):
            docs.append(d)
            infos.append(FileInfo(d.name, d.page_count, d.size_bytes, d.has_text, sample=True))

    seen = {d.name for d in docs}
    for raw_name, data in uploads:
        name = clean_name(raw_name)
        if not name.lower().endswith(".pdf"):
            raise InputError(f"{name} is not a PDF.", 415)
        if len(data) > settings.max_file_bytes:
            raise InputError(f"{name} is larger than {settings.max_file_mb:g} MB.", 413)
        base, n = name, 2
        while name in seen:  # keep citations unambiguous
            name = f"{base[:-4]} ({n}).pdf"
            n += 1
        seen.add(name)
        try:
            doc = parse_pdf(name, data, settings.max_pages_per_file)
        except PdfError as exc:
            raise InputError(str(exc), 422) from exc
        if not doc.has_text:
            warnings.append(f"{name} has no selectable text (it may be a scanned image), so it cannot be searched.")
        docs.append(doc)
        infos.append(FileInfo(doc.name, doc.page_count, doc.size_bytes, doc.has_text))

    chunks = chunk_documents(docs, settings.chunk_chars, settings.chunk_overlap)
    if not chunks:
        raise InputError("None of these PDFs contain selectable text.", 422)
    index = Index(chunks, settings.enable_embeddings, settings.embedding_model)
    pages = {(d.name, i): text for d in docs for i, text in enumerate(d.pages, start=1)}
    session = store.create(infos, chunks, index, pages)
    log.info("Session created: %d files, %d chunks, %s search", len(infos), len(chunks), index.mode)
    return session, warnings
