"""Split each page into overlapping chunks. Chunks never cross a page boundary,
so every chunk can be cited as "file.pdf p.N"."""

from __future__ import annotations

from dataclasses import dataclass

from .pdf_parser import ParsedPdf


@dataclass
class Chunk:
    id: int
    file: str
    page: int  # 1-based
    text: str

    @property
    def label(self) -> str:
        return f"{self.file} p.{self.page}"


def _windows(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            # prefer to cut at a sentence end, else at a space
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut <= start + size // 2:
                cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [w for w in out if w]


def chunk_documents(docs: list[ParsedPdf], size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for page_no, text in enumerate(doc.pages, start=1):
            if not text.strip():
                continue
            for piece in _windows(text, size, overlap):
                chunks.append(Chunk(id=len(chunks), file=doc.name, page=page_no, text=piece))
    return chunks
