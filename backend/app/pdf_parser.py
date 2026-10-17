"""Turn PDF bytes into per-page text, so every chunk keeps its page number."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError

log = logging.getLogger(__name__)


class PdfError(ValueError):
    """Raised when an upload is not a readable PDF."""


@dataclass
class ParsedPdf:
    name: str
    size_bytes: int
    pages: list[str] = field(default_factory=list)  # index 0 = page 1

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def has_text(self) -> bool:
        return any(p.strip() for p in self.pages)


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    # re-join words hyphenated across line breaks, then collapse whitespace
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(name: str, data: bytes, max_pages: int) -> ParsedPdf:
    if not data.startswith(b"%PDF"):
        raise PdfError(f"{name} is not a PDF file.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise PdfError(f"{name} is password protected.") from exc
        if len(reader.pages) > max_pages:
            raise PdfError(f"{name} has {len(reader.pages)} pages; the limit is {max_pages}.")
        pages = []
        for page in reader.pages:
            try:
                pages.append(_clean(page.extract_text() or ""))
            except Exception:  # noqa: BLE001 - one bad page should not sink the file
                log.warning("Could not read a page of %s", name, exc_info=True)
                pages.append("")
    except PdfError:
        raise
    except (PdfReadError, Exception) as exc:  # noqa: BLE001
        raise PdfError(f"{name} could not be read as a PDF.") from exc
    return ParsedPdf(name=name, size_bytes=len(data), pages=pages)
