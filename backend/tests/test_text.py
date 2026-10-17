from app.chunking import _windows
from app.text import TIME_TOKEN, expand_query, split_sentences, stem, tokenize


def test_stem_matches_word_forms():
    assert stem("failed") == stem("fails") == "fail"
    assert stem("02:10") == "02:10"


def test_tokenize_drops_stopwords_and_trailing_colons():
    toks = tokenize("What was the Business impact: a delay at 02:10?")
    assert "the" not in toks and "impact" in toks and "02:10" in toks
    assert TIME_TOKEN in toks


def test_expand_query_adds_synonyms():
    assert "cause" in expand_query(tokenize("why did it fail"))


def test_split_sentences_rejoins_wrapped_lines():
    text = "Root cause: the feed was delivered at 02:55, after\nthe job started. Second sentence."
    assert split_sentences(text) == [
        "Root cause: the feed was delivered at 02:55, after the job started.",
        "Second sentence.",
    ]


def test_windows_overlap_and_cover_text():
    text = " ".join(f"word{i}." for i in range(400))
    parts = _windows(text, 300, 50)
    assert len(parts) > 1
    assert all(len(p) <= 300 for p in parts)
    assert parts[-1].endswith("word399.")


def _encrypted_pdf(user_password: str, algorithm: str) -> bytes:
    import io
    from pathlib import Path

    from pypdf import PdfReader, PdfWriter

    src = Path(__file__).resolve().parent.parent / "app" / "sample_data" / "incident_report_INC-1042.pdf"
    w = PdfWriter()
    for p in PdfReader(src).pages:
        w.add_page(p)
    w.encrypt(user_password=user_password, owner_password="owner", algorithm=algorithm)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_restricted_aes_pdf_opens_without_password():
    from app.pdf_parser import parse_pdf

    doc = parse_pdf("r.pdf", _encrypted_pdf("", "AES-256"), 300)
    assert doc.page_count == 3 and "nightly_batch_07" in doc.pages[0]


def test_password_protected_pdf_has_clear_error():
    import pytest

    from app.pdf_parser import PdfError, parse_pdf

    with pytest.raises(PdfError, match="password protected"):
        parse_pdf("locked.pdf", _encrypted_pdf("secret", "AES-256"), 300)
