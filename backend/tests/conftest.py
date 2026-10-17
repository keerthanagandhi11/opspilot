import os
import sys
from pathlib import Path

# Tests run offline and deterministic: no LLM key, no embedding model download.
os.environ["LLM_API_KEY"] = ""
os.environ["ENABLE_EMBEDDINGS"] = "false"
os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "app" / "sample_data"


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    return (SAMPLE_DIR / "incident_report_INC-1042.pdf").read_bytes()


@pytest.fixture
def sample_session():
    from app.config import get_settings
    from app.pipeline import build_session
    from app.sessions import SessionStore

    session, _ = build_session([], True, get_settings(), SessionStore(3600, 10))
    return session


@pytest.fixture
def sample_index(sample_session):
    return sample_session.chunks, sample_session.index
