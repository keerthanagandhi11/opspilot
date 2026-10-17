import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_config(client):
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["llm_enabled"] is False and cfg["llm_model"] is None


def test_gradio_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "gradio" in r.text.lower()
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/config").json()["agent_enabled"] is False


def test_sample_session_flow(client):
    s = client.post("/api/sessions", data={"include_sample": "true"}).json()
    assert [f["pages"] for f in s["files"]] == [3, 5]
    sid = s["session_id"]

    a = client.post(f"/api/sessions/{sid}/ask", json={"question": "Who approved the rerun?"}).json()
    assert a["found"] and "shift lead" in a["answer"]
    assert a["sources"][0]["label"] == "incident_report_INC-1042.pdf p.3"

    summary = client.get(f"/api/sessions/{sid}/summary").json()
    assert summary["status"] and summary["missing"]

    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.post(f"/api/sessions/{sid}/ask", json={"question": "hi"}).status_code == 404


def test_upload_real_pdf(client, sample_pdf_bytes):
    files = [("files", ("report.pdf", sample_pdf_bytes, "application/pdf"))]
    s = client.post("/api/sessions", files=files).json()
    assert s["files"][0]["name"] == "report.pdf" and s["files"][0]["pages"] == 3
    a = client.post(f"/api/sessions/{s['session_id']}/ask", json={"question": "what was the severity?"}).json()
    assert a["found"] and a["sources"][0]["file"] == "report.pdf"


def test_duplicate_names_are_disambiguated(client, sample_pdf_bytes):
    files = [("files", ("r.pdf", sample_pdf_bytes, "application/pdf"))] * 2
    names = [f["name"] for f in client.post("/api/sessions", files=files).json()["files"]]
    assert names == ["r.pdf", "r (2).pdf"]


def test_rejects_non_pdf(client):
    r = client.post("/api/sessions", files=[("files", ("notes.pdf", b"hello", "application/pdf"))])
    assert r.status_code == 422
    r = client.post("/api/sessions", files=[("files", ("notes.txt", b"%PDF", "text/plain"))])
    assert r.status_code == 415


def test_requires_input(client):
    assert client.post("/api/sessions").status_code == 400


def test_question_validation(client):
    sid = client.post("/api/sessions", data={"include_sample": "true"}).json()["session_id"]
    assert client.post(f"/api/sessions/{sid}/ask", json={"question": "   "}).status_code == 400
    assert client.post(f"/api/sessions/{sid}/ask", json={"question": "x" * 600}).status_code == 400
