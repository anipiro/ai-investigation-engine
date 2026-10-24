import pytest
from fastapi.testclient import TestClient

from engine.api.app import create_app

ANSWER = {"status": "answered", "answer": "Pool reduced to 80 [S1].", "short_answer": "80",
          "claims": [{"statement": "pool reduced to 80", "citations": ["S1"]}], "conflicts": [],
          "missing_information": [], "confidence": 0.8}


@pytest.fixture
def client(services):
    with TestClient(create_app(services)) as c:
        yield c


def _upload(client, name="runbook.md", body=b"# Runbook\n\nThe ledger pool was reduced to 80 connections."):
    return client.post("/api/documents", files=[("files", (name, body, "text/plain"))])


def test_health_and_stats(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["llm_configured"] is True
    assert client.get("/api/stats").json()["documents"] == 0
    assert "counters" in client.get("/api/metrics").json()


def test_document_lifecycle(client):
    r = _upload(client)
    assert r.status_code == 201 and r.json()["results"][0]["status"] == "created"
    doc_id = r.json()["results"][0]["doc_id"]
    assert doc_id == "upload::runbook"
    assert _upload(client).json()["results"][0]["status"] == "unchanged"

    listed = client.get("/api/documents", params={"source": "upload"}).json()["documents"]
    assert [d["doc_id"] for d in listed] == [doc_id]
    detail = client.get(f"/api/documents/{doc_id}").json()
    assert detail["chunks"][0]["text"].startswith("The ledger pool")

    r = client.put(f"/api/documents/{doc_id}", files={"file": ("runbook.md", b"# Runbook\n\nPool is now 200.", "text/plain")})
    assert r.json()["status"] == "updated"
    assert client.delete(f"/api/documents/{doc_id}").json()["status"] == "deleted"
    assert client.get(f"/api/documents/{doc_id}").status_code == 404


def test_upload_errors(client):
    assert client.post("/api/documents", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]).status_code == 415
    assert client.post("/api/documents", files=[("files", ("x.txt", b"   ", "text/plain"))]).status_code == 422


def test_protected_documents_need_admin_token(client, services):
    services.ingestion.ingest(doc_id="hotpot::x", title="X", text="Protected seed text.", source="hotpotqa")
    assert client.delete("/api/documents/hotpot::x").status_code == 403
    assert client.delete("/api/documents/hotpot::x", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.delete("/api/documents/hotpot::x", headers={"X-Admin-Token": "admin-secret"}).status_code == 200


def test_search_endpoint(client):
    _upload(client)
    r = client.post("/api/search", json={"query": "ledger pool", "mode": "hybrid", "rerank": False})
    assert r.status_code == 200 and r.json()["hits"][0]["doc_id"] == "upload::runbook"
    assert client.post("/api/search", json={"query": ""}).status_code == 422


def test_ask_uses_cache_and_invalidates_on_change(client, fake_gemini):
    _upload(client)
    fake_gemini.text(ANSWER)
    body = {"question": "What is the pool size?", "mode": "simple"}
    first = client.post("/api/ask", json=body).json()
    assert first["status"] == "answered" and first["cached"] is False
    assert client.post("/api/ask", json=body).json()["cached"] is True
    _upload(client, "other.md", b"# Other\n\nUnrelated note about receipts.")
    fake_gemini.text(ANSWER)
    assert client.post("/api/ask", json=body).json()["cached"] is False


def test_ask_stream_emits_events(client, fake_gemini):
    _upload(client)
    fake_gemini.text(ANSWER)
    with client.stream("POST", "/api/ask/stream", json={"question": "What is the pool size?", "mode": "simple"}) as r:
        body = "".join(r.iter_text())
    assert "event: plan" in body and "event: search" in body and "event: result" in body
