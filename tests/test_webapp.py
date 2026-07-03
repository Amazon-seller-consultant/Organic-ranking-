"""Web UI router tests against a temporary data dir (no live app data)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from catalog_engine import webapp
from catalog_engine.models import SellerConfig
from catalog_engine.store import Store


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "DATA_DIR", str(tmp_path))
    store = Store(tmp_path)
    store.upsert_seller(SellerConfig(seller_id="t1", display_name="Test"))
    store.close()
    app = FastAPI()
    app.include_router(webapp.router)
    return TestClient(app)


def test_sellers_roundtrip(client):
    sellers = client.get("/catalog/api/sellers").json()
    assert [s["seller_id"] for s in sellers] == ["t1"]
    r = client.post("/catalog/api/sellers", json={
        "seller_id": "t2", "marketplace": "UK", "on_limit_violation": "flag"})
    assert r.status_code == 200
    sellers = client.get("/catalog/api/sellers").json()
    assert {s["seller_id"] for s in sellers} == {"t1", "t2"}


def test_attest_requires_note_and_roundtrips(client):
    r = client.post("/catalog/api/sellers/t1/attest", json={"term": "clear"})
    assert r.status_code == 400  # note is the audit record — mandatory
    r = client.post("/catalog/api/sellers/t1/attest",
                    json={"term": "Clear", "note": "panels are clear acrylic"})
    assert r.json()["attested_terms"] == {"clear": "panels are clear acrylic"}
    r = client.post("/catalog/api/sellers/t1/attest",
                    json={"term": "clear", "remove": True})
    assert r.json()["attested_terms"] == {}


def test_upload_rejects_wrong_filetype(client):
    r = client.post("/catalog/api/sellers/t1/upload",
                    files={"file": ("report.txt", b"hi", "text/plain")})
    assert r.status_code == 400
    assert "not an .xlsm" in r.json()["detail"]


def test_run_requires_uploaded_file(client):
    r = client.post("/catalog/api/sellers/t1/runs", json={"filename": "nope.xlsm"})
    assert r.status_code == 400
    assert "not found" in r.json()["detail"]


def test_unknown_ticket_and_seller_404(client):
    assert client.get("/catalog/api/progress/tX").status_code == 404
    assert client.get("/catalog/api/sellers/ghost/runs").status_code == 404


def test_invalid_ids_rejected(client):
    assert client.get("/catalog/api/sellers/../etc/runs").status_code in (400, 404)
    r = client.get("/catalog/api/sellers/t1/runs/..%2F..%2Fsecret/artifacts")
    assert r.status_code in (400, 404)
