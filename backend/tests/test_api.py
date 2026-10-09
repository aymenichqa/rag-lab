# -*- coding: utf-8 -*-
"""Tests API (pytest + httpx). Nécessite results/*.json (notebook exécuté)."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    import backend.main as backend_main
    importlib.reload(backend_main)
    return TestClient(backend_main.app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_results_chunking(client):
    r = client.get("/api/results/chunking")
    assert r.status_code == 200
    body = r.json()
    assert body["stage"] == "chunking"
    assert len(body["methods"]) == 7
    assert "justification" in body["best"] and len(body["best"]["justification"]) > 80


def test_results_embedding_and_retrieval(client):
    for stage, expected in (("embedding", 7), ("retrieval", 6)):
        r = client.get(f"/api/results/{stage}")
        assert r.status_code == 200
        assert len(r.json()["methods"]) == expected


def test_unknown_stage_404(client):
    assert client.get("/api/results/unknown").status_code == 404


def test_best_methods(client):
    r = client.get("/api/best-methods")
    assert r.status_code == 200
    body = r.json()
    for stage in ("chunking", "embedding", "retrieval"):
        assert stage in body and body[stage]["name"] and len(body[stage]["justification"]) > 80
    assert "pipeline_final" in body


def test_query_live(client):
    r = client.post("/api/query", json={
        "question": "Comment les zones humides réduisent-elles la pollution par les nitrates ?",
        "method": "all",
        "top_k": 3,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["passages"]) == 3
    assert len(body["timings_ms"]) == 6  # les 6 méthodes chronométrées
    assert "generation" in body


def test_query_invalid(client):
    assert client.post("/api/query", json={"question": "ab"}).status_code == 422
