"""API tests: FastAPI endpoints with mocked LLM + retrieval; no network."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main, predict_spp


@pytest.fixture()
def client(temp_db, trained_model, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", temp_db)
    monkeypatch.setattr(predict_spp, "DB_PATH", temp_db)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["schools_in_db"] > 1800
    assert body["model_artifact"] is True


def test_ask_sql_route(client, monkeypatch):
    monkeypatch.setattr(main.router_agent, "route_question",
                        lambda q: {"route": "sql", "school": None})
    monkeypatch.setattr(main.sql_agent, "answer_sql",
                        lambda q: {"sql": "SELECT COUNT(*) AS n FROM schools", "rows": [{"n": 2269}]})
    monkeypatch.setattr(main.synthesis, "synthesize_with_sql", lambda q, s, r: "Ada 2269 sekolah.")
    res = client.post("/ask", json={"question": "berapa banyak sekolah?"})
    assert res.status_code == 200
    body = res.json()
    assert body["route"] == "sql"
    assert body["answer"] == "Ada 2269 sekolah."
    assert body["rows"][0]["n"] == 2269


def test_ask_sql_route_failure_returns_502(client, monkeypatch):
    monkeypatch.setattr(main.router_agent, "route_question",
                        lambda q: {"route": "sql", "school": None})
    def boom(q):
        raise ValueError("two failed attempts")
    monkeypatch.setattr(main.sql_agent, "answer_sql", boom)
    res = client.post("/ask", json={"question": "berapa banyak sekolah?"})
    assert res.status_code == 502


def test_ask_predict_route_known_spp(client, monkeypatch):
    monkeypatch.setattr(main.router_agent, "route_question",
                        lambda q: {"route": "predict", "school": "SMK KRIAN 1 SIDOARJO"})
    monkeypatch.setattr(main.synthesis, "chat", lambda *a, **k: "SPP-nya Rp220.000.")
    res = client.post("/ask", json={"question": "berapa SPP SMK Krian 1?"})
    body = res.json()
    assert body["route"] == "predict"
    assert body["spp_is_recorded"] is True
    assert body["spp"] == 220000.0


def test_ask_predict_route_unknown_school_404(client, monkeypatch):
    monkeypatch.setattr(main.router_agent, "route_question",
                        lambda q: {"route": "predict", "school": "TIDAK ADA XYZ"})
    res = client.post("/ask", json={"question": "berapa SPP TIDAK ADA XYZ?"})
    assert res.status_code == 404


def test_ask_semantic_route(client, monkeypatch):
    monkeypatch.setattr(main.router_agent, "route_question",
                        lambda q: {"route": "semantic", "school": None})
    monkeypatch.setattr(main, "semantic_search", lambda q, k=5: [
        {"npsn": "20533125", "card": "Nama sekolah: SD KHADIJAH 3.",
         "metadata": {"npsn": "20533125", "nama": "SD KHADIJAH 3"}, "distance": 0.2},
    ])
    monkeypatch.setattr(main.synthesis, "synthesize_with_semantic", lambda q, h: "SD KHADIJAH 3 cocok.")
    res = client.post("/ask", json={"question": "rekomendasi sekolah agama"})
    body = res.json()
    assert body["route"] == "semantic"
    assert body["answer"] == "SD KHADIJAH 3 cocok."
    assert body["sources"][0]["npsn"] == "20533125"


def test_ask_validation_min_length(client):
    res = client.post("/ask", json={"question": "ab"})
    assert res.status_code == 422


def test_predict_endpoint(client):
    res = client.post("/predict", json={"school": "SMK KRIAN 1 SIDOARJO"})
    assert res.status_code == 200
    body = res.json()
    assert body["nama"] == "SMK KRIAN 1 SIDOARJO"
    assert body["spp"] == 220000.0 and body["spp_is_recorded"] is True


def test_predict_endpoint_missing_404(client):
    res = client.post("/predict", json={"school": "TIDAK ADA XYZ"})
    assert res.status_code == 404
