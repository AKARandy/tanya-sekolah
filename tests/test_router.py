"""Router tests — all LLM calls mocked; no backend needed."""
from __future__ import annotations

from app import router_agent


def test_valid_sql_route(monkeypatch):
    monkeypatch.setattr(router_agent, "chat", lambda *a, **k: '{"route": "sql", "school": null}')
    assert router_agent.route_question("berapa banyak SMA di Surabaya?")["route"] == "sql"


def test_predict_route_extracts_school(monkeypatch):
    monkeypatch.setattr(router_agent, "chat", lambda *a, **k:
                        '{"route": "predict", "school": "SMK Krian 1 Sidoarjo"}')
    out = router_agent.route_question("berapa SPP SMK Krian 1 Sidoarjo?")
    assert out["route"] == "predict"
    assert out["school"] == "SMK Krian 1 Sidoarjo"


def test_bad_json_falls_back_to_semantic(monkeypatch):
    monkeypatch.setattr(router_agent, "chat", lambda *a, **k: "I think it's sql!")
    out = router_agent.route_question("apapun")
    assert out["route"] == "semantic" and out["error"] == "router_no_json"


def test_invalid_route_falls_back(monkeypatch):
    monkeypatch.setattr(router_agent, "chat", lambda *a, **k: '{"route": "vector"}')
    out = router_agent.route_question("apapun")
    assert out["route"] == "semantic" and out["error"] == "router_bad_route"


def test_predict_without_school_falls_back(monkeypatch):
    monkeypatch.setattr(router_agent, "chat", lambda *a, **k: '{"route": "predict", "school": null}')
    out = router_agent.route_question("berapa SPP sekolah itu?")
    assert out["route"] == "semantic" and out["error"] == "predict_without_school"


def test_llm_failure_falls_back(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("no backend")
    monkeypatch.setattr(router_agent, "chat", boom)
    out = router_agent.route_question("apapun")
    assert out["route"] == "semantic" and out["error"] == "router_unavailable"
