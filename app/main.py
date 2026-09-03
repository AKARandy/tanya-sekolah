"""Sekolah Assistant API.

Pipeline for POST /ask:
  question -> router LLM -> one of:
    sql       -> generate SQL -> guarded SELECT on schools.db -> synthesis LLM
    semantic  -> fastembed+Chroma top-k profile cards -> synthesis LLM
    predict   -> school lookup -> CatBoost SPP estimate -> phrasing LLM

Every LLM hop degrades gracefully: if the backend is down the endpoint still
returns the raw retrieval evidence so the app demos even without a key.

Run:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import sqlite3
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import predict_spp, router_agent, sql_agent, synthesis
from app.build_index import semantic_search
from app.config import CHROMA_PATH, DB_PATH, resolve_llm

app = FastAPI(title="Sekolah Assistant", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class PredictRequest(BaseModel):
    school: str = Field(min_length=2, max_length=200)


@app.get("/health")
def health() -> dict:
    db_ok, n_schools = False, 0
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        n_schools = conn.execute("SELECT COUNT(*) FROM schools").fetchone()[0]
        conn.close()
        db_ok = True
    except sqlite3.Error:
        pass
    base_url, model = resolve_llm()
    return {
        "status": "ok" if db_ok else "degraded",
        "schools_in_db": n_schools,
        "vector_index_ready": CHROMA_PATH.exists(),
        "model_artifact": predict_spp.MODEL_PATH.exists(),
        "llm_base_url": base_url,
        "llm_model": model,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    question = req.question.strip()

    routed = router_agent.route_question(question)
    route, school_name = routed["route"], routed.get("school")
    note = routed.get("error")

    if route == "predict":
        school = predict_spp.find_school_by_name(school_name)
        if school is None:
            raise HTTPException(404, f"School '{school_name}' not found in the database.")
        result = predict_spp.predict_spp_for_school(school)
        kind = ("SPP tercatat (data asli)" if result["known"]
                else "perkiraan dari model CatBoost (SPP asli tidak tersedia)")
        fallback = (
            f"{school['nama']} (NPSN {school['npsn']}) — {kind}: "
            f"Rp{result['spp']:,.0f}".replace(",", ".")
        )
        try:
            answer = synthesis.chat(
                synthesis.SYNTH_SYSTEM,
                f"Question: {question}\n\nEvidence:\nSchool: {school['nama']} (NPSN {school['npsn']}), "
                f"{school['jenjang']}, {school['kota']}, yayasan {school['yayasan']}. "
                f"SPP info: {kind} = Rp{result['spp']:,.0f}.\n\nWrite the final answer.",
                max_tokens=300,
            )
        except Exception:
            answer = fallback
        return {"route": route, "school": school["nama"], "answer": answer,
                "spp": result["spp"], "spp_is_recorded": result["known"], "note": note}

    if route == "sql":
        try:
            result = sql_agent.answer_sql(question)
            rows = result["rows"]
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(502, "The SQL route failed after a repair attempt.") from exc
        try:
            answer = synthesis.synthesize_with_sql(question, result["sql"], rows)
        except Exception:
            answer = f"Hasil query: {rows}" if rows else "Tidak ada baris yang cocok."
        return {"route": route, "sql": result["sql"], "rows": rows, "answer": answer, "note": note}

    # semantic (also the fallback route)
    hits = semantic_search(question, k=5)
    try:
        answer = synthesis.synthesize_with_semantic(question, hits)
    except Exception:
        answer = "\n\n".join(h["card"] for h in hits) or "Tidak ada hasil."
    return {
        "route": "semantic",
        "sources": [h["metadata"] for h in hits],
        "answer": answer,
        "note": note,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    school = predict_spp.find_school_by_name(req.school)
    if school is None:
        raise HTTPException(404, f"School '{req.school}' not found in the database.")
    result = predict_spp.predict_spp_for_school(school)
    return {
        "npsn": school["npsn"],
        "nama": school["nama"],
        "jenjang": school["jenjang"],
        "kota": school["kota"],
        "spp": result["spp"],
        "spp_is_recorded": result["known"],
    }
