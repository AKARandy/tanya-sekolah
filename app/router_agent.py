"""Route a user question to one of the three retrieval paths.

Deliberately no LangChain/LangGraph: this is a single classification call
with a strict JSON output contract — transparent, cheap, and easy to test.
"""
from __future__ import annotations

import json
import re

from app.llm import chat

VALID_ROUTES = {"sql", "semantic", "predict"}

ROUTER_SYSTEM = """You route user questions about Indonesian schools (Surabaya & Sidoarjo) \
to the right retrieval path. Answer ONLY with a JSON object: {"route": "...", "school": null}

Routes:
- "sql": counting, filtering, aggregating, comparisons, rankings. Anything answerable by \
querying a table of schools (columns: nama, jenjang, status, kota, kecamatan, akreditasi, spp, \
prediksi_spp, kurikulum, peserta_didik, guru, pegawai, rombel, r_kelas, r_lab, r_perpus, umur, \
tahun_sk_pendirian, meter_listrik, harga_listrik_bulan, luas_tanah, harga_tanah_kecamatan, \
estimasi_harga_tanah, yayasan, jumlah_sekolah_naungan, grade_yayasan, is_agama, is_website, website).
- "semantic": vague/qualitative/descriptive questions with no exact numeric filter — finding \
schools that fit a fuzzy description.
- "predict": asking for the SPP (tuition) of one specific school, especially one likely to have \
no published fee data.

Set "school" to the school name (string) ONLY for the predict route, otherwise null.
Examples:
Q: "Berapa banyak SMA swasta terakreditasi A di Sidoarjo?" -> {"route": "sql", "school": null}
Q: "Sekolah mana yang cocok untuk anak saya yang suka seni di Surabaya barat?" -> {"route": "semantic", "school": null}
Q: "Berapa SPP SMK Krian 1 Sidoarjo?" -> {"route": "predict", "school": "SMK Krian 1 Sidoarjo"}
Q: "Which schools have a website?" -> {"route": "sql", "school": null}"""


def route_question(question: str) -> dict:
    """Return {'route': 'sql'|'semantic'|'predict', 'school': str|None}.

    Falls back to the semantic route when the router output is unusable — a
    wrong-but-plausible retrieval beats an error for this app.
    """
    try:
        raw = chat(ROUTER_SYSTEM, question, max_tokens=100)
    except Exception:
        return {"route": "semantic", "school": None, "error": "router_unavailable"}

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"route": "semantic", "school": None, "error": "router_no_json"}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"route": "semantic", "school": None, "error": "router_bad_json"}

    route = str(parsed.get("route", "")).lower().strip()
    school = parsed.get("school")
    if route not in VALID_ROUTES:
        return {"route": "semantic", "school": None, "error": "router_bad_route"}
    if route == "predict" and (not isinstance(school, str) or not school.strip()):
        return {"route": "semantic", "school": None, "error": "predict_without_school"}
    return {"route": route, "school": school.strip() if isinstance(school, str) else None}
