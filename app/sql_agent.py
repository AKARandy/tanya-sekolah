"""Text-to-SQL route: LLM writes a SELECT, we execute it under guard.

The guard is the actual engineering here, so it is implemented in plain code
rather than trusted to the LLM:
  - the SQLite connection is opened in read-only mode (uri=True, mode=ro),
  - only a single SELECT statement is accepted (no semicolons, no comment
    tricks, must start with SELECT),
  - a row LIMIT is enforced,
  - on a SQL error the LLM gets one repair attempt with the error message.
"""
from __future__ import annotations

import re
import sqlite3

from app.config import DB_PATH
from app.llm import chat

MAX_ROWS = 50

SCHEMA_HINT = """SQLite schema:
  schools(npsn TEXT, nama TEXT, jenjang TEXT, status TEXT, kota TEXT, kecamatan TEXT,
          kelurahan TEXT, alamat TEXT, akreditasi TEXT, spp REAL, prediksi_spp REAL,
          kurikulum TEXT, peserta_didik REAL, rombel REAL, guru REAL, pegawai REAL,
          r_kelas REAL, r_lab REAL, r_perpus REAL, umur REAL, tahun_sk_pendirian REAL,
          meter_listrik REAL, harga_listrik_bulan REAL, luas_tanah REAL,
          harga_tanah_kecamatan REAL, estimasi_harga_tanah REAL, yayasan TEXT,
          jumlah_sekolah_naungan REAL, grade_yayasan REAL, is_agama INTEGER,
          is_website INTEGER, website TEXT, is_surabaya INTEGER, telepon TEXT,
          email TEXT, kepala_sekolah TEXT, operator TEXT, pimpinan_yayasan TEXT)
  yayasan(nama TEXT, jumlah_sekolah_naungan REAL, grade REAL, pimpinan TEXT)

Notes:
- kota values: 'Kota Surabaya' or 'Kab. Sidoarjo'. kecamatan values look
  like 'KEC. RUNGKUT' (uppercase, prefixed 'KEC. ').
- jenjang values: SD, SMP, SMA, SMK, MI, MTs, MA, PKBM, TK, KB, SPK SMA/SPK SMP etc.
  Use LIKE '%...%' when unsure.
- status: 'NEGERI' or 'SWASTA' (only Surabaya rows have it).
- spp is the recorded monthly tuition; many rows are NULL. prediksi_spp is the model-estimated
  tuition for schools where spp is unknown (mostly Sidoarjo). For tuition questions use
  COALESCE(spp, prediksi_spp) and mention which kind of value it is.
- akreditasi values: A, B, C, 'Belum Terakreditasi', or empty. is_agama/is_website: 1/0.
- For "berapa banyak" (count) questions return one row with the count; for list questions
  include nama and the relevant columns, ordered sensibly."""

FEW_SHOT = (
    "User: Berapa banyak SMA swasta berakreditasi A di Kota Surabaya?\n"
    "SQL: SELECT COUNT(*) AS jumlah FROM schools\n"
    "     WHERE jenjang LIKE 'SMA%' AND status = 'SWASTA'\n"
    "       AND akreditasi = 'A' AND kota = 'Kota Surabaya'\n"
    "\n"
    "User: Rata-rata SPP per jenjang di Sidoarjo?\n"
    "SQL: SELECT jenjang, ROUND(AVG(COALESCE(spp, prediksi_spp))) AS spp_rata2,\n"
    "            COUNT(*) AS jumlah\n"
    "     FROM schools\n"
    "     WHERE kota = 'Kab. Sidoarjo' AND COALESCE(spp, prediksi_spp) > 0\n"
    "     GROUP BY jenjang ORDER BY spp_rata2 DESC\n"
    "\n"
    "User: 5 sekolah dengan peserta didik terbanyak\n"
    "SQL: SELECT nama, jenjang, kota, peserta_didik FROM schools\n"
    "     ORDER BY peserta_didik DESC LIMIT 5\n"
    "\n"
    "User: Sekolah agama yang punya website di Sidoarjo\n"
    "SQL: SELECT nama, jenjang, kota, website FROM schools\n"
    "     WHERE is_agama = 1 AND is_website = 1 AND kota = 'Kab. Sidoarjo'"
)


def _validate_sql(sql: str) -> tuple[bool, str]:
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False, "empty SQL"
    if ";" in stripped:
        return False, "multiple statements are not allowed"
    if "--" in stripped or "/*" in stripped:
        return False, "SQL comments are not allowed"
    if not re.match(r"(?is)^WITH\b.*\bSELECT\b|^SELECT\b", stripped):
        return False, "only SELECT statements are allowed"
    if re.search(r"(?is)\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|VACUUM)\b", stripped):
        return False, "modification statements are not allowed"
    return True, stripped


def run_select(sql: str, db_path=DB_PATH) -> list[dict]:
    ok, checked = _validate_sql(sql)
    if not ok:
        raise ValueError(checked)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(checked + f" LIMIT {MAX_ROWS}").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def generate_sql(question: str) -> str:
    system = (
        "You are a senior SQLite analyst. Given a question about Indonesian schools, "
        "write ONE SQLite SELECT query. Output ONLY the SQL, no explanation, no markdown.\n\n"
        + SCHEMA_HINT
        + "\n\nExamples:\n"
        + FEW_SHOT
    )
    return chat(system, question, max_tokens=400).strip()


def answer_sql(question: str) -> dict:
    """Generate, guard, execute; one LLM repair round on SQL errors."""
    sql = generate_sql(question)
    try:
        rows = run_select(sql)
    except (ValueError, sqlite3.Error) as exc:
        repair_prompt = (
            f"Your SQL failed.\nQuestion: {question}\nSQL: {sql}\nError: {exc}\n"
            "Write ONE corrected SQLite SELECT. Output ONLY the SQL."
        )
        sql = chat(
            "You fix SQL errors. Output ONLY the corrected SQL, no explanation.\n\n" + SCHEMA_HINT,
            repair_prompt,
            max_tokens=400,
        ).strip()
        rows = run_select(sql)  # second failure propagates to the caller
    return {"sql": sql, "rows": rows}
