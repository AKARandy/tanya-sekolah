"""Data layer tests: unified schema, dedup, numeric parsing."""
from __future__ import annotations

import pandas as pd

from app import build_db


def test_row_counts(temp_db):
    import sqlite3
    conn = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    total, sby, sda = conn.execute(
        "SELECT COUNT(*), SUM(is_surabaya), SUM(1 - is_surabaya) FROM schools"
    ).fetchone()
    conn.close()
    assert total == sby + sda
    assert sby > 1500 and sda > 300


def test_no_duplicate_npsn(temp_db):
    import sqlite3
    conn = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    dupes = conn.execute(
        "SELECT npsn, COUNT(*) c FROM schools GROUP BY npsn HAVING c > 1"
    ).fetchall()
    conn.close()
    assert dupes == []


def test_spp_parsing_handles_rp_strings():
    """Sidoarjo fees look like 'Rp220,000' — the parser must read them."""
    series = build_db._to_num(pd.Series(["Rp220,000", "-", None, "300000"]))
    assert series.iloc[0] == 220000
    assert series.iloc[3] == 300000


def test_year_parsing_handles_excel_serials():
    import pandas as pd
    out = build_db._clean_year(pd.Series([1992, 44789, "2011-05-01", None]))
    assert out.iloc[0] == 1992
    assert 2000 <= out.iloc[1] <= 2100  # excel serial mapped into a sane year
    assert out.iloc[2] == 2011
    assert pd.isna(out.iloc[3])


def test_every_school_has_yayasan_key(temp_db):
    import sqlite3
    conn = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    missing = conn.execute(
        "SELECT COUNT(*) FROM schools WHERE yayasan_key IS NULL OR yayasan_key = ''"
    ).fetchone()[0]
    yayasan_rows = conn.execute("SELECT COUNT(*) FROM yayasan").fetchone()[0]
    conn.close()
    assert missing == 0
    assert yayasan_rows > 500


def test_contact_columns_exist_but_card_excludes_them(temp_db):
    import sqlite3
    conn = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(schools)").fetchall()]
    conn.close()
    for contact in ("telepon", "email", "operator", "kepala_sekolah"):
        assert contact in cols
    from app import build_index
    card = build_index.school_card({
        "nama": "SD TEST", "jenjang": "SD", "status": "SWASTA",
        "kota": "Kota Surabaya", "kecamatan": "KEC. RUNGKUT", "akreditasi": "A",
        "peserta_didik": 100, "guru": 10, "yayasan": "YAYASAN TEST",
        "grade_yayasan": 2, "is_agama": 1, "spp": 150000,
        "prediksi_spp": None, "website": "http://x.sch.id",
    })
    for contact in ("telepon", "email", "operator", "kepala"):
        assert contact.lower() not in card.lower()
