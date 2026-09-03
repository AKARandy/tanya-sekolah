"""The SQL guard is the security boundary — test it like one."""
from __future__ import annotations

import pytest

from app.sql_agent import _validate_sql, run_select


class TestValidator:
    def test_plain_select_ok(self):
        ok, checked = _validate_sql("SELECT COUNT(*) FROM schools")
        assert ok and checked.startswith("SELECT")

    def test_with_cte_ok(self):
        ok, checked = _validate_sql(
            "WITH a AS (SELECT jenjang FROM schools) SELECT * FROM a"
        )
        assert ok

    def test_rejects_drop(self):
        ok, _ = _validate_sql("DROP TABLE schools")
        assert not ok

    def test_rejects_insert_update_delete(self):
        for bad in ("INSERT INTO schools VALUES (1)",
                    "UPDATE schools SET spp = 0",
                    "DELETE FROM schools"):
            ok, _ = _validate_sql(bad)
            assert not ok, bad

    def test_rejects_pragma_and_attach(self):
        for bad in ("PRAGMA table_info(schools)",
                    "ATTACH DATABASE 'x.db' AS x"):
            ok, _ = _validate_sql(bad)
            assert not ok

    def test_rejects_multiple_statements(self):
        ok, _ = _validate_sql("SELECT 1; DROP TABLE schools")
        assert not ok

    def test_rejects_comments(self):
        ok, _ = _validate_sql("SELECT 1 -- DROP TABLE schools")
        assert not ok

    def test_rejects_empty(self):
        ok, _ = _validate_sql("   ")
        assert not ok


def test_validator_rejects_multi_statement_dml(temp_db):
    with pytest.raises(ValueError):
        run_select("SELECT 1 AS x FROM schools; DELETE FROM schools", temp_db)


def test_readonly_connection_blocks_writes(temp_db):
    """Even if the guard missed something, mode=ro makes mutations fail."""
    import sqlite3
    conn = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM schools")
    conn.close()


def test_run_select_returns_dicts_with_limit(temp_db):
    rows = run_select("SELECT npsn, nama FROM schools", temp_db)
    assert 0 < len(rows) <= 50
    assert isinstance(rows[0], dict) and "nama" in rows[0]


def test_select_returns_zero_rows_for_absurd_filter(temp_db):
    rows = run_select("SELECT nama FROM schools WHERE peserta_didik > 999999", temp_db)
    assert rows == []
