"""Shared fixtures: a fresh temp DB built from the real CSVs, and a real
trained model artifact in a temp dir (302 rows train in seconds on CPU)."""
from __future__ import annotations

import sqlite3

import pytest

from app import build_db, predict_spp


@pytest.fixture(scope="session")
def temp_db(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("data") / "schools.db"
    schools, yayasan = build_db.build()
    build_db.write_db(schools, yayasan, db_path)
    return db_path


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory, temp_db):
    models_dir = tmp_path_factory.mktemp("models")
    artifact = models_dir / "spp_model.pkl"
    predict_spp.MODELS_DIR = models_dir
    predict_spp.MODEL_PATH = artifact
    predict_spp.train()
    return artifact


@pytest.fixture(scope="session")
def db_conn(temp_db):
    conn = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
