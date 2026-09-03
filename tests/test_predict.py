"""Featurizer + SPP predictor tests (small real model, trained in conftest)."""
from __future__ import annotations

import sqlite3

from app import predict_spp
from app.preprocessing import SchoolFeaturizer


def test_featurizer_fit_transform_shapes():
    df, _, _ = predict_spp.load_training_data()
    featurizer = SchoolFeaturizer().fit(df)
    X = featurizer.transform(df)
    assert list(X.columns) == featurizer.features
    assert len(X) == len(df)
    assert not X[featurizer.features[0]].isna().any()


def test_school_to_row_maps_all_columns(temp_db):
    df, _, _ = predict_spp.load_training_data()
    featurizer = SchoolFeaturizer().fit(df)
    conn = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    school = dict(conn.execute("SELECT * FROM schools LIMIT 1").fetchone())
    conn.close()
    row = featurizer.school_to_row(school)
    assert "Nama_sekolah" in row.columns and "Jenjang" in row.columns
    assert featurizer.transform(row).shape == (1, len(featurizer.features))


def test_predict_known_school_returns_recorded_spp(temp_db, trained_model):
    conn = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    school = dict(conn.execute(
        "SELECT * FROM schools WHERE spp > 0 LIMIT 1").fetchone())
    conn.close()
    result = predict_spp.predict_spp_for_school(school)
    assert result["known"] is True
    assert result["spp"] == school["spp"]


def test_predict_unknown_school_returns_positive_estimate(temp_db, trained_model):
    conn = sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    school = dict(conn.execute(
        "SELECT * FROM schools WHERE spp IS NULL AND peserta_didik > 50 LIMIT 1").fetchone())
    conn.close()
    result = predict_spp.predict_spp_for_school(school)
    assert result["known"] is False
    assert result["spp"] > 0


def test_find_school_by_name_is_case_insensitive(temp_db):
    hit = predict_spp.find_school_by_name("smk krian 1 sidoarjo", temp_db)
    assert hit is not None
    assert "KRIAN" in hit["nama"].upper()


def test_find_school_by_name_missing_returns_none(temp_db):
    assert predict_spp.find_school_by_name("Sekolah Tidak Ada 999 XYZ", temp_db) is None
