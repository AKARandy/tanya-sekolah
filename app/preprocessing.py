"""Cleaning + feature engineering shared by training and inference.

Extracted from the Kaggle train_cv.py logic so the served model applies
exactly the same transformations that were used during cross-validation.
SchoolFeaturizer.fit() captures the fill statistics and categorical column
list from the training frame; transform() then applies them unchanged —
including at inference time, where a school row from schools.db is mapped
into the same feature schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns present in Gabungan_Surabaya_Sidoarjo.csv / schools.db before
# modeling; Yayasan, Website, NPSN are dropped exactly like train_cv.py did.
NON_FEATURE = ["SPP", "SPP_log", "Nama_sekolah", "Yayasan", "Website", "NPSN"]
MEAN_FILL = [
    "meter listrik", "harga_listrik_per_bulan", "estimasi harga tanah sekolah",
    "Tanggal SK Pendirian", "Umur", "luas tanah",
]
DB_TO_TRAINING = {
    "npsn": "NPSN",
    "nama": "Nama_sekolah",
    "jenjang": "Jenjang",
    "akreditasi": "Akreditasi",
    "peserta_didik": "Peserta Didik",
    "rombel": "Rombel",
    "guru": "Guru",
    "pegawai": "Pegawai",
    "r_kelas": "R.Kelas",
    "r_lab": "R.Lab",
    "r_perpus": "R.Perpus",
    "kurikulum": "Kurikulum",
    "yayasan": "Yayasan",
    "jumlah_sekolah_naungan": "Jumlah Sekolah Naungan Yayasan",
    "grade_yayasan": "Grade Yayasan Sekolah",
    "tahun_sk_pendirian": "Tanggal SK Pendirian",
    "umur": "Umur",
    "meter_listrik": "meter listrik",
    "harga_listrik_bulan": "harga_listrik_per_bulan",
    "luas_tanah": "luas tanah",
    "harga_tanah_kecamatan": "Harga Tanah Pasaran per kecamatan",
    "estimasi_harga_tanah": "estimasi harga tanah sekolah",
    "website": "Website",
    "is_surabaya": "is_surabaya",
    "is_website": "is_website",
    "is_agama": "is_agama",
}


NUMERIC_COERCE = ["Peserta Didik", "Rombel", "Guru", "Pegawai", "R.Kelas", "R.Lab", "R.Perpus"]


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric coercion + zero/NaN mean-fills, as in train_cv.py."""
    out = df.copy()
    for col in NUMERIC_COERCE:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in MEAN_FILL:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            nonzero_mean = out.loc[out[col] != 0, col].mean()
            out[col] = out[col].replace(0, nonzero_mean).fillna(nonzero_mean)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio features from train_cv.py."""
    out = df.copy()
    out["rombel_luas_ratio"] = out["Rombel"] / out["luas tanah"]
    out["total_rooms"] = out["R.Kelas"] + out["R.Lab"] + out["R.Perpus"]
    out["total_rooms_luas_ratio"] = out["total_rooms"] / out["luas tanah"]
    out["total_rooms_listrik_ratio"] = out["total_rooms"] / out["meter listrik"]
    out["listrik_cost_rooms_ratio"] = out["harga_listrik_per_bulan"] / out["total_rooms"]
    return out


class SchoolFeaturizer:
    """fit() on the training frame, transform() anything into model features."""

    def __init__(self) -> None:
        self.features: list[str] = []
        self.categorical: list[str] = []
        self.median_fills: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, target: str = "SPP") -> SchoolFeaturizer:
        data = add_features(base_frame(df)).replace([np.inf, -np.inf], np.nan)
        features = [c for c in data.columns if c not in NON_FEATURE + [target]]
        self.features = features
        # Same categorical heuristic as train_cv.py: non-numeric dtype or low
        # cardinality, except R.Perpus which is genuinely numeric.
        # (is_numeric_dtype, not dtype == object: pandas 3 gives text columns
        # a str dtype, and .median() on those raises instead of returning NaN.)
        self.categorical = [
            c for c in features
            if c != "R.Perpus"
            and (
                not pd.api.types.is_numeric_dtype(data[c])
                or data[c].nunique(dropna=False) < 10
            )
        ]
        for c in features:
            if c not in self.categorical:
                self.median_fills[c] = float(data[c].median())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = add_features(base_frame(df)).replace([np.inf, -np.inf], np.nan)
        for c in self.features:
            if c not in data.columns:
                data[c] = np.nan
        for c in self.categorical:
            data[c] = data[c].astype(str)
        for c, med in self.median_fills.items():
            data[c] = pd.to_numeric(data[c], errors="coerce").fillna(med)
        return data[self.features]

    def school_to_row(self, school: dict) -> pd.DataFrame:
        """Map one schools.db row dict into the training frame schema."""
        row = {}
        for db_col, train_col in DB_TO_TRAINING.items():
            if db_col in school:
                row[train_col] = school[db_col]
        return pd.DataFrame([row])

    def state(self) -> dict:
        return {"features": self.features, "categorical": self.categorical,
                "median_fills": self.median_fills}

    @classmethod
    def from_state(cls, state: dict) -> SchoolFeaturizer:
        f = cls()
        f.features = state["features"]
        f.categorical = state["categorical"]
        f.median_fills = state["median_fills"]
        return f
