"""Train-or-load the CatBoost SPP estimator and expose a predict API.

Trains on Gabungan_Surabaya_Sidoarjo.csv (the clean merged frame used for the
Kaggle cross-validation work) with the same featurization, but on CPU and
without the full grid search — a fixed reasonable configuration that trains
in seconds. The Kaggle best-model pkl can replace this artifact: export it
with its featurizer state into models/spp_model.pkl in the same dict format
({'model', 'featurizer'}) and this module will pick it up unchanged.

Usage:  python -m app.predict_spp --train
"""
from __future__ import annotations

import argparse
import sqlite3

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from app.config import DB_PATH, GABUNGAN_CSV, MODELS_DIR
from app.preprocessing import SchoolFeaturizer

MODEL_PATH = MODELS_DIR / "spp_model.pkl"


def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(GABUNGAN_CSV)
    df["SPP"] = pd.to_numeric(df["SPP"], errors="coerce").fillna(0).astype(int)
    df = df[df["SPP"] > 0].reset_index(drop=True)
    return df, df["SPP"], np.log1p(df["SPP"])


def train() -> None:
    df, y_spp, y_log = load_training_data()
    featurizer = SchoolFeaturizer().fit(df)
    X = featurizer.transform(df)

    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        cat_features=featurizer.categorical,
        verbose=False,
    )
    model.fit(X, y_log)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "featurizer": featurizer.state()}, MODEL_PATH)

    # Quick honest sanity check: in-sample log-space R2 and MAE in rupiah.
    pred_spp = np.expm1(model.predict(X))
    mae = float(np.mean(np.abs(pred_spp - y_spp)))
    print(f"Trained on {len(X)} schools -> {MODEL_PATH}")
    print(f"In-sample MAE (rupiah): Rp{mae:,.0f}  (full CV numbers live in the Kaggle notebooks)")


def load_model():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)


def predict_spp_for_school(school: dict) -> dict:
    """Estimate SPP for one schools.db row dict.

    Returns {'spp': float|None, 'log': float|None, 'known': bool} where
    known=True means the school has a real collected SPP and no prediction
    was needed.
    """
    if school.get("spp") is not None and school.get("spp") and school["spp"] > 0:
        return {"spp": float(school["spp"]), "log": None, "known": True}

    bundle = load_model()
    model: CatBoostRegressor = bundle["model"]
    featurizer = SchoolFeaturizer.from_state(bundle["featurizer"])

    row = featurizer.school_to_row(school)
    X = featurizer.transform(row)
    pred_log = float(model.predict(X)[0])
    pred_spp = float(np.expm1(pred_log))
    return {"spp": max(pred_spp, 0.0), "log": pred_log, "known": False}


def fetch_school(npsn: str, db_path=None) -> dict | None:
    # db_path=None resolves the module global at call time, so tests can swap
    # the database by patching app.predict_spp.DB_PATH.
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM schools WHERE npsn = ?", (str(npsn),)).fetchone()
    conn.close()
    return dict(row) if row else None


def find_school_by_name(name: str, db_path=None) -> dict | None:
    """Case-insensitive substring match; returns the closest single school."""
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM schools WHERE UPPER(nama) LIKE ? ORDER BY LENGTH(nama) LIMIT 1",
        (f"%{name.upper()}%",),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="force retrain")
    parser.add_argument("--predict", type=str, default="", help="school name to predict")
    args = parser.parse_args()

    if args.train:
        train()
    if args.predict:
        school = find_school_by_name(args.predict)
        if school is None:
            print("School not found")
        else:
            print(school["nama"], "->", predict_spp_for_school(school))
    if not args.train and not args.predict:
        train()
