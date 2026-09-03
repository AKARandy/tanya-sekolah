"""Build schools.db from the scraped Surabaya + Sidoarjo CSVs.

The two source schemas differ (Surabaya was scraped per-facility, Sidoarjo per
registration), so every column mapping is explicit here. Contact fields
(phones, emails, operator/kepsek names, yayasan leadership) are stored in the
database for completeness but deliberately EXCLUDED from the vector index's
embedding cards in build_index.py — this dataset is public government data,
but embedding personal contact details is unnecessary for the Q&A use case.

Usage:  python -m app.build_db
"""
from __future__ import annotations

import sqlite3
import sys

import pandas as pd

from app.config import DB_PATH, SIDOARJO_CSV, SURABAYA_CSV

# ---------------------------------------------------------------------------
# Target schema: school-level facts one row per school (NPSN is the key).
# ---------------------------------------------------------------------------
SCHOOL_COLUMNS = [
    "npsn", "nama", "jenjang", "status", "kota", "kecamatan", "kelurahan",
    "alamat", "akreditasi", "spp", "prediksi_spp", "kurikulum",
    "peserta_didik", "rombel", "guru", "pegawai", "r_kelas", "r_lab", "r_perpus",
    "umur", "tahun_sk_pendirian", "meter_listrik", "harga_listrik_bulan",
    "luas_tanah", "harga_tanah_kecamatan", "estimasi_harga_tanah",
    "yayasan", "jumlah_sekolah_naungan", "grade_yayasan",
    "is_agama", "is_website", "website", "is_surabaya",
    # contact fields — kept in DB, excluded from embeddings
    "telepon", "email", "kepala_sekolah", "operator", "pimpinan_yayasan",
]

YAYASAN_COLUMNS = ["nama", "jumlah_sekolah_naungan", "grade", "pimpinan"]


def _to_num(series: pd.Series) -> pd.Series:
    """Coerce messy scraped numerics ('Rp220,000', '-', '', '2.000 ') to float."""
    s = series.astype(str).str.replace(r"[Rp\s,\.](?=\d{3})", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s.str.extract(r"(-?[\d.]+)")[0], errors="coerce")


def _clean_year(series: pd.Series) -> pd.Series:
    """SK pendirian appears as plain years, Excel serial dates, or datetimes.

    Excel serials (days since 1899-12-30) show up when the spreadsheet stored
    a real date in a column we read as a number, e.g. 44789 -> 2022.
    """
    num = pd.to_numeric(series, errors="coerce")
    year = num.where((num >= 1900) & (num <= 2100))
    serial = num.where((num > 20000) & (num < 80000))
    serial_year = pd.to_datetime(
        serial, unit="D", origin="1899-12-30", errors="coerce"
    ).dt.year
    as_dt = pd.to_datetime(series, errors="coerce").dt.year
    return year.fillna(serial_year).fillna(as_dt)


def load_surabaya() -> pd.DataFrame:
    df = pd.read_csv(SURABAYA_CSV)
    out = pd.DataFrame(index=df.index)
    out["npsn"] = df["NPSN"].astype(str)
    out["nama"] = df["Nama_sekolah"]
    out["jenjang"] = df["Jenjang"]
    out["status"] = df["Status"]
    out["kota"] = "Kota Surabaya"
    out["kecamatan"] = df["Kecamatan"]
    out["alamat"] = df["Alamat lengkap"].combine_first(df["alamat lengkap"])
    out["akreditasi"] = df["Akreditasi"]
    out["spp"] = _to_num(df["SPP"])
    out["prediksi_spp"] = pd.NA
    out["kurikulum"] = df["Kurikulum"]
    out["peserta_didik"] = pd.to_numeric(df["Peserta Didik"], errors="coerce")
    out["rombel"] = pd.to_numeric(df["Rombel"], errors="coerce")
    out["guru"] = pd.to_numeric(df["Guru"], errors="coerce")
    out["pegawai"] = pd.to_numeric(df["Pegawai"], errors="coerce")
    out["r_kelas"] = pd.to_numeric(df["R.Kelas"], errors="coerce")
    out["r_lab"] = pd.to_numeric(df["R.Lab"], errors="coerce")
    out["r_perpus"] = pd.to_numeric(df["R.Perpus"], errors="coerce")
    out["umur"] = pd.to_numeric(df["Umur"], errors="coerce")
    out["tahun_sk_pendirian"] = _clean_year(df["Tanggal SK Pendirian"])
    out["meter_listrik"] = _to_num(df["meter listrik"])
    out["harga_listrik_bulan"] = _to_num(df["harga_listrik_per_bulan"])
    out["luas_tanah"] = _to_num(df["luas tanah"])
    out["harga_tanah_kecamatan"] = _to_num(df["Harga Tanah Pasaran per kecamatan"])
    out["estimasi_harga_tanah"] = _to_num(df["estimasi harga tanah sekolah"])
    out["yayasan"] = df["Yayasan"]
    out["jumlah_sekolah_naungan"] = pd.to_numeric(df["Jumlah Sekolah Naungan Yayasan"], errors="coerce")
    out["grade_yayasan"] = df["Grade Yayasan Sekolah"]
    out["is_agama"] = pd.to_numeric(df["is_agama"], errors="coerce")
    out["is_website"] = pd.to_numeric(df["is_website"], errors="coerce")
    out["website"] = df["Website"]
    out["is_surabaya"] = 1
    out["telepon"] = df["Telepon"].combine_first(df["Telepon_sekolah"])
    out["email"] = df["Email_sekolah"]
    out["kepala_sekolah"] = df["Nama_Kepsek"]
    out["operator"] = df["Nama_Operator_Sekolah"]
    out["pimpinan_yayasan"] = df["Pimpinan Yayasan"]
    return out


def load_sidoarjo() -> pd.DataFrame:
    df = pd.read_csv(SIDOARJO_CSV)
    out = pd.DataFrame(index=df.index)
    out["npsn"] = df["NPSN"].astype(str)
    out["nama"] = df["Nama_sekolah"]
    out["jenjang"] = df["Jenjang"]
    out["status"] = pd.NA  # not scraped for Sidoarjo
    out["kota"] = "Kab. Sidoarjo"
    out["kecamatan"] = df["Kecamatan"]
    out["kelurahan"] = df["Kelurahan"]
    out["alamat"] = df["Alamat lengkap"]
    out["akreditasi"] = df["Akreditasi"]
    out["spp"] = _to_num(df["SPP"])
    out["prediksi_spp"] = _to_num(df["Prediksi SPP"])
    out["kurikulum"] = df["Kurikulum"]
    out["peserta_didik"] = pd.to_numeric(df["Peserta Didik"], errors="coerce")
    out["rombel"] = pd.to_numeric(df["Rombel"], errors="coerce")
    out["guru"] = pd.to_numeric(df["Guru"], errors="coerce")
    out["pegawai"] = pd.to_numeric(df["Pegawai"], errors="coerce")
    out["r_kelas"] = pd.to_numeric(df["R.Kelas"], errors="coerce")
    out["r_lab"] = pd.to_numeric(df["R.Lab"], errors="coerce")
    out["r_perpus"] = pd.to_numeric(df["R.Perpus"], errors="coerce")
    out["umur"] = pd.to_numeric(df["Umur"], errors="coerce")
    out["tahun_sk_pendirian"] = _clean_year(df["Tanggal SK Pendirian"])
    out["meter_listrik"] = _to_num(df["meter listrik"])
    out["harga_listrik_bulan"] = _to_num(df["harga_listrik_per_bulan"])
    out["luas_tanah"] = _to_num(df["luas tanah"])
    out["harga_tanah_kecamatan"] = _to_num(df["Harga Tanah Pasaran per kecamatan"])
    out["estimasi_harga_tanah"] = _to_num(df["estimasi harga tanah sekolah"])
    out["yayasan"] = df["Yayasan"]
    out["jumlah_sekolah_naungan"] = pd.to_numeric(df["Jumlah Sekolah Naungan Yayasan"], errors="coerce")
    out["grade_yayasan"] = df["Grade Yayasan Sekolah"]
    out["is_agama"] = pd.to_numeric(df["is_agama"], errors="coerce")
    out["is_website"] = pd.to_numeric(df["is_website"], errors="coerce")
    out["website"] = df["Website"]
    out["is_surabaya"] = 0
    out["telepon"] = df["No.Telp"]
    out["email"] = pd.NA
    out["kepala_sekolah"] = df["Kepala Sekolah"]
    out["operator"] = df["Operator"]
    out["pimpinan_yayasan"] = pd.NA

    # Sidoarjo scrapes some schools twice (Dapodik + sekolah.data portals);
    # keep the first occurrence per NPSN.
    return out.drop_duplicates(subset="npsn", keep="first")


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    schools = pd.concat([load_surabaya(), load_sidoarjo()], ignore_index=True)

    # Normalize ids and text keys
    schools["npsn"] = schools["npsn"].str.strip()
    schools = schools.drop_duplicates(subset="npsn", keep="first")

    # Fill missing foundations BEFORE deriving the key: pandas 3 keeps NA
    # through astype(str) (pandas 2 produced 'nan'), which left NULL keys.
    yay = schools["yayasan"].astype("string").str.strip()
    has_yayasan = yay.notna() & yay.ne("") & yay.str.lower().ne("nan")
    schools["yayasan"] = yay.where(has_yayasan, "Tidak diketahui")
    schools["yayasan_key"] = schools["yayasan"].str.upper()

    yayasan = (
        schools.groupby("yayasan_key")
        .agg(
            nama=("yayasan", "first"),
            jumlah_sekolah_naungan=("jumlah_sekolah_naungan", "max"),
            grade=("grade_yayasan", "first"),
            pimpinan=("pimpinan_yayasan", "first"),
        )
        .dropna(subset=["nama"])
        .query("nama != 'Tidak diketahui'")
        .reset_index(drop=True)[YAYASAN_COLUMNS]
    )

    return schools[SCHOOL_COLUMNS + ["yayasan_key"]], yayasan


def write_db(schools: pd.DataFrame, yayasan: pd.DataFrame, db_path=DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS schools")
        conn.execute("DROP TABLE IF EXISTS yayasan")
        schools.to_sql("schools", conn, index=False)
        yayasan.to_sql("yayasan", conn, index=False)
        conn.execute(
            "CREATE INDEX idx_schools_jenjang ON schools(jenjang)"
        )
        conn.execute(
            "CREATE INDEX idx_schools_kecamatan ON schools(kecamatan)"
        )
        conn.execute(
            "CREATE INDEX idx_schools_yayasan_key ON schools(yayasan_key)"
        )
    print(f"Wrote {len(schools)} schools and {len(yayasan)} yayasan to {db_path}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        s, y = build()
        print(s.head(3).to_string())
        print(y.head(3).to_string())
    else:
        s, y = build()
        write_db(s, y)
