"""Build the ChromaDB vector index of school profile cards (classic RAG route).

Each school becomes one Indonesian-language text card covering the facets a
person would actually ask about: identity, location, level, accreditation,
size, foundation, religion, website, tuition. Contact details (phones, emails,
operator names) are deliberately left out — see build_db.py.

Embeddings: fastembed multilingual-e5-small (ONNX, no torch dependency),
following the e5 convention of "passage: " prefixes for documents and
"query: " prefixes for queries.

Usage:  python -m app.build_index
"""
from __future__ import annotations

import sqlite3

import chromadb
from fastembed import TextEmbedding

from app.config import CHROMA_PATH, DB_PATH, EMBED_MODEL

COLLECTION = "schools"

JENJANG_ID = {
    "SD": "Sekolah Dasar (SD)",
    "SMP": "Sekolah Menengah Pertama (SMP)",
    "SMA": "Sekolah Menengah Atas (SMA)",
    "SMK": "Sekolah Menengah Kejuruan (SMK)",
    "MI": "Madrasah Ibtidaiyah (MI)",
    "MTs": "Madrasah Tsanawiyah (MTs)",
    "MA": "Madrasah Aliyah (MA)",
    "PKBM": "PKBM (Pusat Kegiatan Belajar Masyarakat)",
}


def _fmt(value) -> str:
    if value is None:
        return "-"
    s = str(value).strip()
    return s if s and s.lower() != "nan" else "-"


def school_card(row: sqlite3.Row | dict) -> str:
    """Human-readable profile card; this text is what gets embedded."""
    jenjang = JENJANG_ID.get((row["jenjang"] or "").upper(), _fmt(row["jenjang"]))
    parts = [
        f"Nama sekolah: {_fmt(row['nama'])}.",
        f"Jenjang: {jenjang}.",
    ]
    if row["kota"] == "Kota Surabaya":
        parts.append(f"Jenis: {_fmt(row['status']) or 'Swasta'}.")
    parts += [
        f"Lokasi: {_fmt(row['kecamatan']).replace('KEC. ', '').replace('Kec. ', '').title()}, {_fmt(row['kota'])}.",
        f"Akreditasi: {_fmt(row['akreditasi'])}.",
        f"Jumlah peserta didik: {_fmt(row['peserta_didik'])}, dengan {_fmt(row['guru'])} guru.",
        f"Yayasan: {_fmt(row['yayasan'])}"
        + (f" (grade {_fmt(row['grade_yayasan'])})." if row["grade_yayasan"] else "."),
        f"Sekolah {'berbasis agama' if row['is_agama'] == 1 else 'umum/non-agama'}.",
    ]
    if row["spp"] and row["spp"] > 0:
        parts.append(f"SPP tercatat: Rp{int(row['spp']):,}.".replace(",", "."))
    elif row["prediksi_spp"] and row["prediksi_spp"] > 0:
        parts.append(f"Perkiraan SPP dari model prediksi: Rp{int(row['prediksi_spp']):,}.".replace(",", "."))
    if row["website"]:
        parts.append(f"Website: {_fmt(row['website'])}.")
    return " ".join(p for p in parts if p)


def _embedder() -> TextEmbedding:
    return TextEmbedding(model_name=EMBED_MODEL)


def _prefixes() -> tuple[str, str]:
    """e5-family models need 'query: '/'passage: ' prefixes; others use none."""
    if "e5" in EMBED_MODEL:
        return "query: ", "passage: "
    return "", ""


def _embed_texts(model: TextEmbedding, texts: list[str], prefix: str) -> list[list[float]]:
    return [vec.tolist() for vec in model.embed([f"{prefix}{t}" for t in texts])]


def get_collection() -> tuple[chromadb.Collection, TextEmbedding]:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION)
    return collection, _embedder()


def semantic_search(query: str, k: int = 5) -> list[dict]:
    """Top-k schools for a natural-language query; used by the RAG route."""
    collection, embedder = get_collection()
    qvec = _embed_texts(embedder, [query], "query: ")[0]
    res = collection.query(query_embeddings=[qvec], n_results=k,
                           include=["documents", "metadatas", "distances"])
    hits = []
    for i, npsn in enumerate(res["ids"][0]):
        hits.append({
            "npsn": npsn,
            "card": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        })
    return hits


def build(db_path=DB_PATH, chroma_path=CHROMA_PATH) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT npsn, nama, jenjang, status, kota, kecamatan, kelurahan,
               akreditasi, peserta_didik, guru, spp, prediksi_spp, kurikulum,
               yayasan, jumlah_sekolah_naungan, grade_yayasan,
               is_agama, is_website, website
        FROM schools
        """
    ).fetchall()
    conn.close()

    embedder = _embedder()

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(COLLECTION)  # idempotent rebuild
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    batch = 256
    total = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        docs = [school_card(dict(r)) for r in chunk]
        collection.add(
            ids=[r["npsn"] for r in chunk],
            documents=docs,
            embeddings=_embed_texts(embedder, docs, "passage: "),
            metadatas=[
                {
                    "npsn": r["npsn"],
                    "nama": _fmt(r["nama"]),
                    "jenjang": _fmt(r["jenjang"]),
                    "kota": _fmt(r["kota"]),
                    "kecamatan": _fmt(r["kecamatan"]),
                }
                for r in chunk
            ],
        )
        total += len(chunk)
        print(f"  indexed {total}/{len(rows)}")

    print(f"Indexed {collection.count()} schools into {chroma_path}")
    return collection.count()


if __name__ == "__main__":
    build()
