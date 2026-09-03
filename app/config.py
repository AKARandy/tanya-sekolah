"""Central configuration for the Sekolah Assistant.

Everything is driven by environment variables (see .env.example) so the same
code runs against a free hosted API, the Qwen-on-Kaggle zrok tunnel, Ollama,
or any OpenAI-compatible endpoint.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "schools.db"
CHROMA_PATH = BASE_DIR / "chroma_db"

# Source CSVs
SURABAYA_CSV = DATA_DIR / "Data Sekolah - Copy of Data FULL Surabaya.csv"
SIDOARJO_CSV = DATA_DIR / "Data Sekolah - Sidoarjo FULL.csv"
GABUNGAN_CSV = DATA_DIR / "Gabungan_Surabaya_Sidoarjo.csv"

# --- LLM backend (OpenAI-compatible) ---
# Options documented in .env.example:
#   groq | openrouter | ollama | qwen-kaggle | custom | openai
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")  # overrides the provider preset
LLM_MODEL = os.getenv("LLM_MODEL", "")

CHAT_MODEL = LLM_MODEL or os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")

# --- Embeddings ---
# Supported fastembed multilingual models (must match TextEmbedding.list_supported_models()):
#   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (light, default)
#   sentence-transformers/paraphrase-multilingual-mpnet-base-v2
#   intfloat/multilingual-e5-large                               (heavier, stronger)
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# --- App ---
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# Provider presets; LLM_BASE_URL env var always wins over these.
PROVIDER_PRESETS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.3-70b-instruct:free"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.1"},
    "qwen-kaggle": {"base_url": "http://localhost:8000/v1", "model": "qwen"},  # point at your zrok tunnel URL
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "custom": {"base_url": "", "model": ""},
}


def resolve_llm() -> tuple[str, str]:
    """Return (base_url, model) after applying the provider preset and env overrides."""
    preset = PROVIDER_PRESETS.get(LLM_PROVIDER, PROVIDER_PRESETS["custom"])
    base_url = LLM_BASE_URL or preset["base_url"]
    model = CHAT_MODEL if LLM_MODEL else preset["model"]
    return base_url, model
