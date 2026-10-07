"""Central settings. Reads .env locally; on Streamlit Cloud use st.secrets -> env vars."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "structured" / "laptops_india.csv"

QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None
QDRANT_LOCAL_PATH = str(ROOT / "qdrant_local")
COLLECTION = os.getenv("QDRANT_COLLECTION", "laptops")

# 384-dim, small, runs on CPU via ONNX (no torch needed) -> fine for free hosting
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384
# Set EMBEDDER=fake for offline tests (no model download)
EMBEDDER = os.getenv("EMBEDDER", "fastembed")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")          # writes the final answer
GROQ_PARSE_MODEL = os.getenv("GROQ_PARSE_MODEL", "llama-3.1-8b-instant")  # cheap/fast: extracts filters