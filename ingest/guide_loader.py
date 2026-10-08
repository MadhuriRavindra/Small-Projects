"""General buying guides (not tied to one laptop): data/guides/*.pdf|md|txt  -> source_type "guide".
They explain WHY a spec matters (CPU tiers, RAM, GPU, display...) and are retrieved by meaning for any question.
"""
import datetime
from pathlib import Path

from core import config
from core.schema import Document
from ingest.chunking import chunk_text

GUIDE_DIR = config.DATA_DIR / "guides"


def _read(path: Path) -> list[tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        out = []
        for n, page in enumerate(PdfReader(str(path)).pages, start=1):
            try:
                out.append((n, page.extract_text() or ""))
            except Exception:
                out.append((n, ""))
        return out
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def load_guide_documents(guide_dir: Path = GUIDE_DIR) -> list[Document]:
    today = datetime.date.today().isoformat()
    docs = []
    files = [p for p in sorted(Path(guide_dir).glob("*")) if p.suffix.lower() in {".pdf", ".md", ".txt"}]
    if not files:
        print(f"  no guides found in {guide_dir}")
    for path in files:
        title = path.stem.replace("_", " ").replace("-", " ").strip().title()
        n = 0
        for pno, text in _read(path):
            for i, ch in enumerate(chunk_text(text)):
                docs.append(Document(
                    doc_id=f"guide#{path.stem}#p{pno}-{i}",
                    text=f"{title} - {ch}",
                    source_type="guide",
                    source_name=f"Guide: {path.name}",
                    product_id=None,
                    fetched_at=today,
                    meta={"file": path.name, "page": pno, "title": title},
                ))
                n += 1
        print(f"  {path.name}: {n} chunks")
    return docs
