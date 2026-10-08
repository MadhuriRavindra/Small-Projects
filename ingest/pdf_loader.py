"""PDFs in data/pdfs/ -> chunked Documents linked to a product.

How a PDF is linked to a laptop (first match wins):
  1. data/pdfs/pdf_map.csv   (columns: filename,product_id)  - explicit override
  2. the laptop's model name appears in the file name, e.g. lenovo_legion_5_15arp9.pdf
Unmatched PDFs are skipped with a warning (name them after the laptop or add them to pdf_map.csv).
Works page by page, so large PDFs are fine. Scanned PDFs (no text layer) need OCR - they are reported.
"""
import datetime
import re
from pathlib import Path

import pandas as pd

from core import config
from core.matching import build_alias_index, match_text, norm
from core.schema import Document
from ingest.catalog import load_catalog
from ingest.chunking import chunk_text

PDF_DIR = config.DATA_DIR / "pdfs"


def _load_map(pdf_dir: Path = PDF_DIR) -> dict[str, str]:
    p = Path(pdf_dir) / "pdf_map.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype=str)
    return {r.filename.strip(): r.product_id.strip() for r in df.itertuples()}


def resolve_product(filename: str, alias_index, explicit: dict) -> tuple[str | None, str]:
    if filename in explicit:
        return explicit[filename], "pdf_map.csv"
    ids = match_text(Path(filename).stem.replace("_", " ").replace("-", " "), alias_index)
    if len(ids) == 1:
        return next(iter(ids)), "filename"
    if len(ids) > 1:
        return None, f"ambiguous filename (matches {sorted(ids)}); add it to pdf_map.csv"
    return None, "no laptop name found in filename; rename it or add it to pdf_map.csv"


def load_pdf_documents(pdf_dir: Path = PDF_DIR, max_pages: int | None = None) -> list[Document]:
    from pypdf import PdfReader

    catalog = load_catalog()
    by_id = {c["product_id"]: c for c in catalog}
    alias_index = build_alias_index(catalog)
    explicit = _load_map(pdf_dir)
    today = datetime.date.today().isoformat()
    docs: list[Document] = []

    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        print(f"  no PDFs found in {pdf_dir}")
    for path in pdfs:
        pid, how = resolve_product(path.name, alias_index, explicit)
        if not pid or pid not in by_id:
            print(f"  SKIP {path.name}: {how}")
            continue
        prod = by_id[pid]
        reader = PdfReader(str(path))
        n_pages, n_chunks, empty = len(reader.pages), 0, 0
        for pno, page in enumerate(reader.pages, start=1):
            if max_pages and pno > max_pages:
                break
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if len(text.strip()) < 20:
                empty += 1
                continue
            for i, ch in enumerate(chunk_text(text)):
                docs.append(Document(
                    doc_id=f"{pid}#pdf#{norm(path.stem).replace(' ', '-')}#p{pno}-{i}",
                    text=f"{prod['brand']} {prod['model']} - {ch}",
                    source_type="pdf",
                    source_name=f"PDF: {path.name} (p.{pno})",
                    product_id=pid,
                    fetched_at=today,
                    meta={"brand": prod["brand"], "model": prod["model"], "file": path.name, "page": pno},
                ))
                n_chunks += 1
        msg = f"  {path.name} -> {pid} ({prod['brand']} {prod['model']}) via {how}: {n_chunks} chunks from {n_pages} pages"
        if empty:
            msg += f"  [{empty} pages had no text - scanned? needs OCR]"
        print(msg)
    return docs
