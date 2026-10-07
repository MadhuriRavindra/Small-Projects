"""CSV -> list[Document]. One laptop = one document (no chunking needed)."""
import math
from pathlib import Path

import pandas as pd

from core.schema import Document

# columns stored as filterable payload
META_COLS = [
    "brand", "model", "category", "cpu", "ram_gb", "storage_gb", "gpu",
    "has_dedicated_gpu", "screen_in", "resolution", "refresh_hz", "weight_kg",
    "battery_wh", "os", "price_inr", "currency", "rating", "pros", "cons", "good_for",
]


def _clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):  # numpy scalar -> python
        return v.item()
    return v


def load_laptops_csv(path: Path) -> list[Document]:
    df = pd.read_csv(path)
    docs = []
    for _, r in df.iterrows():
        meta = {c: _clean(r[c]) for c in META_COLS if c in df.columns}
        docs.append(Document(
            doc_id=str(r["product_id"]),
            text=str(r["description"]),
            source_type="csv",
            source_name=str(r.get("source_name", "laptops_india.csv")),
            product_id=str(r["product_id"]),
            fetched_at=str(r.get("fetched_at", "")) or None,
            meta=meta,
        ))
    return docs
