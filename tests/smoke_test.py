"""Offline end-to-end check (fake embedder + local Qdrant). Run:  python -m tests.smoke_test"""
import os, shutil
os.environ["EMBEDDER"] = "fake"
os.environ["QDRANT_URL"] = ""

from core import config
shutil.rmtree(config.QDRANT_LOCAL_PATH, ignore_errors=True)

from ingest.csv_loader import load_laptops_csv
from ingest.index import index_documents
from core.retriever import Requirements, search

docs = load_laptops_csv(config.CSV_PATH)
assert len(docs) == 48, len(docs)
client = index_documents(docs, recreate=True)


def show(title, req, k=4):
    print(f"\n== {title}")
    for h in search(req, top_k=k, client=client):
        print(f"  {h['product_id']} {h['brand']} {h['model']} | Rs {h['price_inr']:,.0f} | "
              f"{h['ram_gb']}GB | {h['category']} | gpu={h['has_dedicated_gpu']}")
    return search(req, top_k=50, client=client)


r = show("budget<=70k, >=16GB, coding", Requirements(query="good for coding and college", budget_max=70000, min_ram_gb=16))
assert r and all(h["price_inr"] <= 70000 and h["ram_gb"] >= 16 for h in r)

r = show("gaming with dedicated GPU, <=100k", Requirements(query="gaming laptop", budget_max=100000, needs_dedicated_gpu=True))
assert r and all(h["has_dedicated_gpu"] and h["price_inr"] <= 100000 for h in r)

r = show("light travel laptop <=1.3kg", Requirements(query="travel long battery", max_weight_kg=1.3))
assert r and all(h["weight_kg"] <= 1.3 for h in r)

r = show("Apple only", Requirements(query="laptop", brands=["Apple"]))
assert len(r) == 5 and all(h["brand"] == "Apple" for h in r)
print("\nALL CHECKS PASSED")
