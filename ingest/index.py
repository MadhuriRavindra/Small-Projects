"""Run on YOUR machine (not in the deployed app).

  python -m ingest.index --recreate                       # rebuild from the CSV only
  python -m ingest.index --sources csv pdf web reddit     # add more inputs (additive upsert)
  python -m ingest.index --sources pdf                    # re-index just the PDFs

--recreate DROPS the whole collection, so combine it with every source you want to keep.
"""
import argparse

from qdrant_client.models import PointStruct

from core import config
from core.embeddings import embed_texts
from core.store import ensure_collection, get_client
from ingest.csv_loader import load_laptops_csv

BATCH = 32


def index_documents(docs, recreate=False):
    client = get_client()
    ensure_collection(client, recreate=recreate)
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i + BATCH]
        vectors = embed_texts([d.text for d in batch])
        points = [PointStruct(id=d.point_id, vector=v, payload=d.to_payload()) for d, v in zip(batch, vectors)]
        client.upsert(collection_name=config.COLLECTION, points=points)
        print(f"  indexed {i + len(batch)}/{len(docs)}")
    print("collection size:", client.count(config.COLLECTION).count)
    return client


def load_sources(sources):
    docs = []
    for s in sources:
        print(f"[{s}]")
        if s == "csv":
            docs += load_laptops_csv(config.CSV_PATH)
        elif s == "pdf":
            from ingest.pdf_loader import load_pdf_documents
            docs += load_pdf_documents()
        elif s == "web":
            from ingest.web_loader import load_web_documents
            docs += load_web_documents()
        elif s == "reddit":
            from ingest.reddit_loader import load_reddit_documents
            docs += load_reddit_documents()
    return docs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    ap.add_argument("--sources", nargs="+", default=["csv"], choices=["csv", "pdf", "web", "reddit"])
    args = ap.parse_args()
    docs = load_sources(args.sources)
    print(f"loaded {len(docs)} documents from: {', '.join(args.sources)}")
    index_documents(docs, recreate=args.recreate)
