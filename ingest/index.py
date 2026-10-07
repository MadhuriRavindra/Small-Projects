"""Run on YOUR machine (not in the deployed app):  python -m ingest.index [--recreate]"""
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
        points = [PointStruct(id=d.point_id, vector=v, payload=d.to_payload())
                  for d, v in zip(batch, vectors)]
        client.upsert(collection_name=config.COLLECTION, points=points)
        print(f"  indexed {i + len(batch)}/{len(docs)}")
    print("collection size:", client.count(config.COLLECTION).count)
    return client


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    args = ap.parse_args()
    docs = load_laptops_csv(config.CSV_PATH)
    print(f"loaded {len(docs)} documents from {config.CSV_PATH.name}")
    index_documents(docs, recreate=args.recreate)
