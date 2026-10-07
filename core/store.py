"""Qdrant connection + collection setup."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from core import config

# field -> index type (these are the fields we filter on)
PAYLOAD_INDEXES = {
    "price_inr": PayloadSchemaType.FLOAT,
    "ram_gb": PayloadSchemaType.INTEGER,
    "storage_gb": PayloadSchemaType.INTEGER,
    "weight_kg": PayloadSchemaType.FLOAT,
    "battery_wh": PayloadSchemaType.FLOAT,
    "rating": PayloadSchemaType.FLOAT,
    "has_dedicated_gpu": PayloadSchemaType.BOOL,
    "category": PayloadSchemaType.KEYWORD,
    "brand": PayloadSchemaType.KEYWORD,
    "source_type": PayloadSchemaType.KEYWORD,
    "product_id": PayloadSchemaType.KEYWORD,
}


def get_client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    return QdrantClient(path=config.QDRANT_LOCAL_PATH)  # local dev, no server needed


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    name = config.COLLECTION
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )
    for field, schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(name, field_name=field, field_schema=schema)
        except Exception:
            pass  # index may already exist / local mode ignores indexes
