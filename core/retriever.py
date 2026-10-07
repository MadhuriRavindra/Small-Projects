"""Hard filters first (budget, RAM, GPU...), then semantic ranking for soft preferences."""
from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from core import config
from core.embeddings import embed_query
from core.store import get_client


@dataclass
class Requirements:
    query: str = ""                       # free-text soft preferences ("good for coding, light")
    budget_max: float | None = None       # INR
    budget_min: float | None = None
    min_ram_gb: int | None = None
    min_storage_gb: int | None = None
    max_weight_kg: float | None = None
    needs_dedicated_gpu: bool | None = None
    categories: list[str] | None = None
    brands: list[str] | None = None
    min_rating: float | None = None


def build_filter(req: Requirements) -> Filter | None:
    must = []
    if req.budget_max is not None or req.budget_min is not None:
        must.append(FieldCondition(key="price_inr", range=Range(gte=req.budget_min, lte=req.budget_max)))
    if req.min_ram_gb is not None:
        must.append(FieldCondition(key="ram_gb", range=Range(gte=req.min_ram_gb)))
    if req.min_storage_gb is not None:
        must.append(FieldCondition(key="storage_gb", range=Range(gte=req.min_storage_gb)))
    if req.max_weight_kg is not None:
        must.append(FieldCondition(key="weight_kg", range=Range(lte=req.max_weight_kg)))
    if req.min_rating is not None:
        must.append(FieldCondition(key="rating", range=Range(gte=req.min_rating)))
    if req.needs_dedicated_gpu is not None:
        must.append(FieldCondition(key="has_dedicated_gpu", match=MatchValue(value=req.needs_dedicated_gpu)))
    if req.categories:
        must.append(FieldCondition(key="category", match=MatchAny(any=req.categories)))
    if req.brands:
        must.append(FieldCondition(key="brand", match=MatchAny(any=req.brands)))
    return Filter(must=must) if must else None


def search(req: Requirements, top_k: int = 5, client=None) -> list[dict]:
    client = client or get_client()
    text = req.query.strip() or "good laptop"
    res = client.query_points(
        collection_name=config.COLLECTION,
        query=embed_query(text),
        query_filter=build_filter(req),
        limit=top_k,
        with_payload=True,
    )
    return [{"score": round(p.score, 4), **p.payload} for p in res.points]
