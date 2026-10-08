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
    intent: str = "recommend"            # recommend | clarify | other (set by the parse step)


def build_filter(req: Requirements) -> Filter | None:
    must = [FieldCondition(key="source_type", match=MatchValue(value="csv"))]  # product cards only
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
    return Filter(must=must)


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


def search_evidence(query: str, product_ids: list[str], per_product: int = 2, client=None) -> list[dict]:
    """Supporting chunks (PDF / web / Reddit) for the products already chosen.
    Pass 1 takes the best chunk of each source type per product (so answers can mix spec sheets and
    opinions); pass 2 fills up to per_product."""
    if not product_ids:
        return []
    client = client or get_client()
    flt = Filter(
        must=[FieldCondition(key="product_id", match=MatchAny(any=list(product_ids)))],
        must_not=[FieldCondition(key="source_type", match=MatchValue(value="csv"))],
    )
    res = client.query_points(
        collection_name=config.COLLECTION,
        query=embed_query(query.strip() or "laptop review"),
        query_filter=flt,
        limit=min(80, max(12, per_product * len(product_ids) * 4)),
        with_payload=True,
    )
    pts = sorted(res.points, key=lambda p: -p.score)
    taken, seen_type, count = [], set(), {}
    for p in pts:  # pass 1: one per (product, source type)
        pl = p.payload
        key = (pl["product_id"], pl["source_type"])
        if key in seen_type or count.get(pl["product_id"], 0) >= per_product:
            continue
        seen_type.add(key); count[pl["product_id"]] = count.get(pl["product_id"], 0) + 1
        taken.append(p)
    for p in pts:  # pass 2: fill
        if p in taken or count.get(p.payload["product_id"], 0) >= per_product:
            continue
        count[p.payload["product_id"]] = count.get(p.payload["product_id"], 0) + 1
        taken.append(p)
    order = {pid: i for i, pid in enumerate(product_ids)}
    taken.sort(key=lambda p: (order.get(p.payload["product_id"], 99), -p.score))
    return [{**p.payload, "similarity": round(p.score, 4)} for p in taken]


def search_guides(query: str, top_k: int = 2, client=None) -> list[dict]:
    """General buying-guide chunks (not tied to one laptop) that best match the question."""
    client = client or get_client()
    flt = Filter(must=[FieldCondition(key="source_type", match=MatchValue(value="guide"))])
    res = client.query_points(
        collection_name=config.COLLECTION,
        query=embed_query(query.strip() or "how to choose a laptop"),
        query_filter=flt, limit=top_k, with_payload=True,
    )
    return [{**p.payload, "similarity": round(p.score, 4)} for p in res.points]
