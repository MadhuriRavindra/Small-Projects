"""Embedding wrapper. Real model = fastembed (ONNX). 'fake' = hashed bag-of-words for offline tests."""
import hashlib
import math
import re
from functools import lru_cache

from core import config


class _FakeEmbedder:
    """Deterministic, no downloads. Only for tests - NOT for real retrieval quality."""
    def __init__(self, dim: int):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts):
        return [self._vec(t) for t in texts]


class _FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts):
        return [v.tolist() for v in self.model.embed(list(texts))]


@lru_cache(maxsize=1)
def _get():
    if config.EMBEDDER == "fake":
        return _FakeEmbedder(config.EMBED_DIM)
    return _FastEmbedder(config.EMBED_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _get().embed(texts)


def embed_query(query: str) -> list[float]:
    return _get().embed([query])[0]
