"""
Embedding provider abstraction (Section 39).

The app must remain fully functional with zero external API keys. The
"mock" provider produces deterministic, semantically-crude but consistent
vectors via hashed n-gram bag-of-words projected into a fixed dimension --
good enough for exact/near-duplicate-phrase retrieval on a small synthetic
corpus, and enough to demonstrate the full pgvector pipeline end to end.
Swap EMBEDDING_PROVIDER=openai (with OPENAI_API_KEY set) for real
embeddings without changing any calling code.
"""
import hashlib
import math
import re
from abc import ABC, abstractmethod

from app.core.config import get_settings

EMBEDDING_DIM = 384


class EmbeddingProvider(ABC):
    dim: int = EMBEDDING_DIM

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashed bag-of-words embedding. No network calls."""

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Real embeddings via OpenAI's API. Requires OPENAI_API_KEY."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def embed(self, text: str) -> list[float]:
        import openai  # imported lazily so it's not a hard dependency

        client = openai.OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding


def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.EMBEDDING_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAIEmbeddingProvider(settings.OPENAI_API_KEY)
    return MockEmbeddingProvider()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
