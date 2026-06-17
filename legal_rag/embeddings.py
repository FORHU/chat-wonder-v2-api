import hashlib
import time
from openai import OpenAI

_embedding_cache: dict = {}
_CACHE_TTL = 3600
_MAX_CACHE_ENTRIES = 2000  # ~24MB cap; prevents OOM during bulk ingestion


def _cache_key(text: str, model: str) -> str:
    return hashlib.md5(f"{model}:{text}".encode()).hexdigest()


class EmbeddingService:
    def __init__(self, api_key: str, model: str, base_url=None):
        params = {"api_key": api_key}
        if base_url:
            params["base_url"] = base_url
        self.client = OpenAI(**params)
        self.model = model

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        global _embedding_cache
        if len(_embedding_cache) > _MAX_CACHE_ENTRIES:
            _embedding_cache = {}
        now = time.time()
        vectors: list[list[float]] = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            key = _cache_key(text, self.model)
            entry = _embedding_cache.get(key)
            if entry and now - entry["ts"] < _CACHE_TTL:
                vectors[i] = entry["vec"]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        for idx in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[idx: idx + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            for j, row in enumerate(response.data):
                original_i = uncached_indices[idx + j]
                vectors[original_i] = row.embedding
                key = _cache_key(batch[j], self.model)
                _embedding_cache[key] = {"vec": row.embedding, "ts": now}

        return vectors
