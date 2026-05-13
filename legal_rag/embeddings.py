from openai import OpenAI


class EmbeddingService:
    def __init__(self, api_key: str, model: str, base_url=None):
        params = {"api_key": api_key}
        if base_url:
            params["base_url"] = base_url
        self.client = OpenAI(**params)
        self.model = model

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for idx in range(0, len(texts), batch_size):
            batch = texts[idx : idx + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend([row.embedding for row in response.data])
        return vectors

