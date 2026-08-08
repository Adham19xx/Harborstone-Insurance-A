"""
Embedding pipeline wrapper.
"""
from typing import List

class EmbeddingPipeline:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        # Mock embedding logic for testing/evaluation
        return [[0.1 * i for i in range(1536)] for _ in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1 * i for i in range(1536)] for _ in texts]
