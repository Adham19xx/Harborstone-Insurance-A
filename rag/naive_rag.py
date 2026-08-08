"""
Baseline Naive RAG Architecture.
"""
from typing import List, Dict, Any
from rag.vector_store import VectorStoreManager

class NaiveRAG:
    def __init__(self, vector_store: VectorStoreManager, llm_client):
        self.vector_store = vector_store
        self.llm = llm_client

    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        docs = self.vector_store.similarity_search(query, top_k=top_k)
        context = "\n\n".join([d["content"] for d in docs])
        response = self.llm.generate(f"Context:\n{context}\n\nQuery: {query}")
        return {"query": query, "response": response, "retrieved_context": docs}
