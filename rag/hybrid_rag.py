"""
Hybrid Search Architecture (Dense Vector + BM25 Sparse Search).
"""
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from rag.vector_store import VectorStoreManager

class HybridRAG:
    def __init__(self, vector_store: VectorStoreManager, corpus: List[Dict[str, Any]], llm_client):
        self.vector_store = vector_store
        self.corpus = corpus
        self.llm = llm_client
        tokenized_corpus = [doc["content"].lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        # Vector search
        v_results = self.vector_store.similarity_search(query, top_k=top_k*2)
        
        # BM25 search
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k*2]
        bm25_results = [self.corpus[i] for i in top_indices if scores[i] > 0]

        # RRF Fusion
        doc_map = {d["id"]: d for d in v_results + bm25_results}
        rrf_scores = {}
        for rank, d in enumerate(v_results):
            rrf_scores[d["id"]] = rrf_scores.get(d["id"], 0) + 1.0 / (60 + rank + 1)
        for rank, d in enumerate(bm25_results):
            rrf_scores[d["id"]] = rrf_scores.get(d["id"], 0) + 1.0 / (60 + rank + 1)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        final_docs = [doc_map[doc_id] for doc_id in sorted_ids]
        
        context = "\n\n".join([d["content"] for d in final_docs])
        response = self.llm.generate(f"Context:\n{context}\n\nQuery: {query}")
        return {"query": query, "response": response, "retrieved_context": final_docs}
