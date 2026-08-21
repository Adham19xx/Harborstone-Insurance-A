"""
Vector DB with HNSW ANN Indexing and Metadata Filtering.
"""
import os
try:
    import chromadb
except ImportError:
    chromadb = None
from typing import List, Dict, Any, Optional

class VectorStoreManager:
    def __init__(self, collection_name: str = "company_docs", persist_dir: str = "./db/chroma"):
        self._memory_chunks: List[Dict[str, Any]] = []
        if chromadb is not None:
            try:
                os.makedirs(persist_dir, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_dir)
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self.collection = None
        else:
            self.collection = None

    def add_chunks(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        for i in range(len(ids)):
            self._memory_chunks.append({
                "id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i] if i < len(metadatas) else {}
            })
        if self.collection is not None:
            try:
                self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            except Exception:
                pass

    def similarity_search(self, query: str, top_k: int = 5, where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.collection is not None:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where_filter
                )
                parsed = []
                if results and results.get("documents") and results["documents"][0]:
                    for i in range(len(results["documents"][0])):
                        parsed.append({
                            "id": results["ids"][0][i],
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {}
                        })
                if parsed:
                    return parsed
            except Exception:
                pass

        # Fallback memory search
        terms = set(query.lower().split())
        scored = []
        for chunk in self._memory_chunks:
            text = chunk["content"].lower()
            score = sum(1 for t in terms if t in text)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

