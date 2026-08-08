"""
Vector DB with HNSW ANN Indexing and Metadata Filtering.
"""
import os
import chromadb
from typing import List, Dict, Any, Optional

class VectorStoreManager:
    def __init__(self, collection_name: str = "company_docs", persist_dir: str = "./db/chroma"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def similarity_search(self, query: str, top_k: int = 5, where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
        return parsed
