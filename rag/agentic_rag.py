"""
Multi-hop Reasoning Agentic RAG.
"""
from typing import List, Dict, Any
from rag.vector_store import VectorStoreManager

class AgenticRAG:
    def __init__(self, vector_store: VectorStoreManager, llm_client, max_hops: int = 3):
        self.vector_store = vector_store
        self.llm = llm_client
        self.max_hops = max_hops

    def run(self, query: str, top_k: int = 2) -> Dict[str, Any]:
        accumulated_docs = []
        doc_ids = set()
        
        for _ in range(self.max_hops):
            docs = self.vector_store.similarity_search(query, top_k=top_k)
            for d in docs:
                if d["id"] not in doc_ids:
                    doc_ids.add(d["id"])
                    accumulated_docs.append(d)
            
            context = "\n".join([d["content"] for d in accumulated_docs])
            check = self.llm.generate(f"Is context sufficient for '{query}'?\n{context}\nReply YES/NO.")
            if "YES" in check.upper():
                break

        context = "\n\n".join([d["content"] for d in accumulated_docs])
        response = self.llm.generate(f"Context:\n{context}\n\nQuery: {query}")
        return {"query": query, "response": response, "retrieved_context": accumulated_docs}
