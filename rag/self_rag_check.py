"""
Self-RAG Verification Checks for Groundedness and Relevance.
"""
from typing import List, Dict, Any

class SelfRAGVerifier:
    def __init__(self, llm_client):
        self.llm = llm_client

    def verify_relevance(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relevant = []
        for chunk in chunks:
            res = self.llm.generate(f"Is chunk '{chunk.get('content')}' relevant to '{query}'? Reply YES/NO.")
            if "YES" in res.upper():
                relevant.append(chunk)
        return relevant

    def verify_groundedness(self, response: str, chunks: List[Dict[str, Any]]) -> bool:
        if not chunks:
            return False
        context = "\n".join([c.get("content", "") for c in chunks])
        res = self.llm.generate(f"Context:\n{context}\n\nAnswer:\n{response}\nIs the answer grounded? Reply YES/NO.")
        return "YES" in res.upper()
