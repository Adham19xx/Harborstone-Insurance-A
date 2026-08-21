"""
Harborstone Policy Knowledge Base & RAG Retriever.
Combines VectorStore (ChromaDB) with Hybrid search for insurance policy terms.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from rag.vector_store import VectorStoreManager
from rag.hybrid_rag import HybridRAG


HARBORSTONE_POLICY_DOCUMENTS = [
    {
        "id": "policy_hull_terms",
        "content": (
            "HARBORSTONE INSURANCE: MARINE HULL & AUTO POLICY TERMS\n"
            "1. Claims above $10,000 USD require explicit approval from a Senior Claims Officer or Manager.\n"
            "2. Storm and collision damage claims must include official meteorological verification and police reports.\n"
            "3. Evidence requirements for claim settlement: accident photos, police report, and repair estimate report.\n"
            "4. Total loss settlements are capped at the declared vessel or vehicle market value minus policy deductible."
        ),
        "metadata": {"category": "claims", "topic": "coverage_and_evidence"}
    },
    {
        "id": "policy_cancellation_terms",
        "content": (
            "HARBORSTONE INSURANCE: POLICY CANCELLATION & REFUND POLICY\n"
            "1. Policyholders may request mid-term cancellation with 30 days written notice.\n"
            "2. Pro-rata premium refund is calculated on unexpired policy days, subject to a standard $150 administrative fee.\n"
            "3. Retention incentives: Policyholders considering cancellation may be offered up to 10% loyalty renewal discount or increased coverage limits.\n"
            "4. Policies with open or pending claims cannot be cancelled until claim resolution."
        ),
        "metadata": {"category": "cancellation", "topic": "refund_and_retention"}
    },
    {
        "id": "policy_underwriting_addition",
        "content": (
            "HARBORSTONE INSURANCE: VESSEL & VEHICLE ADDITION UNDERWRITING RULES\n"
            "1. Supported vessel types: Boat, Yacht, Cargo, Tanker, Passenger, Fishing.\n"
            "2. Maximum vessel age: 20 years from year built.\n"
            "3. Luxury vessels valued at or above $500,000 USD require a mandatory independent marine surveyor appraisal report.\n"
            "4. Required documentation: proof of ownership/purchase invoice, current registration, and valuation certificate."
        ),
        "metadata": {"category": "underwriting", "topic": "vehicle_addition"}
    },
]


class PolicyRAGRetriever:
    """
    RAG Retriever for Harborstone insurance policy terms, guidelines, and rules.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        corpus: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.corpus = corpus or HARBORSTONE_POLICY_DOCUMENTS
        self.vector_store = vector_store or VectorStoreManager(collection_name="harborstone_policies")
        self._init_corpus()

    def _init_corpus(self) -> None:
        try:
            existing = self.vector_store.similarity_search("insurance", top_k=1)
            if not existing:
                ids = [d["id"] for d in self.corpus]
                docs = [d["content"] for d in self.corpus]
                metas = [d.get("metadata", {}) for d in self.corpus]
                self.vector_store.add_chunks(ids=ids, documents=docs, metadatas=metas)
        except Exception:
            pass

    def retrieve(self, query: str, top_k: int = 2) -> str:
        """
        Retrieve relevant policy sections as context text.
        Falls back to keyword matching over the corpus if vector search is unavailable.
        """
        try:
            results = self.vector_store.similarity_search(query, top_k=top_k)
            if results:
                return "\n\n".join(r["content"] for r in results)
        except Exception:
            pass

        # Robust keyword fallback
        query_terms = set(query.lower().split())
        scored_docs = []
        for doc in self.corpus:
            content_lower = doc["content"].lower()
            overlap = sum(1 for t in query_terms if t in content_lower)
            scored_docs.append((overlap, doc["content"]))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [content for _, content in scored_docs[:top_k]]
        return "\n\n".join(top_docs) if top_docs else self.corpus[0]["content"]

    def retrieve_chunks(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Return structured chunk objects."""
        try:
            results = self.vector_store.similarity_search(query, top_k=top_k)
            if results:
                return results
        except Exception:
            pass
        return self.corpus[:top_k]
