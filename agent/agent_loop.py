"""
Agent Loop Integration combining Memory, Context, RAG, and Self-RAG.
"""
from rag.vector_store import VectorStoreManager
from rag.hybrid_rag import HybridRAG
from rag.self_rag_check import SelfRAGVerifier

class AgentLoop:
    def __init__(self, llm_client, corpus):
        self.llm = llm_client
        self.vector_store = VectorStoreManager()
        self.retriever = HybridRAG(self.vector_store, corpus, self.llm)
        self.verifier = SelfRAGVerifier(self.llm)

    def process_turn(self, user_query: str) -> str:
        # Retrieve candidate context
        retrieved = self.retriever.run(user_query)
        
        # Verify Context Relevance
        relevant_chunks = self.verifier.verify_relevance(user_query, retrieved["retrieved_context"])
        
        # Groundedness Check
        if not self.verifier.verify_groundedness(retrieved["response"], relevant_chunks):
            return "Unable to provide a verified grounded response based on context."
            
        return retrieved["response"]
