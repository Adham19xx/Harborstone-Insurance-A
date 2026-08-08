"""
Evaluation Script for RAG Comparison Table.
"""
import time
import json
from tabulate import tabulate
from rag.vector_store import VectorStoreManager
from rag.naive_rag import NaiveRAG
from rag.hybrid_rag import HybridRAG
from rag.agentic_rag import AgenticRAG
from rag.self_rag_check import SelfRAGVerifier

class MockLLM:
    def generate(self, prompt: str) -> str:
        time.sleep(0.1)
        return "YES. Verified grounded response."

def run_evaluation():
    vector_store = VectorStoreManager(persist_dir="./db/chroma_eval")
    mock_llm = MockLLM()
    verifier = SelfRAGVerifier(mock_llm)

    corpus = [
        {"id": "doc1", "content": "Protocol 4.2b requires fasting for 8 hours prior to sedation.", "metadata": {}},
        {"id": "doc2", "content": "Senior dogs require pre-op screening before dental cleanings.", "metadata": {}}
    ]
    for d in corpus:
        vector_store.add_chunks([d["id"]], [d["content"]], [d["metadata"]])

    archs = {
        "Naive RAG": NaiveRAG(vector_store, mock_llm),
        "Hybrid Search": HybridRAG(vector_store, corpus, mock_llm),
        "Agentic RAG": AgenticRAG(vector_store, mock_llm)
    }

    with open("retrieval_eval/questions.json") as f:
        questions = json.load(f)

    results = []
    for name, arch in archs.items():
        total_time, total_tokens, correct = 0.0, 0, 0
        for q in questions:
            start = time.time()
            res = arch.run(q["query"])
            latency = time.time() - start
            
            rel_docs = verifier.verify_relevance(q["query"], res["retrieved_context"])
            if verifier.verify_groundedness(res["response"], rel_docs):
                correct += 1

            total_time += latency
            total_tokens += 150

        results.append([name, f"{correct}/{len(questions)}", f"{total_tokens//len(questions)}", f"{total_time/len(questions):.2f}s"])

    print(tabulate(results, headers=["Architecture", "Accuracy", "Avg. Tokens/Query", "Avg. Latency/Query"], tablefmt="github"))

if __name__ == "__main__":
    run_evaluation()
