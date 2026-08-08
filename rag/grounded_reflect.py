from dataclasses import dataclass
import re
from typing import Callable, List, Optional


@dataclass
class Draft:
    query: str
    answer: str
    chunks: List[str]


@dataclass
class Critique:
    passed: bool
    reason: str
    suggested_query: Optional[str] = None


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "and", "or",
    "with", "from", "this", "that", "these", "those",
    "it", "its", "be", "as", "at", "by",
    "does", "do", "did", "can", "could",
    "would", "should", "what", "which", "who",
    "how", "why", "when", "where"
}


def normalize_text(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    return [
        word
        for word in words
        if word not in STOPWORDS and len(word) > 2
    ]


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def sentence_support_score(
    sentence: str,
    chunks: List[str]
) -> float:

    sentence_words = set(normalize_text(sentence))

    if not sentence_words:
        return 1.0

    context_words = set()

    for chunk in chunks:
        context_words.update(normalize_text(chunk))

    matched_words = sentence_words.intersection(context_words)

    return len(matched_words) / len(sentence_words)


def build_draft_answer(
    query: str,
    search_tool: Callable,
    answer_builder: Callable,
    top_k: int = 3
) -> Draft:

    hits = search_tool(query, top_k)

    chunks = []

    for item in hits:

        if isinstance(item, tuple):
            chunk = item[0]

        elif isinstance(item, dict):
            chunk = item.get("content", "")

        else:
            chunk = str(item)

        if chunk:
            chunks.append(chunk)

    answer = answer_builder(query, chunks)

    return Draft(
        query=query,
        answer=answer,
        chunks=chunks
    )


def critique_answer(
    draft: Draft,
    min_support: float = 0.50
) -> Critique:

    if not draft.chunks:
        return Critique(
            passed=False,
            reason="No retrieved context was available.",
            suggested_query=draft.query
        )

    sentences = split_sentences(draft.answer)

    if not sentences:
        return Critique(
            passed=False,
            reason="The draft answer is empty.",
            suggested_query=draft.query
        )

    unsupported = []

    for sentence in sentences:

        score = sentence_support_score(
            sentence,
            draft.chunks
        )

        if score < min_support:
            unsupported.append(
                f"{sentence} (support={score:.2f})"
            )

    if not unsupported:
        return Critique(
            passed=True,
            reason="All answer sentences are sufficiently supported."
        )

    return Critique(
        passed=False,
        reason="Unsupported claims detected: " + "; ".join(unsupported),
        suggested_query=draft.query
    )


def answer_with_grounding_check(
    query: str,
    search_tool: Callable,
    answer_builder: Callable,
    top_k: int = 3,
    min_support: float = 0.50
) -> str:

    # First attempt
    draft = build_draft_answer(
        query=query,
        search_tool=search_tool,
        answer_builder=answer_builder,
        top_k=top_k
    )

    critique = critique_answer(
        draft,
        min_support=min_support
    )

    if critique.passed:
        return draft.answer

    # Exactly ONE retry
    retry_query = (
        critique.suggested_query
        or query
    )

    retry_draft = build_draft_answer(
        query=retry_query,
        search_tool=search_tool,
        answer_builder=answer_builder,
        top_k=top_k
    )

    retry_critique = critique_answer(
        retry_draft,
        min_support=min_support
    )

    if retry_critique.passed:
        return retry_draft.answer

    return (
        "I couldn't find a sufficiently grounded answer "
        "in the available knowledge base."
    )


def deterministic_answer_builder(
    query: str,
    chunks: List[str]
) -> str:

    if not chunks:
        return "No supporting information was found."

    return " ".join(chunks)


def fake_search_knowledge_base(
    query: str,
    top_k: int = 3
):

    knowledge_base = [
        (
            "Harborstone provides marine hull insurance for eligible vessels.",
            0.95
        ),
        (
            "Claims above $10,000 require explicit approval from a Senior Claims Officer.",
            0.90
        ),
        (
            "Storm damage claims must include official meteorological verification.",
            0.85
        ),
        (
            "Engine failures require proof of regular maintenance within six months.",
            0.80
        ),
    ]

    query_words = set(normalize_text(query))

    scored = []

    for text, base_score in knowledge_base:

        text_words = set(normalize_text(text))

        overlap = len(
            query_words.intersection(text_words)
        )

        score = base_score + (overlap * 0.05)

        scored.append(
            (text, score)
        )

    scored.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return scored[:top_k]


if __name__ == "__main__":

    query = "What is required for storm damage claims?"

    result = answer_with_grounding_check(
        query=query,
        search_tool=fake_search_knowledge_base,
        answer_builder=deterministic_answer_builder,
        top_k=3,
        min_support=0.50
    )

    print("=" * 60)
    print("GROUNDED VERIFICATION TEST")
    print("=" * 60)

    print("Query:")
    print(query)

    print("\nFinal answer:")
    print(result)