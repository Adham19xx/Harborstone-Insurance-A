"""
tree_of_thoughts.py — Tree of Thoughts for Harborstone Insurance
=====================================================================
Adapted from AmrSheta22/task_decomposition_and_planning.
Keeps the original generate/evaluate/BFS-search loop.
Used for Harborstone SYNTHESIS nodes where several alternative
response strategies exist (e.g. "How should we phrase the premium
change recommendation to the customer?").

Router rule (see router.py):
  ToT → synthesis nodes that need lookahead / comparing alternatives
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Pydantic models — compatible with toolkit's Thought model
# ---------------------------------------------------------------------------

class ThoughtCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates: list[str] = Field(min_length=1, max_length=3)


class ThoughtEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


@dataclass
class Thought:
    """A node in the ToT search tree."""
    state: str
    score: float = 0.5
    rationale: str = ""
    depth: int = 0
    parent_state: str = ""
    children: list["Thought"] = field(default_factory=list)


@dataclass
class ToTResult:
    best_thought: Thought
    all_thoughts: list[Thought]
    llm_calls: int
    tokens_used: int
    latency_s: float
    success: bool = True


# ---------------------------------------------------------------------------
# Core BFS Tree-of-Thoughts (preserves toolkit interface)
# ---------------------------------------------------------------------------

_GENERATE_SYSTEM = """\
You are a Harborstone Insurance response strategist.
Given the current state of solving a marine insurance sub-task, generate
2-3 distinct candidate next steps or response approaches.
Each candidate should be meaningfully different (not paraphrases).
Harborstone context:
- Always ground recommendations in policy data retrieved from MCP tools
- Premium changes must be justified numerically
- Document requirements come from get_policy_update_requirements
"""

_EVALUATE_SYSTEM = """\
You are a Harborstone Insurance quality evaluator.
Score the candidate response step from 0.0 (terrible) to 1.0 (perfect).
Evaluation rubric:
  0.8-1.0 : Cites specific numbers, mentions eligibility, lists required docs
  0.5-0.8 : Partially complete, missing some specifics
  0.0-0.5 : Vague, missing key information, or factually wrong
Be strict. A score > 0.8 is reserved for responses that cite real data.
"""


def tree_of_thoughts(
    problem: str,
    llm: BaseChatModel,
    *,
    depth: int = 2,
    beam_width: int = 2,
    task_id: str = "unknown",
    context: dict[str, Any] | None = None,
) -> ToTResult:
    """
    BFS Tree-of-Thoughts search for the best Harborstone synthesis response.

    Parameters
    ----------
    problem : str
        The synthesis task (e.g. "Summarise the policy update recommendation").
    llm : BaseChatModel
        LangChain-compatible chat model.
    depth : int
        BFS depth (how many rounds of generate/evaluate).
    beam_width : int
        How many best thoughts to keep per level.
    task_id : str
        DAG node id for tracing.
    context : dict
        Upstream MCP tool results to include in the problem description.

    Returns
    -------
    ToTResult
    """
    t0 = time.perf_counter()
    ctx_text = ""
    if context:
        import json
        ctx_text = "\n\nData from MCP tools:\n" + json.dumps(context, indent=2, default=str)

    full_problem = f"{problem}{ctx_text}"

    # Seed the frontier with the root node
    frontier: list[Thought] = [
        Thought(state=f"Initial approach to: {problem[:80]}", score=0.5, rationale="root", depth=0)
    ]

    all_thoughts: list[Thought] = list(frontier)
    llm_calls = 0
    tokens = 0

    def _add_tokens(resp: Any) -> None:
        nonlocal tokens
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            tokens += resp.usage_metadata.get("total_tokens", 0)
        elif hasattr(resp, "response_metadata"):
            meta = resp.response_metadata or {}
            usage = meta.get("usage", {})
            tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

    for d in range(depth):
        candidates: list[Thought] = []
        for parent in frontier:
            # --- Generate ---
            gen_resp = llm.with_structured_output(
                ThoughtCandidates,
                method="json_schema",
            ).invoke([
                ("system", _GENERATE_SYSTEM),
                ("human", (
                    f"Problem:\n{full_problem}\n\n"
                    f"Current approach:\n{parent.state}\n\n"
                    "Generate 2-3 candidate next steps."
                )),
            ])
            llm_calls += 1
            _add_tokens(gen_resp)

            for cand_text in gen_resp.candidates:
                # --- Evaluate ---
                eval_resp = llm.with_structured_output(
                    ThoughtEvaluation,
                    method="json_schema",
                ).invoke([
                    ("system", _EVALUATE_SYSTEM),
                    ("human", (
                        f"Problem:\n{full_problem}\n\n"
                        f"Candidate step:\n{cand_text}\n\n"
                        "Score this candidate (0.0-1.0) with rationale."
                    )),
                ])
                llm_calls += 1
                _add_tokens(eval_resp)

                thought = Thought(
                    state=cand_text,
                    score=eval_resp.score,
                    rationale=eval_resp.rationale,
                    depth=d + 1,
                    parent_state=parent.state,
                )
                parent.children.append(thought)
                candidates.append(thought)
                all_thoughts.append(thought)

        # BFS: keep top beam_width by score
        frontier = sorted(candidates, key=lambda t: t.score, reverse=True)[:beam_width]

    best = max(all_thoughts, key=lambda t: t.score)
    latency = time.perf_counter() - t0

    return ToTResult(
        best_thought=best,
        all_thoughts=all_thoughts,
        llm_calls=llm_calls,
        tokens_used=tokens,
        latency_s=round(latency, 3),
        success=best.score >= 0.5,
    )


# Alias
tree_of_thoughts_search = tree_of_thoughts

