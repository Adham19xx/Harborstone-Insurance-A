"""
plan_and_solve.py — Plan-and-Solve for Harborstone Insurance
=====================================================================
Adapted from the reference toolkit (AmrSheta22/task_decomposition_and_planning).
Keeps the original two-phase interface (PLAN then SOLVE) but wraps it
around a real Harborstone sub-task whose "solution" is a concrete MCP
tool call plus a natural-language explanation.

Person 2 owns this file.  Person 1's decomposition.py calls plan_and_solve()
for any DAG node whose `kind == "mcp"` and whose tool is deterministic
(no ambiguity in arguments).

Route selection (see router.py):
  PS  → deterministic MCP lookups (get_customer_policies, get_vessel …)
  ToT → synthesis nodes that require comparing alternatives
  LATS→ high-stakes nodes (eligibility + premium estimation together)
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlanAndSolveResult:
    """Structured result from a Plan-and-Solve run."""
    task_id: str
    tool_name: str | None
    plan: str                      # the PLAN phase text
    solution: str                  # the SOLVE phase text (final answer)
    llm_calls: int = 2             # always 2: one plan, one solve
    tokens_used: int = 0
    latency_s: float = 0.0
    success: bool = True


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
You are a Harborstone Insurance planning assistant.
You will receive a single sub-task that is part of a larger insurance policy-update request.
Your job is to produce a clear, numbered PLAN for how to solve that sub-task.
Do NOT execute the plan yet — only describe the steps.
Be concise: 2-4 steps maximum.
"""

_SOLVE_SYSTEM = """\
You are a Harborstone Insurance execution assistant.
You will receive a sub-task and its PLAN.
Execute the plan step-by-step and produce the final answer.
If the sub-task requires calling an MCP tool, specify:
  TOOL: <tool_name>
  ARGUMENTS: <json dict>
  RESULT INTERPRETATION: <what the result means for the customer>
Always end with a CONCLUSION section.
"""


# ---------------------------------------------------------------------------
# Core function — kept interface-compatible with the toolkit's plan_and_solve
# ---------------------------------------------------------------------------

def plan_and_solve(
    question: str,
    llm: BaseChatModel,
    *,
    task_id: str = "unknown",
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> PlanAndSolveResult:
    """
    Two-phase Plan-and-Solve for a single Harborstone sub-task.

    Phase 1 — PLAN : ask the LLM to devise a numbered plan.
    Phase 2 — SOLVE: ask the LLM to execute the plan step by step.

    This matches the original toolkit's interface (plan_and_solve(question, llm))
    and adds Harborstone-specific context (tool signatures, customer data).

    Parameters
    ----------
    question : str
        The sub-task instruction (e.g. "Check vessel eligibility for …").
    llm : BaseChatModel
        Any LangChain-compatible chat model (Anthropic Claude, OpenAI, Gemini …).
    task_id : str
        The DAG node id, used for tracing.
    tool_name : str | None
        The Harborstone MCP tool this node should call (if kind == "mcp").
    arguments : dict | None
        Pre-filled arguments from the decomposition DAG.
    context : dict | None
        Outputs from upstream DAG nodes (dependency results).

    Returns
    -------
    PlanAndSolveResult
    """
    t0 = time.perf_counter()
    context = context or {}
    arguments = arguments or {}

    # Build a rich human prompt that includes tool signature + upstream context
    context_block = ""
    if context:
        context_block = "\n\nContext from upstream tasks:\n" + json.dumps(context, indent=2, default=str)

    tool_block = ""
    if tool_name:
        tool_block = f"\n\nYou must call MCP tool: {tool_name}\nArguments available: {json.dumps(arguments, indent=2)}"

    human_prompt = f"""{question}{tool_block}{context_block}

First understand the problem and devise a plan to solve it. Then carry out the
plan step by step. Check that each step is consistent with the available tool
signatures and the Harborstone Insurance business rules:
- Vessel value must be > 0
- vessel_type must be one of: cargo, tanker, passenger, fishing, yacht
- Premiums are always positive floats in USD
- Policy updates require eligibility check BEFORE premium estimation
"""

    # -----------------------------------------------------------------------
    # Phase 1: PLAN
    # -----------------------------------------------------------------------
    plan_response = llm.invoke([
        ("system", _PLAN_SYSTEM),
        ("human", human_prompt),
    ])
    plan_text: str = plan_response.content if hasattr(plan_response, "content") else str(plan_response)

    # -----------------------------------------------------------------------
    # Phase 2: SOLVE
    # -----------------------------------------------------------------------
    solve_response = llm.invoke([
        ("system", _SOLVE_SYSTEM),
        ("human", f"Sub-task: {question}\n\nPLAN:\n{plan_text}\n\nNow execute the plan step by step."),
    ])
    solution_text: str = solve_response.content if hasattr(solve_response, "content") else str(solve_response)

    latency = time.perf_counter() - t0

    # Rough token accounting (works for Anthropic & OpenAI response objects)
    tokens = 0
    for resp in (plan_response, solve_response):
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            tokens += resp.usage_metadata.get("total_tokens", 0)
        elif hasattr(resp, "response_metadata"):
            meta = resp.response_metadata or {}
            usage = meta.get("usage", {})
            tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

    return PlanAndSolveResult(
        task_id=task_id,
        tool_name=tool_name,
        plan=plan_text,
        solution=solution_text,
        llm_calls=2,
        tokens_used=tokens,
        latency_s=round(latency, 3),
        success=True,
    )
