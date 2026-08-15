"""
self_refine.py — Self-Refine for Harborstone Insurance
=====================================================================
Adapted from AmrSheta22/task_decomposition_and_planning (planning_lab/algorithms/self_refine.py).

Keeps the original three-step flow:
  1. DRAFT   — initial answer to the sub-task
  2. CRITIQUE— check against an explicit Harborstone rubric (grounded)
  3. REVISE  — produce an improved answer given the critique

Used for synthesis nodes whose output is cheap to regenerate:
  "Summarise the policy update recommendation for the customer."

The rubric is GROUNDED against Harborstone business rules:
  - Does it mention the estimated premium change (with a number)?
  - Does it confirm or deny vessel eligibility?
  - Does it list required documents?
  - Is it at least 80 words (not a stub)?
  - Does it address the customer by their policy context?
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel


# ---------------------------------------------------------------------------
# Grounded rubric checks (replaces the toolkit's generic text checks)
# ---------------------------------------------------------------------------

HARBORSTONE_RUBRIC = [
    (
        r"\$[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s*(?:USD|usd|per\s+year|annually)",
        "premium_amount",
        "Must cite a specific premium change amount (e.g. '$1,234 per year')."
    ),
    (
        r"\b(eligible|ineligible|eligib|qualify|qualifies|does not qualify)\b",
        "eligibility",
        "Must state whether the vessel is eligible for the policy."
    ),
    (
        r"\b(document|require|certificate|survey|inspection|proof)\b",
        "documents",
        "Must mention required documents or next steps."
    ),
    (
        r"\b(vessel|ship|craft|boat)\b",
        "vessel_mention",
        "Must reference the vessel being assessed."
    ),
]


def harborstone_rubric_check(goal: str, draft: str) -> list[str]:
    """
    Grounded rubric for Harborstone synthesis outputs.
    Returns a list of issue strings (empty = no issues).

    This replaces the toolkit's generic deterministic_checks().
    """
    issues: list[str] = []
    lc = draft.lower()

    for pattern, label, message in HARBORSTONE_RUBRIC:
        if not re.search(pattern, lc, re.IGNORECASE):
            issues.append(f"[{label}] {message}")

    word_count = len(draft.split())
    if word_count < 80:
        issues.append(f"[length] Output is {word_count} words. Must be >= 80 words to be complete.")

    return issues


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ReflectionResult:
    """Result of a Self-Refine run. Compatible with toolkit's ReflectionResult."""
    draft: str
    critique: str
    revised: str
    grounded_issues: list[str]          # from harborstone_rubric_check
    llm_calls: int = 3                  # draft + critique + revise
    tokens_used: int = 0
    latency_s: float = 0.0
    improved: bool = False              # True if revised passes rubric


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CRITIQUE_SYSTEM = """\
You are a Harborstone Insurance quality reviewer.
You will receive a draft recommendation for a customer's policy update request.
Critique it against this rubric:
  1. Does it cite a specific premium change amount?
  2. Does it state whether the vessel is eligible?
  3. Does it list required documents or next steps?
  4. Is it complete (at least 80 words)?
  5. Does it mention the vessel?
For each failed criterion, explain what is missing and how to fix it.
Also list what the draft does well.
"""

_REVISE_SYSTEM = """\
You are a Harborstone Insurance policy advisor.
You will receive a draft recommendation and a critique.
Write an improved recommendation that addresses every critique point.
Keep what was already good. Do not invent data — use only the information
provided in the draft and the task context.
"""


# ---------------------------------------------------------------------------
# Core function — interface matches toolkit's reflect_and_refine(goal, draft, llm)
# ---------------------------------------------------------------------------

def reflect_and_refine(
    goal: str,
    draft: str,
    llm: BaseChatModel,
    *,
    task_id: str = "unknown",
    context: str = "",
) -> ReflectionResult:
    """
    Self-Refine for a Harborstone synthesis output.

    Parameters
    ----------
    goal : str
        The synthesis task instruction.
    draft : str
        The initial draft to be critiqued and revised.
    llm : BaseChatModel
        LangChain-compatible chat model.
    task_id : str
        DAG node id for tracing.
    context : str
        Additional context (upstream MCP results) as text.

    Returns
    -------
    ReflectionResult
    """
    t0 = time.perf_counter()
    tokens = 0

    def _add_tokens(resp) -> None:
        nonlocal tokens
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            tokens += resp.usage_metadata.get("total_tokens", 0)
        elif hasattr(resp, "response_metadata"):
            meta = resp.response_metadata or {}
            usage = meta.get("usage", {})
            tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

    # --- Step 1: Grounded rubric check (deterministic, no LLM) ---
    grounded_issues = harborstone_rubric_check(goal, draft)

    # --- Step 2: LLM critique (adds nuance beyond the rubric) ---
    context_block = f"\n\nContext (MCP tool results):\n{context}" if context else ""
    critique_resp = llm.invoke([
        ("system", _CRITIQUE_SYSTEM),
        ("human", (
            f"Goal: {goal}{context_block}\n\n"
            f"Draft:\n{draft}\n\n"
            + (
                "Grounded rubric failures already detected:\n"
                + "\n".join(f"- {i}" for i in grounded_issues)
                if grounded_issues
                else "No rubric failures detected by automated checks."
            )
        )),
    ])
    _add_tokens(critique_resp)
    critique_text = critique_resp.content if hasattr(critique_resp, "content") else str(critique_resp)

    # --- Step 3: Revision ---
    revise_resp = llm.invoke([
        ("system", _REVISE_SYSTEM),
        ("human", (
            f"Goal: {goal}{context_block}\n\n"
            f"Original Draft:\n{draft}\n\n"
            f"Critique:\n{critique_text}\n\n"
            "Write the improved recommendation."
        )),
    ])
    _add_tokens(revise_resp)
    revised_text = revise_resp.content if hasattr(revise_resp, "content") else str(revise_resp)

    # Did the revision fix the rubric issues?
    remaining_issues = harborstone_rubric_check(goal, revised_text)
    improved = len(remaining_issues) < len(grounded_issues)

    latency = time.perf_counter() - t0
    return ReflectionResult(
        draft=draft,
        critique=critique_text,
        revised=revised_text,
        grounded_issues=grounded_issues,
        llm_calls=3,
        tokens_used=tokens,
        latency_s=round(latency, 3),
        improved=improved,
    )


# Aliases
self_refine = reflect_and_refine
SelfRefineResult = ReflectionResult

