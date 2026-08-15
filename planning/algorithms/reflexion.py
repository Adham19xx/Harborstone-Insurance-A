"""
reflexion.py — Reflexion with Episodic Memory for Harborstone Insurance
=====================================================================
Adapted from AmrSheta22/task_decomposition_and_planning (planning_lab/algorithms/reflexion.py).

Key differences from the toolkit:
  1. The Environment used here is the REAL grounded HarborstoneMCPEnvironment,
     not the randomized fake one.
  2. The episodic memory buffer is explicitly typed and persisted across trials
     within the same run (capped at `memory_size` most-recent reflections).
  3. The evaluate step checks the output against real Harborstone business rules
     (vessel schema, premium positivity, document list completeness).

Used for sub-tasks where a single retry isn't enough and the agent needs
to LEARN across attempts within the same run:
  "Generate the full policy update proposal for customer X including
   eligibility, premium impact, and required documents."

This sub-task frequently fails on the first attempt because:
  - The LLM hallucinates vessel types not in the Harborstone schema
  - Premium estimates are missing numeric values
  - Required documents are omitted from the draft
Only Reflexion's cross-trial memory reliably fixes all three across runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel

from .environment import Environment, EnvironmentFeedback


# ---------------------------------------------------------------------------
# Data classes — compatible with toolkit's ReflexionTrial / ReflexionResult
# ---------------------------------------------------------------------------

@dataclass
class ReflexionTrial:
    """A single Reflexion trial."""
    number: int
    attempt: str
    feedback: EnvironmentFeedback
    reflection: str | None = None       # verbal reflection (if trial failed)


@dataclass
class EpisodicMemoryBuffer:
    """
    Capped episodic memory for Reflexion.

    Stores verbal reflections from failed trials so that each new trial
    can read from past mistakes.  max_size enforces the cap.
    """
    max_size: int = 3
    _entries: list[str] = field(default_factory=list)

    def add(self, reflection: str) -> None:
        self._entries.append(reflection)
        if len(self._entries) > self.max_size:
            self._entries = self._entries[-self.max_size:]

    def recall(self) -> list[str]:
        """Return the most-recent reflections (up to max_size)."""
        return list(self._entries)

    def as_context_block(self) -> str:
        """Format for inclusion in a prompt."""
        if not self._entries:
            return "- No prior trials."
        return "\n".join(f"- {entry}" for entry in self._entries)


@dataclass
class ReflexionResult:
    success: bool
    output: str
    trials: list[ReflexionTrial]
    memory: list[str]                   # contents of the episodic buffer at end
    llm_calls: int = 0
    tokens_used: int = 0
    latency_s: float = 0.0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ATTEMPT_SYSTEM = """\
You are a Harborstone Insurance policy advisor.
Your task is to produce a complete, accurate response to the given insurance request.
If you have prior trial reflections (episodic memory), learn from them.

Harborstone business rules you must follow:
  - vessel_type must be: cargo, tanker, passenger, fishing, or yacht
  - All premium values must be positive floats in USD
  - A policy update always requires: eligibility confirmation, premium impact estimate,
    and a list of required documents
  - Do not invent data — describe what the MCP tools would return
"""

_EVALUATE_SYSTEM = """\
You are a Harborstone Insurance critic evaluating a trial attempt.
The attempt should:
  1. Confirm vessel eligibility with a clear YES/NO
  2. State the estimated premium change (with a numeric value in USD)
  3. List required documents for the policy update
  4. Respect Harborstone vessel_type constraints
Respond with:
VERDICT: PASS or FAIL
ISSUES: <bullet list of specific problems if FAIL, or "None" if PASS>
"""

_REFLECT_SYSTEM = """\
You are a Harborstone Insurance self-improvement agent.
The previous attempt failed. Write a verbal reflection (2-3 sentences) that:
  1. Names the specific Harborstone rule or rubric point that was violated
  2. Explains concretely what to do differently in the next attempt
  3. References the actual failure (e.g., "missing USD premium value", "invalid vessel type")
Be specific — vague reflections like "do better" earn no credit.
"""


# ---------------------------------------------------------------------------
# Core Reflexion function — interface matches toolkit's reflexion()
# ---------------------------------------------------------------------------

def reflexion(
    task: str,
    llm: BaseChatModel,
    environment: Environment,
    *,
    max_trials: int = 3,
    memory_size: int = 3,
    task_id: str = "unknown",
    context: str = "",
) -> ReflexionResult:
    """
    Reflexion with capped episodic memory for Harborstone policy update tasks.

    Each trial:
      1. Recalls the episodic buffer (verbal reflections from prior failures)
      2. Produces an attempt informed by those reflections
      3. Evaluates the attempt with the GROUNDED environment
      4. If failed, generates a verbal reflection and adds it to the episodic buffer
      5. Repeats until success or max_trials exhausted

    Parameters
    ----------
    task : str
        The sub-task instruction.
    llm : BaseChatModel
        LangChain-compatible chat model.
    environment : Environment
        The REAL grounded Harborstone environment (not the toolkit's random default).
    max_trials : int
        Maximum number of retry attempts (default 3).
    memory_size : int
        Episodic buffer capacity (default 3 = capped).
    task_id : str
        DAG node id for tracing.
    context : str
        Upstream MCP tool results as formatted text.

    Returns
    -------
    ReflexionResult
    """
    if max_trials < 1 or memory_size < 1:
        raise ValueError("max_trials and memory_size must be positive integers")

    t0 = time.perf_counter()
    memory = EpisodicMemoryBuffer(max_size=memory_size)
    trials: list[ReflexionTrial] = []
    best_attempt = ""
    best_score = -1.0
    llm_calls = 0
    tokens = 0

    def _add_tokens(resp) -> None:
        nonlocal tokens
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            tokens += resp.usage_metadata.get("total_tokens", 0)
        elif hasattr(resp, "response_metadata"):
            meta = resp.response_metadata or {}
            usage = meta.get("usage", {})
            tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

    context_block = f"\n\nContext (upstream MCP results):\n{context}" if context else ""

    for trial_num in range(1, max_trials + 1):
        # --- Step 1: Recall episodic memory ---
        recalled = memory.as_context_block()

        # --- Step 2: Attempt ---
        attempt_resp = llm.invoke([
            ("system", _ATTEMPT_SYSTEM),
            ("human", (
                f"Task:\n{task}{context_block}\n\n"
                f"Episodic memory from prior trials:\n{recalled}\n\n"
                "Produce your best response now."
            )),
        ])
        llm_calls += 1
        _add_tokens(attempt_resp)
        attempt_text = attempt_resp.content if hasattr(attempt_resp, "content") else str(attempt_resp)

        # Track best regardless of outcome
        if attempt_text and len(attempt_text) > len(best_attempt):
            best_attempt = attempt_text

        # --- Step 3: Grounded evaluation ---
        feedback = environment.evaluate(task=task, output=attempt_text)

        # --- Step 4 (optional): LLM also evaluates for nuanced critique ---
        eval_resp = llm.invoke([
            ("system", _EVALUATE_SYSTEM),
            ("human", (
                f"Task:\n{task}\n\n"
                f"Attempt:\n{attempt_text}\n\n"
                f"Grounded environment feedback:\n"
                f"Score: {feedback.score:.2f} | {feedback.message}"
            )),
        ])
        llm_calls += 1
        _add_tokens(eval_resp)

        if feedback.score > best_score:
            best_score = feedback.score
            best_attempt = attempt_text

        trial = ReflexionTrial(
            number=trial_num,
            attempt=attempt_text,
            feedback=feedback,
        )

        # --- Step 5: Reflect on failure and update episodic memory ---
        if not feedback.success:
            reflect_resp = llm.invoke([
                ("system", _REFLECT_SYSTEM),
                ("human", (
                    f"Task:\n{task}\n\n"
                    f"Failed attempt (Trial {trial_num}):\n{attempt_text}\n\n"
                    f"Environment issues:\n"
                    + "\n".join(f"- {i}" for i in feedback.caught_issues)
                    + f"\n\nEnvironment score: {feedback.score:.2f}"
                )),
            ])
            llm_calls += 1
            _add_tokens(reflect_resp)
            reflection_text = reflect_resp.content if hasattr(reflect_resp, "content") else str(reflect_resp)
            trial.reflection = reflection_text
            # Add to episodic buffer (capped)
            memory.add(reflection_text)
        else:
            # Success — no reflection needed
            trials.append(trial)
            break

        trials.append(trial)

    latency = time.perf_counter() - t0
    return ReflexionResult(
        success=best_score >= environment.success_threshold,
        output=best_attempt,
        trials=trials,
        memory=memory.recall(),
        llm_calls=llm_calls,
        tokens_used=tokens,
        latency_s=round(latency, 3),
    )


# Alias
run_reflexion = reflexion

