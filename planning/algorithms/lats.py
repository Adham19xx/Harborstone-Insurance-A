"""
lats.py — Language Agent Tree Search (LATS) for Harborstone Insurance
=====================================================================
Adapted from AmrSheta22/task_decomposition_and_planning (planning_lab/algorithms/lats.py).
Preserves the MCTS four-phase loop (select → expand+simulate → evaluate/reflect → backpropagate)
but replaces the toolkit's randomized Environment with the real HarborstoneMCPEnvironment.

Router rule (see router.py):
  LATS → high-stakes nodes: check_vessel_eligibility + estimate_policy_premium_change
         together, where a wrong plan costs real money / customer trust.

Key changes from the toolkit:
  - environment.Environment is the REAL grounded evaluator (environment.py in this folder)
  - Reflections carry Harborstone-specific failure reasons (schema issues, missing keys)
  - The action generator is primed with Harborstone tool signatures
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from .environment import Environment, EnvironmentFeedback


# ---------------------------------------------------------------------------
# Pydantic helpers — compatible with toolkit's LATSAction / LATSActionBatch
# ---------------------------------------------------------------------------

class LATSAction(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    action: str = Field(min_length=2)
    state: str = Field(min_length=2)


class LATSActionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actions: list[LATSAction] = Field(min_length=1, max_length=3)


class ValueEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class LATSNode:
    state: str
    action: str = "root"
    parent: "LATSNode | None" = field(default=None, repr=False)
    children: list["LATSNode"] = field(default_factory=list, repr=False)
    visits: int = 0
    value_sum: float = 0.0
    environment_score: float = 0.0
    model_score: float = 0.0
    feedback: EnvironmentFeedback | None = None
    reflections: list[str] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class LATSResult:
    success: bool
    output: str
    best_score: float
    iterations: int
    root: LATSNode
    llm_calls: int = 0
    tokens_used: int = 0
    latency_s: float = 0.0


# ---------------------------------------------------------------------------
# MCTS helpers (unchanged from toolkit)
# ---------------------------------------------------------------------------

def _uct(node: LATSNode, exploration_weight: float) -> float:
    if node.visits == 0:
        return float("inf")
    parent_visits = max(node.parent.visits if node.parent else 1, 1)
    return node.mean_value + exploration_weight * math.sqrt(math.log(parent_visits) / node.visits)


def _select_leaf(root: LATSNode, exploration_weight: float) -> LATSNode:
    node = root
    while node.children:
        node = max(node.children, key=lambda child: _uct(child, exploration_weight))
    return node


def _backpropagate(node: LATSNode, value: float) -> None:
    while node is not None:
        node.visits += 1
        node.value_sum += value
        node = node.parent


def _trajectory_reflections(node: LATSNode) -> list[str]:
    """Collect reflections from all ancestors."""
    reflections: list[str] = []
    current = node
    while current is not None:
        reflections.extend(current.reflections)
        current = current.parent
    return reflections


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ACTION_SYSTEM = """\
You are a Harborstone Insurance planning agent.
Generate 2-3 alternative next actions to take toward solving the given insurance sub-task.
Each action should be a concrete step (e.g., "Call check_vessel_eligibility with …",
"Estimate premium using current_premium=X and vessel_value=Y", etc.).
Ground every action in the available Harborstone MCP tools:
  - get_customer_policies(customer_id)
  - check_vessel_eligibility(vessel_type, year_built, value)
  - estimate_policy_premium_change(current_premium, vessel_type, vessel_value)
  - get_policy_update_requirements(vessel_type, vessel_value)
"""

_SIMULATE_SYSTEM = """\
You are simulating the outcome of a Harborstone Insurance action.
Given the current state and the proposed action, describe the resulting state
as if the MCP tool was called and returned a realistic result.
Be specific: include plausible numeric values, eligibility outcomes, and document lists.
"""

_VALUE_SYSTEM = """\
You are evaluating the quality of a Harborstone Insurance agent state.
Score the state 0.0-1.0:
  1.0 → all needed data retrieved, eligibility confirmed, premium estimated, docs listed
  0.7 → most data present but one piece missing
  0.4 → significant gaps
  0.0 → wrong tool used or contradicts business rules
"""

_REFLECT_SYSTEM = """\
You are a Harborstone Insurance critic.
The previous action failed or scored poorly.
Write a concise verbal reflection (1-2 sentences) explaining:
1. Why it failed (what business rule or schema check it violated)
2. What the next action should do differently
Be specific to Harborstone: mention vessel_type validity, premium positivity, or missing doc list.
"""


# ---------------------------------------------------------------------------
# Core LATS function — grounded with real Harborstone environment
# ---------------------------------------------------------------------------

def lats(
    task: str,
    llm: BaseChatModel,
    environment: Environment,
    *,
    max_iterations: int = 5,
    exploration_weight: float = 1.4,
    task_id: str = "unknown",
    context: dict[str, Any] | None = None,
) -> LATSResult:
    """
    MCTS-guided LATS for a Harborstone high-stakes sub-task.

    The four-phase loop:
      1. SELECT    — UCT-based tree traversal to a leaf
      2. EXPAND    — Generate 2-3 alternative actions from the leaf
      3. SIMULATE  — Simulate outcome of each action
      4. EVALUATE  — Score with REAL grounded environment (not random)
      5. REFLECT   — Generate verbal reflection on failures
      6. BACKPROP  — Update visit/value counts up the tree

    Parameters
    ----------
    task : str
        The sub-task (e.g. "Determine eligibility and estimate premium for vessel X").
    llm : BaseChatModel
        LangChain-compatible LLM.
    environment : Environment
        The grounded Harborstone environment (from environment.py).
    max_iterations : int
        MCTS budget.
    exploration_weight : float
        UCT exploration constant.
    task_id : str
        DAG node id for tracing.
    context : dict
        Upstream MCP results.

    Returns
    -------
    LATSResult
    """
    t0 = time.perf_counter()
    ctx_text = ""
    if context:
        import json
        ctx_text = "\n\nContext from upstream tasks:\n" + json.dumps(context, indent=2, default=str)

    full_task = f"{task}{ctx_text}"
    root = LATSNode(state=f"Initial state: {task[:80]}")
    llm_calls = 0
    tokens = 0
    best_node = root
    best_score = -1.0

    def _add_tokens(resp: Any) -> None:
        nonlocal tokens
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            tokens += resp.usage_metadata.get("total_tokens", 0)
        elif hasattr(resp, "response_metadata"):
            meta = resp.response_metadata or {}
            usage = meta.get("usage", {})
            tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

    for iteration in range(max_iterations):
        # --- PHASE 1: SELECT ---
        leaf = _select_leaf(root, exploration_weight)

        # --- PHASE 2: EXPAND — generate 2-3 actions ---
        trajectory = _trajectory_reflections(leaf)
        reflection_block = ""
        if trajectory:
            reflection_block = "\n\nPrior failure reflections (learn from these):\n" + \
                               "\n".join(f"- {r}" for r in trajectory[-3:])

        action_resp = llm.with_structured_output(
            LATSActionBatch,
            method="json_schema",
        ).invoke([
            ("system", _ACTION_SYSTEM),
            ("human", (
                f"Task:\n{full_task}\n\n"
                f"Current state:\n{leaf.state}"
                f"{reflection_block}\n\n"
                "Generate 2-3 alternative next actions."
            )),
        ])
        llm_calls += 1
        _add_tokens(action_resp)

        for action_item in action_resp.actions:
            # --- PHASE 3: SIMULATE ---
            sim_resp = llm.invoke([
                ("system", _SIMULATE_SYSTEM),
                ("human", (
                    f"Task:\n{full_task}\n\n"
                    f"Current state:\n{leaf.state}\n\n"
                    f"Action:\n{action_item.action}\n\n"
                    "Describe the resulting state after this action."
                )),
            ])
            llm_calls += 1
            _add_tokens(sim_resp)

            sim_state = sim_resp.content if hasattr(sim_resp, "content") else str(sim_resp)

            # --- PHASE 4A: EVALUATE with grounded environment ---
            feedback = environment.evaluate(task=full_task, output=sim_state)

            # --- PHASE 4B: Model value estimate ---
            val_resp = llm.with_structured_output(
                ValueEstimate,
                method="json_schema",
            ).invoke([
                ("system", _VALUE_SYSTEM),
                ("human", f"Task:\n{full_task}\n\nCurrent state:\n{sim_state}\n\nScore this state."),
            ])
            llm_calls += 1
            _add_tokens(val_resp)

            combined_score = 0.6 * feedback.score + 0.4 * val_resp.score
            child = LATSNode(
                state=sim_state,
                action=action_item.action,
                parent=leaf,
                environment_score=feedback.score,
                model_score=val_resp.score,
                feedback=feedback,
            )
            leaf.children.append(child)

            # --- PHASE 4C: REFLECT on failure ---
            if not feedback.success:
                reflect_resp = llm.invoke([
                    ("system", _REFLECT_SYSTEM),
                    ("human", (
                        f"Task:\n{full_task}\n\n"
                        f"Failed state:\n{sim_state}\n\n"
                        f"Environment feedback:\n{feedback.message}\n"
                        f"Caught issues:\n" + "\n".join(f"- {i}" for i in feedback.caught_issues)
                    )),
                ])
                llm_calls += 1
                _add_tokens(reflect_resp)
                reflection = reflect_resp.content if hasattr(reflect_resp, "content") else str(reflect_resp)
                child.reflections.append(reflection)

            # --- PHASE 5: BACKPROPAGATE ---
            _backpropagate(child, combined_score)

            if combined_score > best_score:
                best_score = combined_score
                best_node = child

    latency = time.perf_counter() - t0
    return LATSResult(
        success=best_score >= environment.success_threshold,
        output=best_node.state,
        best_score=round(best_score, 3),
        iterations=max_iterations,
        root=root,
        llm_calls=llm_calls,
        tokens_used=tokens,
        latency_s=round(latency, 3),
    )


# Alias
lats_search = lats

