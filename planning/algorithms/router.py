"""Sub-Task Planning Router for Harborstone Insurance.

This module provides explicit routing logic to dispatch sub-tasks in a DAG to
the optimal planning algorithm:
- Plan-and-Solve (PS): Linear deterministic calculations, mathematical formulas, and actuarial rating.
- Tree of Thoughts (ToT): Multi-candidate ranking, multi-criteria risk prioritization, and tradeoff search.
- LATS: High-stakes policy proposals requiring MCTS search guided by real Grounded Environment feedback.
- MCP Direct: Single deterministic tool execution against the Harborstone MCP server.
"""

from __future__ import annotations

from typing import Any, Dict, Literal
from pydantic import BaseModel, ConfigDict

PlanningMethod = Literal["PS", "ToT", "LATS", "MCP_DIRECT", "SELF_REFINE"]


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    instruction: str
    selected_method: PlanningMethod
    rationale: str
    estimated_complexity: str


def route_subtask(task_id: str, instruction: str, context: Dict[str, Any]) -> RoutingDecision:
    """
    Decide the optimal planning method for a sub-task based on its structural characteristics.
    """
    inst_lower = instruction.lower()
    kind = context.get("kind", "mcp")
    tool_name = context.get("tool_name")

    # 1. Direct MCP Tool call
    if kind == "mcp" and tool_name:
        return RoutingDecision(
            task_id=task_id,
            instruction=instruction,
            selected_method="MCP_DIRECT",
            rationale=f"Deterministic single-hop lookup via MCP tool '{tool_name}'.",
            estimated_complexity="O(1) - Single tool call",
        )

    # 2. Plan-and-Solve (PS): Mathematical, actuarial, or formulaic computations
    if any(k in inst_lower for k in ["calculate", "estimate", "premium", "math", "rate", "formula", "tax", "fee", "arithmetic"]):
        return RoutingDecision(
            task_id=task_id,
            instruction=instruction,
            selected_method="PS",
            rationale="Deterministic linear calculation task requiring explicit step-by-step math without branching.",
            estimated_complexity="Linear 2-phase (Plan -> Solve)",
        )

    # 3. Tree of Thoughts (ToT): Ranking, comparison, trade-off optimization
    if any(k in inst_lower for k in ["rank", "prioritize", "compare", "tradeoff", "portfolio", "options", "best strategy"]):
        return RoutingDecision(
            task_id=task_id,
            instruction=instruction,
            selected_method="ToT",
            rationale="Combinatorial multi-option reasoning task benefiting from candidate generation, heuristic evaluation, and branch search.",
            estimated_complexity="Tree search (BFS/DFS with branch pruning)",
        )

    # 4. LATS: Final policy endorsement proposals, high-stakes external compliance
    if any(k in inst_lower for k in ["propose", "endorse", "structure", "finalize", "underwriting", "compliance", "action plan"]):
        return RoutingDecision(
            task_id=task_id,
            instruction=instruction,
            selected_method="LATS",
            rationale="High-stakes action proposal with external constraints, requiring MCTS search guided by real Grounded Environment feedback.",
            estimated_complexity="MCTS 4-phase loop with verbal reflections",
        )

    # 5. Default synthesis / refinement
    return RoutingDecision(
        task_id=task_id,
        instruction=instruction,
        selected_method="SELF_REFINE",
        rationale="Customer communication or summary synthesis requiring single-draft rubric critique.",
        estimated_complexity="Iterative Critique & Revision",
    )
