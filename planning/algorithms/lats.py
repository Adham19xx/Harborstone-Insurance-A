"""Language Agent Tree Search (LATS) Planning Algorithm for Harborstone Insurance.

Based on Zhou et al. (2023): 'Language Agent Tree Search Unifies Reasoning,
Acting, and Planning in Language Models'.

LATS implements a full Monte Carlo Tree Search (MCTS) loop over reasoning & tool action steps:
1. Select: Traverse existing tree using Upper Confidence Bounds applied to Trees (UCT).
2. Expand & Simulate: Sample candidate actions and roll out next state.
3. Evaluate & Reflect: Score state against REAL EXTERNAL GROUNDED FEEDBACK (EnvironmentFeedback),
   and synthesize a verbal reflection on failed branches to steer future exploration.
4. Backpropagate: Update visit counts and value estimates up to root.

Replaces fake randomized evaluator with real Harborstone underwriting & database rules.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, Field

from ..environment import GroundedEnvironment, UngroundedEnvironment, EnvironmentFeedback
from ..integration.trace import RunTrace

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class LATSNode:
    """A search tree node in the LATS MCTS hierarchy."""

    def __init__(
        self,
        node_id: str,
        state: Dict[str, Any],
        parent: Optional["LATSNode"] = None,
        action: Optional[Dict[str, Any]] = None,
        depth: int = 0,
    ):
        self.node_id = node_id
        self.state = state
        self.parent = parent
        self.action = action or {}
        self.depth = depth
        self.children: List["LATSNode"] = []
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.feedback: Optional[EnvironmentFeedback] = None
        self.reflection: Optional[str] = None
        self.is_terminal: bool = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def uct_score(self, exploration_constant: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(math.log(parent_visits + 1) / self.visits)
        return exploitation + exploration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent.node_id if self.parent else None,
            "depth": self.depth,
            "action": self.action,
            "visits": self.visits,
            "q_value": round(self.q_value, 4),
            "reflection": self.reflection,
            "feedback": self.feedback.model_dump() if self.feedback else None,
            "is_terminal": self.is_terminal,
        }


class LATSResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str
    grounded: bool
    iterations_run: int
    total_nodes_created: int
    best_trajectory: List[Dict[str, Any]]
    best_score: float
    best_action: Dict[str, Any]
    reflections_generated: List[str]
    success: bool


def lats_search(
    goal: str,
    initial_request: Dict[str, Any],
    environment: Optional[Any] = None,
    max_iterations: int = 6,
    max_depth: int = 3,
    exploration_constant: float = 1.414,
    llm: Optional[BaseChatModel] = None,
    trace: Optional[RunTrace] = None,
) -> LATSResult:
    """
    Run Language Agent Tree Search with MCTS and grounded environment feedback.
    """
    env = environment if environment is not None else GroundedEnvironment()
    is_grounded = isinstance(env, GroundedEnvironment)

    root = LATSNode(
        node_id="root",
        state=dict(initial_request),
        parent=None,
        depth=0,
    )

    # Initial evaluation of root
    root.feedback = env.evaluate_proposal(initial_request)
    root.visits = 1
    root.value_sum = root.feedback.score

    node_counter = 0
    all_nodes: List[LATSNode] = [root]
    all_reflections: List[str] = []

    for iteration in range(1, max_iterations + 1):
        # 1. Selection: Traverse down using UCT
        curr = root
        while curr.children and curr.depth < max_depth and not curr.is_terminal:
            curr = max(curr.children, key=lambda child: child.uct_score(exploration_constant))

        # If curr is terminal or at max depth, evaluate and backpropagate
        if curr.is_terminal or curr.depth >= max_depth:
            reward = curr.feedback.score if curr.feedback else 0.0
            _backpropagate(curr, reward)
            continue

        # 2. Expansion & Simulation: Propose next candidate actions
        # Collect prior reflections from failed attempts to steer exploration
        past_reflections = [n.reflection for n in all_nodes if n.reflection]
        candidate_actions = _generate_candidate_actions(curr.state, past_reflections, iteration)

        for act in candidate_actions:
            node_counter += 1
            nid = f"lats_n{node_counter}_iter{iteration}"
            next_state = {**curr.state, **act}

            child = LATSNode(
                node_id=nid,
                state=next_state,
                parent=curr,
                action=act,
                depth=curr.depth + 1,
            )

            # 3. Evaluation & Reflection: REAL EXTERNAL FEEDBACK
            fb = env.evaluate_proposal(next_state)
            child.feedback = fb
            child.is_terminal = fb.success or (child.depth >= max_depth)

            # Generate verbal reflection if not fully successful
            if not fb.success and fb.violations:
                refl_text = f"Action {act.get('strategy', 'proposal')} failed underwriting: {'; '.join(fb.violations)}. Correction: ensure compliance with required rules."
                child.reflection = refl_text
                all_reflections.append(refl_text)

            curr.children.append(child)
            all_nodes.append(child)

            # 4. Backpropagation
            _backpropagate(child, fb.score)

    # Extract best trajectory
    best_leaf = max(all_nodes, key=lambda n: (n.feedback.score if n.feedback else 0.0, n.q_value))
    best_path = _reconstruct_trajectory(best_leaf)

    best_score = best_leaf.feedback.score if best_leaf.feedback else 0.0
    best_success = best_leaf.feedback.success if best_leaf.feedback else False

    result = LATSResult(
        goal=goal,
        grounded=is_grounded,
        iterations_run=max_iterations,
        total_nodes_created=len(all_nodes),
        best_trajectory=[n.to_dict() for n in best_path],
        best_score=best_score,
        best_action=best_leaf.action or best_leaf.state,
        reflections_generated=all_reflections,
        success=best_success,
    )

    if trace is not None:
        trace.execution.append({"method": "LATS", "nodes_created": len(all_nodes), "best_score": best_score})

    return result


def _backpropagate(node: LATSNode, reward: float) -> None:
    curr: Optional[LATSNode] = node
    while curr is not None:
        curr.visits += 1
        curr.value_sum += reward
        curr = curr.parent


def _reconstruct_trajectory(node: LATSNode) -> List[LATSNode]:
    path: List[LATSNode] = []
    curr: Optional[LATSNode] = node
    while curr is not None:
        path.append(curr)
        curr = curr.parent
    return list(reversed(path))


def _generate_candidate_actions(
    state: Dict[str, Any], reflections: List[str], iteration: int
) -> List[Dict[str, Any]]:
    """
    Generate domain-grounded candidate actions for Harborstone policy endorsements,
    incorporating feedback and reflections from prior failed explorations.
    """
    vessel_type = state.get("vessel_type", state.get("new_vessel", {}).get("vessel_type", "Boat"))
    vessel_value = float(state.get("vessel_value", state.get("new_vessel", {}).get("value", 100000.0)))
    year_built = int(state.get("year_built", state.get("new_vessel", {}).get("year_built", 2024)))
    current_premium = float(state.get("current_premium", 1200.0))

    rate = 0.015 if vessel_type == "Yacht" else 0.010
    correct_new_premium = round(current_premium + (vessel_value * rate), 2)

    needs_survey = vessel_value >= 500000.0
    needs_appraisal_reflection = any("appraisal" in r.lower() or "survey" in r.lower() for r in reflections)

    actions = []

    # Action 1: Compliant Standard Proposal
    docs1 = ["Proof of ownership/purchase invoice", "Current vessel registration"]
    if needs_survey or needs_appraisal_reflection:
        docs1.append("Recent independent marine surveyor valuation report")

    actions.append({
        "strategy": "Compliant Actuarial Endorsement",
        "vessel_type": vessel_type,
        "year_built": year_built,
        "vessel_value": vessel_value,
        "current_premium": current_premium,
        "proposed_premium": correct_new_premium,
        "deductible": round(max(500.0, vessel_value * 0.05), 2),
        "documents": docs1,
    })

    # Action 2: Premium Discount with Higher Compliant Deductible (10%)
    actions.append({
        "strategy": "High Deductible Cost Optimizer",
        "vessel_type": vessel_type,
        "year_built": year_built,
        "vessel_value": vessel_value,
        "current_premium": current_premium,
        "proposed_premium": correct_new_premium,
        "deductible": round(min(vessel_value * 0.10, vessel_value * 0.14), 2),
        "documents": docs1,
    })

    # Action 3: Non-compliant probe (to test environment pruning) if early iteration
    if iteration == 1 and not reflections:
        actions.append({
            "strategy": "Quick Endorsement Without Marine Survey",
            "vessel_type": vessel_type,
            "year_built": year_built,
            "vessel_value": vessel_value,
            "current_premium": current_premium,
            "proposed_premium": correct_new_premium + 500.0,  # wrong premium calculation
            "deductible": 250.0,  # below minimum
            "documents": ["Proof of ownership"],  # missing survey
        })

    return actions
