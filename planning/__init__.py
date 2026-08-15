"""Harborstone Insurance Week 4 Autonomous Decomposition & Planning System.

Unified package interface exporting all planning algorithms, models, environment,
and integration components.
"""

from .models import Task, Plan
from .environment import (
    GroundedEnvironment,
    UngroundedEnvironment,
    EnvironmentFeedback,
    UnderwritingRules,
)
from .algorithms import (
    decompose_goal,
    execute_plan,
    final_output,
    dynamic_decomposition,
    plan_and_solve,
    PlanAndSolveResult,
    tree_of_thoughts_search,
    ToTResult,
    lats_search,
    LATSResult,
    route_subtask,
    RoutingDecision,
    self_refine,
    SelfRefineResult,
    run_reflexion,
    ReflexionResult,
)
from .integration.trace import RunTrace
from .integration.mcp_executor import HarborstoneMCPExecutor
from .requests.harborstone_requests import (
    REAL_REQUESTS,
    ELIGIBLE_REQUEST,
    INELIGIBLE_REQUEST,
)

__all__ = [
    # Models & DAGs
    "Task",
    "Plan",
    # Environment & Validation
    "GroundedEnvironment",
    "UngroundedEnvironment",
    "EnvironmentFeedback",
    "UnderwritingRules",
    # Algorithms
    "decompose_goal",
    "execute_plan",
    "final_output",
    "dynamic_decomposition",
    "plan_and_solve",
    "PlanAndSolveResult",
    "tree_of_thoughts_search",
    "ToTResult",
    "lats_search",
    "LATSResult",
    "route_subtask",
    "RoutingDecision",
    "self_refine",
    "SelfRefineResult",
    "run_reflexion",
    "ReflexionResult",
    # Integration & Traces
    "RunTrace",
    "HarborstoneMCPExecutor",
    # Fixtures
    "REAL_REQUESTS",
    "ELIGIBLE_REQUEST",
    "INELIGIBLE_REQUEST",
]
