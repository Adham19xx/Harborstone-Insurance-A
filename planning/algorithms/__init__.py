# ── Person 1 (unchanged) ──────────────────────────────────────────────────
from .decomposition import decompose_goal, execute_plan, final_output
from .dynamic_decomposition import dynamic_decomposition

# ── Person 2: Planning Algorithms ─────────────────────────────────────────
from .plan_and_solve import plan_and_solve, PlanAndSolveResult
from .tree_of_thoughts import tree_of_thoughts, ToTResult, Thought
from .lats import lats, LATSResult, LATSNode

# ── Person 2: Self-Correction ─────────────────────────────────────────────
from .self_refine import reflect_and_refine, ReflectionResult
from .reflexion import reflexion, ReflexionResult, EpisodicMemoryBuffer

# ── Person 2: Grounded Environment ────────────────────────────────────────
from .environment import Environment, EnvironmentFeedback

# ── Person 2: Router ──────────────────────────────────────────────────────
from .router import classify_subtask, explain_routing, route_subtask, RoutingDecision

# ── Aliases for backward & forward compatibility ─────────────────────────
tree_of_thoughts_search = tree_of_thoughts
lats_search = lats
self_refine = reflect_and_refine
SelfRefineResult = ReflectionResult
run_reflexion = reflexion
GroundedEnvironment = Environment


__all__ = [
    # Person 1
    "decompose_goal", "execute_plan", "final_output",
    "dynamic_decomposition",
    # Person 2 — Planning
    "plan_and_solve", "PlanAndSolveResult",
    "tree_of_thoughts", "tree_of_thoughts_search", "ToTResult", "Thought",
    "lats", "lats_search", "LATSResult", "LATSNode",
    # Person 2 — Self-Correction
    "reflect_and_refine", "self_refine", "ReflectionResult", "SelfRefineResult",
    "reflexion", "run_reflexion", "ReflexionResult", "EpisodicMemoryBuffer",
    # Person 2 — Environment
    "Environment", "GroundedEnvironment", "EnvironmentFeedback",
    # Person 2 — Router
    "classify_subtask", "route_subtask", "explain_routing", "RoutingDecision",
]

