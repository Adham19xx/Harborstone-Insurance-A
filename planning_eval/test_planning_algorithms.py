"""Comprehensive unit & integration test suite for Harborstone Planning algorithms,
self-correction, and grounded environment.
"""

import pytest
from planning.environment import GroundedEnvironment, UngroundedEnvironment, EnvironmentFeedback
from planning.algorithms.plan_and_solve import plan_and_solve, PlanAndSolveResult
from planning.algorithms.tree_of_thoughts import tree_of_thoughts, ToTResult
from planning.algorithms.lats import lats, LATSResult
from planning.algorithms.router import classify_subtask, route_subtask, explain_routing, RoutingDecision
from planning.algorithms.self_refine import reflect_and_refine, ReflectionResult
from planning.algorithms.reflexion import reflexion, ReflexionResult, EpisodicMemoryBuffer
from planning.requests.harborstone_requests import REAL_REQUESTS, INELIGIBLE_REQUEST, ELIGIBLE_REQUEST


# ---------------------------------------------------------------------------
# Test Mocks
# ---------------------------------------------------------------------------

class FakeStructured:
    def __init__(self, value):
        self.value = value

    def invoke(self, *_args, **_kwargs):
        return self.value


class MockLLM:
    """Deterministic Mock LLM for fast unit tests."""
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema, **_kwargs):
        schema_name = getattr(schema, "__name__", str(schema))
        if "ThoughtCandidates" in schema_name or "candidates" in str(schema):
            class CandidatesObj:
                candidates = [
                    "Option A: Standard 2.5% Deductible with full Hull & Machinery coverage",
                    "Option B: High 10.0% Deductible with Premium Discount",
                ]
            return FakeStructured(CandidatesObj())
        elif "ThoughtEvaluation" in schema_name:
            class EvalObj:
                score = 0.90
                rationale = "Compliant with Harborstone guidelines."
            return FakeStructured(EvalObj())
        elif "LATSActionBatch" in schema_name:
            class ActionItem:
                action = "Endorse yacht with compliant survey and premium"
                rationale = "Follows underwriting standards"
            class ActionBatch:
                actions = [ActionItem(), ActionItem()]
            return FakeStructured(ActionBatch())
        else:
            class Generic:
                score = 0.85
                rationale = "Valid"
            return FakeStructured(Generic())

    def invoke(self, *_args, **_kwargs):
        self.calls += 1
        class Response:
            content = (
                "Harborstone Insurance Endorsement: Vessel Boston Whaler (Boat), "
                "annual premium $2,700, eligible under standard policy guidelines. "
                "Required document list provided."
            )
            usage_metadata = {"input_tokens": 15, "output_tokens": 20, "total_tokens": 35}
        return Response()


# ==========================================
# 1. Grounded Environment & Contrast Tests
# ==========================================

def test_grounded_environment_approves_valid_vessel():
    env = GroundedEnvironment(current_year=2026)
    fb = env.evaluate_vessel_addition(
        vessel_type="Boat",
        year_built=2024,
        vessel_value=150000.0,
        current_premium=1200.0,
        proposed_premium=2700.0,
        deductible=2500.0,
        documentation_provided=["Proof of ownership/purchase invoice", "Current vessel registration"],
    )
    assert fb.success is True
    assert fb.score == 1.0
    assert len(fb.violations) == 0


def test_grounded_environment_rejects_over_age_vessel():
    env = GroundedEnvironment(current_year=2026)
    fb = env.evaluate_vessel_addition(
        vessel_type="Yacht",
        year_built=1998,  # Age 28 years > 20
        vessel_value=300000.0,
        current_premium=2000.0,
    )
    assert fb.success is False
    assert any("20-year underwriting limit" in v or "exceeds" in v for v in fb.violations)
    assert fb.score < 1.0


def test_grounded_environment_requires_survey_for_luxury_yacht():
    env = GroundedEnvironment(current_year=2026)
    # Luxury yacht >= $500k missing survey
    fb = env.evaluate_vessel_addition(
        vessel_type="Yacht",
        year_built=2024,
        vessel_value=750000.0,
        documentation_provided=["Proof of purchase"],
    )
    assert fb.success is False
    assert any("surveyor appraisal report" in v or "survey" in v for v in fb.violations)

    # Adding surveyor report resolves violation
    fb_valid = env.evaluate_vessel_addition(
        vessel_type="Yacht",
        year_built=2024,
        vessel_value=750000.0,
        documentation_provided=["Proof of purchase", "Recent independent marine surveyor appraisal report"],
    )
    assert not any("surveyor appraisal report" in v for v in fb_valid.violations)


def test_grounded_vs_ungrounded_critique_contrast():
    """
    Deliberately shows the failure case the grounded version catches that
    the ungrounded version missed (Rubric Requirement).
    """
    invalid_proposal = {
        "vessel_name": "Heritage Mariner",
        "vessel_type": "Yacht",
        "year_built": 1995,  # 31 years old (Violation)
        "value": 750000.0,  # >= $500k without survey (Violation)
        "proposed_premium": 5000.0,  # Inaccurate rate (Violation)
        "deductible": 200.0,  # Below $500 min (Violation)
    }

    ungrounded_env = UngroundedEnvironment()
    grounded_env = GroundedEnvironment(current_year=2026)

    # Ungrounded passes naively with score 1.0
    ungrounded_feedback = ungrounded_env.evaluate_proposal(invalid_proposal)
    assert ungrounded_feedback.success is True
    assert ungrounded_feedback.score == 1.0

    # Grounded environment catches underwriting violations
    grounded_feedback = grounded_env.evaluate_proposal(invalid_proposal)
    assert grounded_feedback.success is False
    assert grounded_feedback.score <= 0.35
    assert len(grounded_feedback.violations) >= 2


# ==========================================
# 2. Plan-and-Solve (PS) Tests
# ==========================================

def test_plan_and_solve_execution():
    llm = MockLLM()
    result = plan_and_solve(
        question="Calculate premium for Boston Whaler boat addition",
        llm=llm,
        tool_name="estimate_policy_premium_change",
        context={"vessel_value": 150000, "current_premium": 1200},
    )
    assert isinstance(result, PlanAndSolveResult)
    assert result.llm_calls == 2
    assert result.success is True
    assert result.plan != ""
    assert result.solution != ""



# ==========================================
# 3. Tree of Thoughts (ToT) Tests
# ==========================================

def test_tree_of_thoughts_search():
    llm = MockLLM()
    result = tree_of_thoughts(
        problem="Synthesize customer recommendation for adding Sunseeker Yacht",
        llm=llm,
        depth=2,
        beam_width=2,
    )
    assert isinstance(result, ToTResult)
    assert result.success is True
    assert len(result.all_thoughts) >= 2
    assert result.best_thought.score > 0.5


# ==========================================
# 4. Language Agent Tree Search (LATS) Tests
# ==========================================

def test_lats_search_with_grounded_environment():
    llm = MockLLM()
    env = GroundedEnvironment(current_year=2026)
    result = lats(
        task="Check eligibility and estimate premium for $800k Yacht",
        llm=llm,
        environment=env,
        max_iterations=2,
    )
    assert isinstance(result, LATSResult)
    assert result.llm_calls > 0
    assert result.iterations == 2


# ==========================================
# 5. Sub-Task Planning Router Tests
# ==========================================

def test_subtask_router_dispatching():
    # Deterministic lookup -> plan_and_solve
    algo1 = classify_subtask(kind="mcp", tool_name="get_customer_policies")
    assert algo1 == "plan_and_solve"

    # High-stakes action -> lats
    algo2 = classify_subtask(kind="mcp", tool_name="check_vessel_eligibility")
    assert algo2 == "lats"

    # Synthesis node -> tree_of_thoughts
    algo3 = classify_subtask(kind="synthesis", tool_name=None)
    assert algo3 == "tree_of_thoughts"

    # Router explanation
    exp = explain_routing("synthesis", None)
    assert "[Router]" in exp
    assert "TREE_OF_THOUGHTS" in exp


# ==========================================
# 6. Self-Refine Tests
# ==========================================

def test_self_refine_loop():
    llm = MockLLM()
    initial_draft = "Your yacht was added. Your premium changed."
    res = reflect_and_refine(
        goal="Synthesize formal policy update notice",
        draft=initial_draft,
        llm=llm,
    )
    assert isinstance(res, ReflectionResult)
    assert res.llm_calls == 3
    assert res.critique != ""
    assert res.revised != ""


# ==========================================
# 7. Reflexion Multi-Trial Learning Tests
# ==========================================

def test_reflexion_multi_trial_recovery():
    llm = MockLLM()
    env = GroundedEnvironment(current_year=2026)
    res = reflexion(
        task="Endorse high-value cargo vessel with compliant documentation",
        llm=llm,
        environment=env,
        max_trials=2,
    )
    assert isinstance(res, ReflexionResult)
    assert len(res.trials) > 0
    assert res.llm_calls > 0
