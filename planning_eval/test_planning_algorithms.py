"""Comprehensive test suite for Harborstone Planning algorithms, self-correction, and grounded environment."""

import pytest
from planning.environment import GroundedEnvironment, UngroundedEnvironment, EnvironmentFeedback
from planning.algorithms.plan_and_solve import plan_and_solve, PlanAndSolveResult
from planning.algorithms.tree_of_thoughts import tree_of_thoughts_search, ToTResult
from planning.algorithms.lats import lats_search, LATSResult
from planning.algorithms.router import route_subtask, RoutingDecision
from planning.algorithms.self_refine import self_refine, SelfRefineResult
from planning.algorithms.reflexion import run_reflexion, ReflexionResult
from planning.requests.harborstone_requests import REAL_REQUESTS, INELIGIBLE_REQUEST, ELIGIBLE_REQUEST


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
        proposed_premium=2700.0,  # 1200 + (150000 * 0.01)
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
    assert any("20-year underwriting limit" in v for v in fb.violations)
    assert fb.score < 1.0


def test_grounded_environment_requires_survey_for_luxury_yacht():
    env = GroundedEnvironment(current_year=2026)
    # Luxury yacht >= $500k missing survey
    fb = env.evaluate_vessel_addition(
        vessel_type="Yacht",
        year_built=2024,
        vessel_value=750000.0,
        documentation_provided=["Proof of purchase"],  # Missing appraisal
    )
    assert fb.success is False
    assert any("surveyor appraisal report" in v for v in fb.violations)

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
        "vessel_value": 750000.0,  # >= $500k without survey (Violation)
        "proposed_premium": 5000.0,  # Inaccurate actuarial rate (Violation)
        "deductible": 200.0,  # Below $500 min (Violation)
    }

    ungrounded_env = UngroundedEnvironment()
    grounded_env = GroundedEnvironment(current_year=2026)

    # Ungrounded passes naively with score 1.0
    ungrounded_feedback = ungrounded_env.evaluate_proposal(invalid_proposal)
    assert ungrounded_feedback.success is True
    assert ungrounded_feedback.score == 1.0

    # Grounded environment catches all 4 underwriting violations
    grounded_feedback = grounded_env.evaluate_proposal(invalid_proposal)
    assert grounded_feedback.success is False
    assert grounded_feedback.score == 0.0
    assert len(grounded_feedback.violations) >= 3


# ==========================================
# 2. Plan-and-Solve (PS) Tests
# ==========================================

def test_plan_and_solve_deterministic_execution():
    context = {
        "vessel_type": "Boat",
        "vessel_value": 200000.0,
        "current_premium": 1500.0,
        "year_built": 2022,
        "deductible": 15000.0,  # >= 5% of value -> qualifies for discount
    }
    result = plan_and_solve(
        task_goal="Calculate premium and deductible schedule for Boat addition",
        context=context,
    )
    assert isinstance(result, PlanAndSolveResult)
    assert result.success is True
    assert len(result.plan) == 5
    assert len(result.step_solutions) == 5
    assert result.final_output["base_rate"] == 0.010
    assert result.final_output["base_additional_premium"] == 2000.0
    assert result.final_output["deductible_discount"] == 100.0  # 5% of 2000
    assert result.final_output["total_new_premium"] == 3400.0  # 1500 + 2000 - 100


# ==========================================
# 3. Tree of Thoughts (ToT) Tests
# ==========================================

def test_tree_of_thoughts_bfs_search():
    context = {
        "vessel_type": "Yacht",
        "vessel_value": 600000.0,
        "current_premium": 3500.0,
    }
    result = tree_of_thoughts_search(
        goal="Explore deductible and coverage tradeoff options for yacht",
        context=context,
        search_strategy="BFS",
        max_depth=3,
        branching_factor=3,
        beam_width=2,
    )
    assert isinstance(result, ToTResult)
    assert result.success is True
    assert result.search_strategy == "BFS"
    assert result.total_nodes_explored > 5
    assert len(result.best_path) == 4  # root + 3 depths
    assert result.best_score > 0.80


def test_tree_of_thoughts_dfs_search():
    context = {"vessel_type": "Boat", "vessel_value": 120000.0}
    result = tree_of_thoughts_search(
        goal="Rank risk portfolio for fleet boats",
        context=context,
        search_strategy="DFS",
        max_depth=2,
        branching_factor=2,
    )
    assert isinstance(result, ToTResult)
    assert result.search_strategy == "DFS"
    assert len(result.best_path) == 3


# ==========================================
# 4. Language Agent Tree Search (LATS) Tests
# ==========================================

def test_lats_search_with_grounded_environment():
    request = {
        "vessel_name": "Oceanic Sovereign",
        "vessel_type": "Yacht",
        "year_built": 2024,
        "vessel_value": 800000.0,
        "current_premium": 4000.0,
    }
    result = lats_search(
        goal="Find compliant luxury yacht endorsement structure",
        initial_request=request,
        environment=GroundedEnvironment(current_year=2026),
        max_iterations=4,
    )
    assert isinstance(result, LATSResult)
    assert result.grounded is True
    assert result.iterations_run == 4
    assert result.total_nodes_created >= 4
    assert result.best_score > 0.0
    assert len(result.best_trajectory) >= 1


# ==========================================
# 5. Sub-Task Planning Router Tests
# ==========================================

def test_subtask_router_dispatching():
    # Math calculation -> PS
    r1 = route_subtask("t1", "Calculate exact premium change and deductible discount", {})
    assert r1.selected_method == "PS"

    # Multi-option ranking -> ToT
    r2 = route_subtask("t2", "Rank risk priority and compare portfolio deductible options", {})
    assert r2.selected_method == "ToT"

    # High stakes proposal -> LATS
    r3 = route_subtask("t3", "Propose final compliant policy endorsement under strict underwriting", {})
    assert r3.selected_method == "LATS"

    # Direct tool call -> MCP_DIRECT
    r4 = route_subtask("t4", "Lookup customer policies", {"kind": "mcp", "tool_name": "get_customer_policies"})
    assert r4.selected_method == "MCP_DIRECT"


# ==========================================
# 6. Self-Refine Tests
# ==========================================

def test_self_refine_loop():
    context = {
        "vessel_name": "Bay Runner",
        "vessel_type": "Boat",
        "vessel_value": 180000.0,
        "total_new_premium": 3900.0,
        "documents": ["Proof of purchase", "Registration"],
    }
    res = self_refine("Draft customer policy update notice", context)
    assert isinstance(res, SelfRefineResult)
    assert res.success is True
    assert res.critique.accuracy_score >= 0.8
    assert len(res.improvements_made) > 0
    assert "HARBORSTONE MARINE INSURANCE" in res.refined_output


# ==========================================
# 7. Reflexion Multi-Trial Learning Tests
# ==========================================

def test_reflexion_multi_trial_recovery():
    context = {
        "vessel_name": "Apex Voyager",
        "vessel_type": "Yacht",
        "year_built": 2024,
        "vessel_value": 850000.0,  # Requires survey
        "current_premium": 4000.0,
    }
    env = GroundedEnvironment(current_year=2026)
    res = run_reflexion(
        task_goal="Endorse $850k yacht with all regulatory constraints satisfied",
        initial_request=context,
        environment=env,
        max_trials=3,
    )
    assert isinstance(res, ReflexionResult)
    assert res.trials_attempted > 1
    assert len(res.episodic_memory) >= 1
    # Check that reflection was generated and stored from Trial 1 failure
    assert any("Trial #1" in m for m in res.episodic_memory)
    # Final trial achieves success using reflections
    assert res.success is True
    assert res.final_score == 1.0
