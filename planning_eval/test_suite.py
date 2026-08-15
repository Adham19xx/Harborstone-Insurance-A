"""
planning_eval/test_suite.py — Fixed Test Suite for Harborstone Planning
=====================================================================
This test suite is FIXED once evaluation starts (per the PDF requirement:
"Keep your planning test suite fixed once you start evaluating.")

Ten real-request test cases are defined here:
  - TC-01 to TC-04 : favor decomposition-first
  - TC-05 to TC-06 : favor dynamic decomposition (early surprise changes course)
  - TC-07 to TC-08 : need lookahead search (ToT or LATS)
  - TC-09 to TC-10 : need cross-trial memory (only Reflexion helps)

Each test case includes:
  - request    : the natural-language customer request
  - request_data: structured data (customer_id, vessel info)
  - expected_methods: which algorithms should be used
  - min_success_score: the grounded environment threshold to pass
  - tags: for filtering in run_eval.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestCase:
    __test__ = False
    id: str
    description: str
    request: str
    request_data: dict[str, Any]
    expected_methods: list[str]    # which algorithms this case exercises
    min_success_score: float = 0.65
    tags: list[str] = field(default_factory=list)



# ---------------------------------------------------------------------------
# The fixed test suite
# ---------------------------------------------------------------------------

TEST_SUITE: list[TestCase] = [
    # ── TC-01: Simple vessel add — favors decomposition-first ──────────────
    TestCase(
        id="TC-01",
        description="Add a new cargo vessel to an existing policy — straightforward",
        request=(
            "I'd like to add my new cargo vessel 'MV Harborstone Pride' "
            "(built 2019, value $2,100,000) to my existing marine policy."
        ),
        request_data={
            "customer_id": 1,
            "new_vessel": {"name": "MV Harborstone Pride", "type": "cargo", "year_built": 2019, "value": 2100000},
        },
        expected_methods=["plan_and_solve", "decomposition_first"],
        tags=["decomposition_first", "plan_and_solve"],
    ),
    # ── TC-02: Coverage lookup + requirements — favors decomposition-first ─
    TestCase(
        id="TC-02",
        description="Customer asks what documents are needed to update a tanker policy",
        request=(
            "My tanker vessel was recently upgraded. What documents do I need "
            "to update my hull insurance policy? Vessel value is now $5,800,000."
        ),
        request_data={
            "customer_id": 2,
            "new_vessel": {"type": "tanker", "value": 5800000},
        },
        expected_methods=["plan_and_solve", "decomposition_first"],
        tags=["decomposition_first", "plan_and_solve"],
    ),
    # ── TC-03: Fishing vessel eligibility check ────────────────────────────
    TestCase(
        id="TC-03",
        description="Eligibility check for a 1998 fishing vessel — deterministic",
        request=(
            "Can you check if my 1998 fishing vessel worth $320,000 qualifies "
            "for our standard marine hull policy?"
        ),
        request_data={
            "customer_id": 3,
            "new_vessel": {"type": "fishing", "year_built": 1998, "value": 320000},
        },
        expected_methods=["lats", "plan_and_solve"],
        min_success_score=0.70,
        tags=["decomposition_first", "lats"],
    ),
    # ── TC-04: Premium estimation for yacht ──────────────────────────────
    TestCase(
        id="TC-04",
        description="Estimate premium change for adding a luxury yacht",
        request=(
            "I want to add a 2022 luxury yacht (value $4,500,000) to policy #7. "
            "What will my new annual premium be?"
        ),
        request_data={
            "customer_id": 4,
            "policy_id": 7,
            "new_vessel": {"type": "yacht", "year_built": 2022, "value": 4500000},
        },
        expected_methods=["lats", "plan_and_solve"],
        min_success_score=0.70,
        tags=["decomposition_first", "lats"],
    ),
    # ── TC-05: Dynamic divergence — ineligible vessel reshapes the plan ───
    TestCase(
        id="TC-05",
        description=(
            "Old vessel triggers eligibility failure → dynamic decomposition "
            "changes course (decomposition-first would proceed blindly)"
        ),
        request=(
            "Please assess whether my 1985 passenger ferry (value $180,000) "
            "can be added to any of my existing policies."
        ),
        request_data={
            "customer_id": 5,
            "new_vessel": {"type": "passenger", "year_built": 1985, "value": 180000},
        },
        expected_methods=["dynamic_decomposition", "lats"],
        min_success_score=0.65,
        tags=["dynamic_decomposition", "lats", "divergence_case"],
    ),
    # ── TC-06: Dynamic divergence — missing policy forces re-plan ─────────
    TestCase(
        id="TC-06",
        description=(
            "Customer has no existing policies → dynamic decomposition pivots "
            "to 'new policy' path instead of 'update' path"
        ),
        request=(
            "I want to add a cargo vessel to my marine policy. "
            "Customer ID is 99. Vessel built 2020, value $1,250,000."
        ),
        request_data={
            "customer_id": 99,    # no policies in DB for this ID
            "new_vessel": {"type": "cargo", "year_built": 2020, "value": 1250000},
        },
        expected_methods=["dynamic_decomposition", "plan_and_solve"],
        tags=["dynamic_decomposition", "plan_and_solve", "divergence_case"],
    ),
    # ── TC-07: Lookahead needed — competing update strategies ─────────────
    TestCase(
        id="TC-07",
        description="Two valid policy update paths exist; ToT picks the better one",
        request=(
            "I have three active marine policies. I want to add a tanker "
            "(built 2018, value $3,000,000). Which policy should I update, "
            "and what is the best way to structure the update?"
        ),
        request_data={
            "customer_id": 6,
            "new_vessel": {"type": "tanker", "year_built": 2018, "value": 3000000},
        },
        expected_methods=["tree_of_thoughts", "lats"],
        min_success_score=0.65,
        tags=["tree_of_thoughts", "lookahead"],
    ),
    # ── TC-08: Ambiguous vessel type needs lookahead ──────────────────────
    TestCase(
        id="TC-08",
        description="Ambiguous request — 'workboat' not in schema, ToT resolves it",
        request=(
            "I have a workboat used for cargo transport. It was built in 2015 "
            "and is valued at $750,000. Can I add it to my policy?"
        ),
        request_data={
            "customer_id": 7,
            "new_vessel": {"type": "workboat", "year_built": 2015, "value": 750000},  # invalid type
        },
        expected_methods=["tree_of_thoughts", "plan_and_solve"],
        tags=["tree_of_thoughts", "lookahead", "invalid_vessel_type"],
    ),
    # ── TC-09: Cross-trial memory needed — complex synthesis fails twice ──
    TestCase(
        id="TC-09",
        description=(
            "Full proposal generation fails on first two trials due to missing "
            "numeric premium + missing docs; only Reflexion's episodic memory fixes it"
        ),
        request=(
            "Generate a complete policy update proposal for customer 8, "
            "who wants to add a cargo vessel (built 2021, value $6,500,000). "
            "Include eligibility decision, estimated premium impact, and all "
            "required documentation."
        ),
        request_data={
            "customer_id": 8,
            "new_vessel": {"type": "cargo", "year_built": 2021, "value": 6500000},
        },
        expected_methods=["reflexion", "lats"],
        min_success_score=0.70,
        tags=["reflexion", "cross_trial_memory"],
    ),
    # ── TC-10: Self-Refine vs Reflexion on synthesis quality ─────────────
    TestCase(
        id="TC-10",
        description=(
            "Synthesis output needs rubric improvement; compare Self-Refine "
            "(one draft + critique) vs Reflexion (multiple trials)"
        ),
        request=(
            "Write a customer-facing summary explaining the outcome of the "
            "vessel eligibility check and the new premium estimate for "
            "customer 9's policy update (fishing vessel, value $450,000)."
        ),
        request_data={
            "customer_id": 9,
            "new_vessel": {"type": "fishing", "year_built": 2016, "value": 450000},
        },
        expected_methods=["self_refine", "reflexion", "tree_of_thoughts"],
        tags=["self_refine", "reflexion", "synthesis"],
    ),
]


def get_test_case(tc_id: str) -> TestCase | None:
    for tc in TEST_SUITE:
        if tc.id == tc_id:
            return tc
    return None


def get_tests_by_tag(tag: str) -> list[TestCase]:
    return [tc for tc in TEST_SUITE if tag in tc.tags]
