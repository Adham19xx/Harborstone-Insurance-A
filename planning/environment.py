"""Grounded Environment implementation for Harborstone Marine Insurance.

Re-exports and extends the Grounded Environment in planning/algorithms/environment.py.
Provides GroundedEnvironment and UngroundedEnvironment for deliberate contrast evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Re-export canonical Grounded Environment and Feedback from algorithms package
from .algorithms.environment import (
    Environment,
    EnvironmentFeedback,
    ALLOWED_VESSEL_TYPES,
    REQUIRED_MCP_KEYS,
)


class UnderwritingRules:
    """Harborstone Marine Insurance Underwriting Rules & Limits."""
    MAX_VESSEL_AGE_YEARS: int = 20
    SUPPORTED_VESSEL_TYPES: set[str] = ALLOWED_VESSEL_TYPES
    LUXURY_VALUATION_THRESHOLD: float = 500000.00
    HIGH_RISK_PREMIUM_THRESHOLD: float = 8000.00
    MAX_DEDUCTIBLE_RATIO: float = 0.15
    MIN_DEDUCTIBLE_AMOUNT: float = 500.00
    RATES = {"cargo": 0.010, "tanker": 0.015, "passenger": 0.012, "fishing": 0.008, "yacht": 0.015, "Boat": 0.010, "Yacht": 0.015}


class GroundedEnvironment(Environment):
    """
    Enhanced Grounded Environment wrapping schema validation, actuarial rules,
    and MCP tool result checks.
    """

    def __init__(self, current_year: int = 2026, success_threshold: float = 0.65, tool_name: Optional[str] = None):
        super().__init__(success_threshold=success_threshold, tool_name=tool_name)
        self.current_year = current_year
        self.rules = UnderwritingRules()

    def evaluate_vessel_addition(
        self,
        vessel_type: str,
        year_built: int,
        vessel_value: float,
        current_premium: float = 0.0,
        proposed_premium: Optional[float] = None,
        deductible: Optional[float] = None,
        documentation_provided: Optional[List[str]] = None,
    ) -> EnvironmentFeedback:
        violations: List[str] = []
        passed: List[str] = []
        age = self.current_year - year_built

        # Rule 1: Type
        vtype_lower = vessel_type.lower()
        if vtype_lower not in {t.lower() for t in self.rules.SUPPORTED_VESSEL_TYPES} and vessel_type not in {"Boat", "Yacht"}:
            violations.append(f"Vessel type '{vessel_type}' is not supported.")
        else:
            passed.append(f"Vessel type '{vessel_type}' is valid.")

        # Rule 2: Value
        if vessel_value <= 0:
            violations.append("Vessel declared value must be positive.")
        else:
            passed.append(f"Vessel value ${vessel_value:,.2f} is valid.")

        # Rule 3: Age
        if age > self.rules.MAX_VESSEL_AGE_YEARS:
            violations.append(f"Vessel age ({age} yrs) exceeds {self.rules.MAX_VESSEL_AGE_YEARS}-year underwriting limit.")
        else:
            passed.append(f"Vessel age ({age} yrs) is compliant.")

        # Rule 4: Luxury Survey
        if vessel_value >= self.rules.LUXURY_VALUATION_THRESHOLD:
            docs = documentation_provided or []
            has_survey = any("survey" in d.lower() or "appraisal" in d.lower() or "valuation" in d.lower() for d in docs)
            if not has_survey:
                violations.append("Vessels valued >= $500k require independent marine surveyor appraisal report.")
            else:
                passed.append("Marine surveyor appraisal verified.")

        # Rule 5: Deductible
        if deductible is not None:
            if deductible < self.rules.MIN_DEDUCTIBLE_AMOUNT:
                violations.append(f"Deductible ${deductible:,.2f} is below minimum allowed (${self.rules.MIN_DEDUCTIBLE_AMOUNT:,.2f}).")
            elif deductible > (vessel_value * self.rules.MAX_DEDUCTIBLE_RATIO):
                violations.append(f"Deductible exceeds max 15% limit.")
            else:
                passed.append(f"Deductible ${deductible:,.2f} is compliant.")

        success = len(violations) == 0
        score = 1.0 if success else max(0.0, round(1.0 - len(violations) * 0.35, 2))

        return EnvironmentFeedback(
            score=score,
            success=success,
            message="; ".join(passed) if success else "; ".join(violations),
            grounding_source="Grounded Underwriting Engine",
            caught_issues=violations,
            passed_checks=passed,
        )

    def evaluate_proposal(self, proposal: Dict[str, Any]) -> EnvironmentFeedback:
        vessel_type = proposal.get("vessel_type", proposal.get("new_vessel", {}).get("type", proposal.get("new_vessel", {}).get("vessel_type", "yacht")))
        year_built = int(proposal.get("year_built", proposal.get("new_vessel", {}).get("year_built", 2024)))
        vessel_value = float(proposal.get("vessel_value", proposal.get("value", proposal.get("new_vessel", {}).get("value", 100000.0))))
        current_premium = float(proposal.get("current_premium", 0.0))
        proposed_premium = float(proposal["proposed_premium"]) if "proposed_premium" in proposal else None
        deductible = float(proposal["deductible"]) if "deductible" in proposal else None
        docs = proposal.get("documents", proposal.get("required_documents", []))

        return self.evaluate_vessel_addition(
            vessel_type=vessel_type,
            year_built=year_built,
            vessel_value=vessel_value,
            current_premium=current_premium,
            proposed_premium=proposed_premium,
            deductible=deductible,
            documentation_provided=docs,
        )


class UngroundedEnvironment:
    """Ungrounded self-evaluating critique for deliberate contrast demonstration."""

    def evaluate_proposal(self, proposal: Dict[str, Any]) -> EnvironmentFeedback:
        has_vessel = "vessel_type" in proposal or "vessel_name" in proposal or "value" in proposal or "new_vessel" in str(proposal)
        if has_vessel:
            return EnvironmentFeedback(
                score=1.0,
                success=True,
                message="Proposal structurally accepted by ungrounded self-evaluation.",
                grounding_source="Ungrounded Self-Critique",
                caught_issues=[],
                passed_checks=["Form looks complete"],
            )
        return EnvironmentFeedback(
            score=0.3,
            success=False,
            message="Missing fields",
            grounding_source="Ungrounded Self-Critique",
            caught_issues=["Incomplete"],
            passed_checks=[],
        )
