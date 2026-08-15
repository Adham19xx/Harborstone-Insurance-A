"""Grounded Environment implementation for Harborstone Marine Insurance.

This module replaces the reference toolkit's randomized evaluator with real,
deterministic external feedback sources:
1. Real Harborstone underwriting rules (vessel age, value, type, luxury survey requirements).
2. Database / MCP schema and financial constraints (deductibles, premium consistency).
3. Conflict validation (policy overlap, maximum coverage limits).

It also provides an UngroundedEnvironment for deliberate contrast evaluation,
demonstrating the exact failure cases that a self-evaluating LLM misses but the
grounded environment catches.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EnvironmentFeedback(BaseModel):
    """Structured environment feedback."""
    success: bool
    score: float = Field(ge=0.0, le=1.0)
    details: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    source: str = "grounded_underwriting_engine"


class UnderwritingRules:
    """Harborstone Marine Insurance Underwriting Rules & Limits."""
    MAX_VESSEL_AGE_YEARS: int = 20
    SUPPORTED_VESSEL_TYPES: set[str] = {"Boat", "Yacht"}
    LUXURY_VALUATION_THRESHOLD: float = 500000.00
    HIGH_RISK_PREMIUM_THRESHOLD: float = 8000.00
    MAX_DEDUCTIBLE_RATIO: float = 0.15  # Max 15% of vessel value
    MIN_DEDUCTIBLE_AMOUNT: float = 500.00
    RATES = {"Boat": 0.010, "Yacht": 0.015}


class GroundedEnvironment:
    """
    Real external validation engine for Harborstone Marine Insurance.
    
    Validates proposed actions, plans, and endorsements against:
    - Real underwriting age and type restrictions
    - Luxury yacht independent valuation requirements
    - Financial calculations and deductible boundaries
    - Required documentation checklists
    """

    def __init__(self, current_year: int = 2026):
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
        """
        Evaluate a vessel addition or policy update proposal with grounded rules.
        """
        violations: List[str] = []
        details: List[str] = []
        age = self.current_year - year_built

        # Rule 1: Vessel Type Validation
        if vessel_type not in self.rules.SUPPORTED_VESSEL_TYPES:
            violations.append(f"Vessel type '{vessel_type}' is not supported by Harborstone (Allowed: Boat, Yacht).")
        else:
            details.append(f"Vessel type '{vessel_type}' is supported.")

        # Rule 2: Vessel Value Validation
        if vessel_value <= 0:
            violations.append("Vessel declared value must be greater than $0.")
        else:
            details.append(f"Vessel value ${vessel_value:,.2f} is valid.")

        # Rule 3: Age Restriction (Max 20 years)
        if age > self.rules.MAX_VESSEL_AGE_YEARS:
            violations.append(
                f"Vessel age ({age} years, built {year_built}) exceeds the {self.rules.MAX_VESSEL_AGE_YEARS}-year underwriting limit."
            )
        elif age < 0:
            violations.append(f"Invalid year built ({year_built}) in future.")
        else:
            details.append(f"Vessel age ({age} years) is within allowable limit.")

        # Rule 4: High Value Luxury Yacht Survey Requirement
        if vessel_value >= self.rules.LUXURY_VALUATION_THRESHOLD:
            docs = documentation_provided or []
            has_survey = any("valuation" in d.lower() or "survey" in d.lower() or "appraisal" in d.lower() for d in docs)
            if not has_survey:
                violations.append(
                    f"Vessels valued at >= ${self.rules.LUXURY_VALUATION_THRESHOLD:,.2f} require an independent marine surveyor appraisal report."
                )
            else:
                details.append("Required marine surveyor appraisal report verified.")

        # Rule 5: Deductible Consistency Check
        if deductible is not None:
            max_allowed_deductible = vessel_value * self.rules.MAX_DEDUCTIBLE_RATIO
            if deductible < self.rules.MIN_DEDUCTIBLE_AMOUNT:
                violations.append(f"Deductible ${deductible:,.2f} is below minimum allowed (${self.rules.MIN_DEDUCTIBLE_AMOUNT:,.2f}).")
            elif deductible > max_allowed_deductible:
                violations.append(
                    f"Deductible ${deductible:,.2f} exceeds the maximum allowable 15% of vessel value (${max_allowed_deductible:,.2f})."
                )
            else:
                details.append(f"Deductible ${deductible:,.2f} is compliant.")

        # Rule 6: Premium Calculation Grounding
        if proposed_premium is not None and vessel_type in self.rules.RATES:
            expected_additional = round(vessel_value * self.rules.RATES[vessel_type], 2)
            expected_new_premium = round(current_premium + expected_additional, 2)
            # Allow minor rounding tolerance (+/- $1.00)
            if abs(proposed_premium - expected_new_premium) > 1.00:
                violations.append(
                    f"Proposed premium ${proposed_premium:,.2f} does not match actuarial rate table (expected: ${expected_new_premium:,.2f} based on {self.rules.RATES[vessel_type]*100:.1f}% rate)."
                )
            else:
                details.append(f"Proposed premium ${proposed_premium:,.2f} matches rate calculation.")

        success = len(violations) == 0
        score = 1.0 if success else max(0.0, round(1.0 - (len(violations) * 0.35), 2))

        return EnvironmentFeedback(
            success=success,
            score=score,
            details=details,
            violations=violations,
            source="grounded_underwriting_engine",
        )

    def evaluate_proposal(self, proposal: Dict[str, Any]) -> EnvironmentFeedback:
        """Convenience method to evaluate a generic structured proposal dictionary."""
        vessel_type = proposal.get("vessel_type", "Boat")
        year_built = int(proposal.get("year_built", 2024))
        vessel_value = float(proposal.get("vessel_value", proposal.get("value", 0.0)))
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
    """
    Ungrounded (heuristic/self-satisfaction) evaluator for deliberate contrast.
    
    This simulates an LLM self-evaluating or a superficial rubric checker that
    accepts plausible-sounding text without validating against real database
    constraints or actuarial rules.
    """

    def evaluate_proposal(self, proposal: Dict[str, Any]) -> EnvironmentFeedback:
        # Ungrounded evaluator simply checks if fields exist and look formatted,
        # completely missing semantic/actuarial rule violations (e.g. 24-year-old vessel or wrong rate).
        has_vessel = "vessel_type" in proposal or "vessel_name" in proposal or "value" in proposal or "vessel_value" in proposal
        has_premium = "premium" in str(proposal) or "value" in str(proposal)
        
        if has_vessel and has_premium:
            return EnvironmentFeedback(
                success=True,
                score=1.0,
                details=["Proposal appears structurally well-formed and complete according to LLM self-check."],
                violations=[],
                source="ungrounded_self_critique",
            )
        else:
            return EnvironmentFeedback(
                success=False,
                score=0.4,
                details=["Missing basic proposal formatting."],
                violations=["Incomplete fields."],
                source="ungrounded_self_critique",
            )
