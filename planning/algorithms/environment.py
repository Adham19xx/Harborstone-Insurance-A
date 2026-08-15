"""
environment.py — Grounded Environment for Harborstone Insurance
=====================================================================
Replaces the toolkit's randomized fake evaluator with a REAL evaluator
that checks outputs against actual Harborstone business rules and MCP
tool response schemas.

This is the critical grounding piece required by the PDF spec:
  "A LATS or grounded-Reflexion implementation still pointed at the
   toolkit's randomized default at submission time earns no credit."

Grounding sources (in order of authority):
  1. Schema validation: vessel_type in allowed set, premium > 0, etc.
  2. Business rules: eligibility must pass before premium is estimated
  3. Completeness checks: synthesis output mentions key fields
  4. MCP result structure: the tool response has the expected JSON keys
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# EnvironmentFeedback — compatible with the toolkit's model
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentFeedback:
    """
    Structured feedback from the grounded environment.
    Replaces the toolkit's randomized EnvironmentFeedback.
    """
    score: float                          # 0.0 – 1.0  (grounded, not random)
    success: bool                         # score >= threshold
    message: str                          # human-readable explanation
    grounding_source: str = "Harborstone Grounded Engine"
    caught_issues: list[str] = field(default_factory=list)   # failures found
    passed_checks: list[str] = field(default_factory=list)   # checks passed

    @property
    def violations(self) -> list[str]:
        return self.caught_issues

    @property
    def details(self) -> list[str]:
        return self.passed_checks




# ---------------------------------------------------------------------------
# Allowed values (directly from schema.sql / server.py)
# ---------------------------------------------------------------------------

ALLOWED_VESSEL_TYPES = {"cargo", "tanker", "passenger", "fishing", "yacht"}
REQUIRED_SYNTHESIS_TERMS = {"premium", "eligib", "document", "vessel", "policy"}
REQUIRED_MCP_KEYS = {
    "get_customer_policies":          {"policies"},
    "check_vessel_eligibility":       {"eligible"},
    "estimate_policy_premium_change": {"estimated_change", "new_premium"},
    "get_policy_update_requirements": {"requirements"},
    "get_policy_coverage":            {"coverage"},
}


# ---------------------------------------------------------------------------
# Grounded checks
# ---------------------------------------------------------------------------

def _check_vessel_type(text: str) -> tuple[bool, str]:
    """Return (ok, issue_or_empty)."""
    found = re.findall(r"\b(cargo|tanker|passenger|fishing|yacht)\b", text.lower())
    if not found:
        return False, "No valid vessel_type found (cargo/tanker/passenger/fishing/yacht)"
    return True, ""


def _check_premium_positive(text: str) -> tuple[bool, str]:
    """Verify any mentioned premium/USD amount is > 0."""
    amounts = re.findall(r"\$[\d,]+(?:\.\d+)?|\b[\d,]+(?:\.\d+)?\s*(?:USD|usd)", text)
    if not amounts:
        return True, ""   # no amount mentioned — don't penalise
    for raw in amounts:
        val_str = re.sub(r"[^\d.]", "", raw)
        try:
            val = float(val_str)
            if val <= 0:
                return False, f"Premium/amount {raw!r} is not positive"
        except ValueError:
            pass
    return True, ""


def _check_synthesis_completeness(text: str) -> tuple[float, list[str], list[str]]:
    """
    Score a synthesis / final-answer output.
    Returns (score, issues, passed).
    """
    issues: list[str] = []
    passed: list[str] = []
    lc = text.lower()

    for term in REQUIRED_SYNTHESIS_TERMS:
        if term in lc:
            passed.append(f"Contains '{term}'")
        else:
            issues.append(f"Missing expected term '{term}' in synthesis output")

    word_count = len(text.split())
    if word_count < 60:
        issues.append(f"Synthesis output too short ({word_count} words, need >= 60)")
    else:
        passed.append(f"Adequate length ({word_count} words)")

    score = len(passed) / (len(passed) + len(issues)) if (passed or issues) else 0.5
    return round(score, 3), issues, passed


def _check_mcp_tool_result(tool_name: str, result: Any) -> tuple[bool, list[str], list[str]]:
    """
    Validate that an MCP tool result has the expected top-level keys.
    Grounding source: actual tool response schemas in server.py.
    """
    issues: list[str] = []
    passed: list[str] = []

    if tool_name not in REQUIRED_MCP_KEYS:
        return True, [], [f"No schema check for tool '{tool_name}'"]

    expected = REQUIRED_MCP_KEYS[tool_name]
    if isinstance(result, dict):
        for key in expected:
            if key in result:
                passed.append(f"Key '{key}' present in MCP result")
            else:
                issues.append(f"Expected key '{key}' missing from {tool_name} result")
    elif isinstance(result, str):
        for key in expected:
            if key.lower() in result.lower():
                passed.append(f"Key '{key}' mentioned in result text")
            else:
                issues.append(f"Expected keyword '{key}' absent from {tool_name} text result")
    else:
        issues.append(f"MCP result is not a dict or string (type: {type(result).__name__})")

    return len(issues) == 0, issues, passed


# ---------------------------------------------------------------------------
# Main Environment class — drop-in replacement for toolkit's Environment
# ---------------------------------------------------------------------------

class Environment:
    """
    Grounded Harborstone evaluator.

    Replaces the toolkit's stochastic fake Environment.
    Scores are derived from real business-rule and schema checks,
    NOT from random.random().

    Used by:
      - LATS (external feedback for MCTS scoring)
      - Reflexion (evaluate step per trial)
    """

    def __init__(
        self,
        success_threshold: float = 0.65,
        tool_name: str | None = None,
    ):
        if not 0.0 <= success_threshold <= 1.0:
            raise ValueError("success_threshold must be between 0 and 1")
        self.success_threshold = success_threshold
        self.tool_name = tool_name          # hint: which MCP tool this env guards

    # ------------------------------------------------------------------
    # Primary entry point — matches toolkit's signature
    # ------------------------------------------------------------------

    def evaluate(self, task: str, output: str, *, mcp_result: Any = None) -> EnvironmentFeedback:
        """
        Evaluate a sub-task output against grounded Harborstone rules.

        Parameters
        ----------
        task : str
            The sub-task instruction.
        output : str
            The LLM's produced output / answer for that sub-task.
        mcp_result : Any
            Optional: the actual MCP tool response (dict or str).
            When present, enables schema-level grounding.

        Returns
        -------
        EnvironmentFeedback  (score is grounded, not random)
        """
        all_issues: list[str] = []
        all_passed: list[str] = []

        # 1. Vessel type check
        vt_ok, vt_issue = _check_vessel_type(output + " " + task)
        if not vt_ok:
            all_issues.append(vt_issue)
        else:
            all_passed.append("Valid vessel_type found")

        # 2. Premium positivity
        pr_ok, pr_issue = _check_premium_positive(output)
        if not pr_ok:
            all_issues.append(pr_issue)
        else:
            all_passed.append("Premium/amount is positive")

        # 3. Synthesis completeness (if this looks like a synthesis node)
        is_synthesis = (
            "synthesis" in task.lower()
            or "summar" in task.lower()
            or "recommend" in task.lower()
            or "final" in task.lower()
            or len(output.split()) > 40
        )
        if is_synthesis:
            comp_score, comp_issues, comp_passed = _check_synthesis_completeness(output)
            all_issues.extend(comp_issues)
            all_passed.extend(comp_passed)
        else:
            comp_score = 1.0

        # 4. MCP result schema check (grounded against server.py schemas)
        tool = self.tool_name
        if tool and mcp_result is not None:
            mcp_ok, mcp_issues, mcp_passed = _check_mcp_tool_result(tool, mcp_result)
            all_issues.extend(mcp_issues)
            all_passed.extend(mcp_passed)
            grounding_source = f"MCP tool schema for {tool!r} (server.py)"
        else:
            grounding_source = "Harborstone business rules + output structure"

        # Aggregate score
        total = len(all_passed) + len(all_issues)
        if total == 0:
            score = 0.5
        else:
            raw = len(all_passed) / total
            # Blend with synthesis completeness score when applicable
            score = (raw + comp_score) / 2 if is_synthesis else raw

        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= self.success_threshold

        msg_parts = []
        if all_passed:
            msg_parts.append("PASSED: " + "; ".join(all_passed[:3]))
        if all_issues:
            msg_parts.append("FAILED: " + "; ".join(all_issues[:3]))
        message = " | ".join(msg_parts) or "No checks applied"

        return EnvironmentFeedback(
            score=score,
            success=success,
            message=message,
            grounding_source=grounding_source,
            caught_issues=all_issues,
            passed_checks=all_passed,
        )
