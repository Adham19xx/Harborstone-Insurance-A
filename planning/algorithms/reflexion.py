"""Reflexion Multi-Trial Reinforcement Planning for Harborstone Insurance.

Based on Shinn et al. (2023): 'Reflexion: Language Agents with Verbal Reinforcement Learning'.

Reflexion addresses hard multi-step problems where a single shot or single retry
is insufficient. It executes multiple trials, evaluating each attempt with
the Grounded Environment, generating verbal self-reflections upon failure,
and carrying an episodic memory buffer of past reflections into subsequent trials.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, Field

from ..environment import GroundedEnvironment, EnvironmentFeedback
from ..integration.trace import RunTrace

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class TrialRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trial_number: int
    proposed_action: Dict[str, Any]
    feedback: EnvironmentFeedback
    verbal_reflection: Optional[str] = None


class ReflexionMemory(BaseModel):
    """Capped episodic memory buffer for verbal reflections."""
    model_config = ConfigDict(extra="forbid")
    max_reflections: int = 5
    reflections: List[str] = Field(default_factory=list)

    def add_reflection(self, reflection: str) -> None:
        self.reflections.append(reflection)
        if len(self.reflections) > self.max_reflections:
            self.reflections.pop(0)  # Maintain capped buffer size


class ReflexionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_goal: str
    trials_attempted: int
    success: bool
    final_score: float
    final_solution: Dict[str, Any]
    episodic_memory: List[str]
    trials: List[TrialRecord]


def run_reflexion(
    task_goal: str,
    initial_request: Dict[str, Any],
    environment: Optional[GroundedEnvironment] = None,
    max_trials: int = 4,
    llm: Optional[BaseChatModel] = None,
    trace: Optional[RunTrace] = None,
) -> ReflexionResult:
    """
    Execute Reflexion multi-trial loop with grounded environment feedback and verbal memory.
    """
    env = environment if environment is not None else GroundedEnvironment()
    memory = ReflexionMemory(max_reflections=5)
    trials: List[TrialRecord] = []

    current_context = dict(initial_request)
    success = False
    final_score = 0.0
    final_solution = {}

    for trial_idx in range(1, max_trials + 1):
        # 1. Action Generation (conditioned on prior episodic reflections)
        if llm is not None:
            action = _generate_action_with_llm(task_goal, current_context, memory, llm, trace)
        else:
            action = _generate_action_deterministic(current_context, memory.reflections, trial_idx)

        # 2. Grounded Environment Evaluation
        feedback = env.evaluate_proposal(action)

        # 3. Reflection Generation on Failure
        reflection: Optional[str] = None
        if not feedback.success:
            reflection = (
                f"[Trial #{trial_idx} Failed]: Underwriting violations encountered: {'; '.join(feedback.violations)}. "
                f"For the next attempt, explicitly fix: {', '.join(feedback.violations)}."
            )
            memory.add_reflection(reflection)
        else:
            success = True
            final_score = feedback.score
            final_solution = action

        record = TrialRecord(
            trial_number=trial_idx,
            proposed_action=action,
            feedback=feedback,
            verbal_reflection=reflection,
        )
        trials.append(record)

        if success:
            break

    if not success and trials:
        final_score = trials[-1].feedback.score
        final_solution = trials[-1].proposed_action

    res = ReflexionResult(
        task_goal=task_goal,
        trials_attempted=len(trials),
        success=success,
        final_score=final_score,
        final_solution=final_solution,
        episodic_memory=memory.reflections,
        trials=trials,
    )

    if trace is not None:
        trace.execution.append({
            "method": "Reflexion",
            "trials": len(trials),
            "success": success,
            "reflections_count": len(memory.reflections),
        })

    return res


def _generate_action_with_llm(
    goal: str,
    context: Dict[str, Any],
    memory: ReflexionMemory,
    llm: BaseChatModel,
    trace: Optional[RunTrace],
) -> Dict[str, Any]:
    prompt = f"""Goal: {goal}
Context: {json.dumps(context, default=str)}

Episodic Memory of Prior Reflections from Failed Attempts:
{chr(10).join(f"- {r}" for r in memory.reflections) if memory.reflections else "None (First Trial)"}

Propose a valid policy endorsement structure that avoids all past mistakes."""

    class ActionProposal(BaseModel):
        vessel_type: str
        year_built: int
        vessel_value: float
        current_premium: float
        proposed_premium: float
        deductible: float
        documents: List[str]

    try:
        res = llm.with_structured_output(ActionProposal, method="json_schema").invoke([("human", prompt)])
        if trace is not None:
            trace.add_llm_usage(res)
        return res.model_dump()
    except Exception:
        return _generate_action_deterministic(context, memory.reflections, len(memory.reflections) + 1)


def _generate_action_deterministic(
    context: Dict[str, Any], reflections: List[str], trial_idx: int
) -> Dict[str, Any]:
    """
    Demonstrates realistic progression where the agent learns from grounded reflections:
    - Trial 1: naive proposal (e.g. wrong deductible or missing required survey for >$500k yacht).
    - Trial 2+: incorporates reflection to add survey and fix deductible/rate.
    """
    vessel_type = context.get("vessel_type", context.get("new_vessel", {}).get("vessel_type", "Yacht"))
    vessel_value = float(context.get("vessel_value", context.get("new_vessel", {}).get("value", 600000.0)))
    year_built = int(context.get("year_built", context.get("new_vessel", {}).get("year_built", 2024)))
    current_premium = float(context.get("current_premium", 1500.0))

    rate = 0.015 if vessel_type == "Yacht" else 0.010
    correct_premium = round(current_premium + (vessel_value * rate), 2)

    has_survey_reflection = any("survey" in r.lower() or "appraisal" in r.lower() or "500,000" in r for r in reflections)
    has_deductible_reflection = any("deductible" in r.lower() for r in reflections)

    # Initial naive proposal (omits survey and has low deductible)
    if trial_idx == 1 and not reflections:
        return {
            "strategy": "Initial Naive Proposal",
            "vessel_type": vessel_type,
            "year_built": year_built,
            "vessel_value": vessel_value,
            "current_premium": current_premium,
            "proposed_premium": correct_premium,
            "deductible": 200.0,  # Below minimum
            "documents": ["Proof of purchase"],  # Missing survey
        }

    # Second trial fixing deductible if flagged
    if trial_idx == 2 and not has_survey_reflection:
        return {
            "strategy": "Adjusted Deductible Proposal",
            "vessel_type": vessel_type,
            "year_built": year_built,
            "vessel_value": vessel_value,
            "current_premium": current_premium,
            "proposed_premium": correct_premium,
            "deductible": 5000.0,  # Compliant deductible
            "documents": ["Proof of purchase", "Registration"],  # Still missing survey
        }

    # Final trial fully incorporating episodic verbal reflections
    return {
        "strategy": "Fully Grounded Reflexion Solution",
        "vessel_type": vessel_type,
        "year_built": year_built,
        "vessel_value": vessel_value,
        "current_premium": current_premium,
        "proposed_premium": correct_premium,
        "deductible": round(vessel_value * 0.05, 2),
        "documents": [
            "Proof of purchase",
            "Current vessel registration",
            "Recent independent marine surveyor appraisal report",
        ],
    }
