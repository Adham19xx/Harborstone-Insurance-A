from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ..integration.trace import RunTrace

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class DynamicDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool
    task_id: str = ""
    instruction: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


DYNAMIC_SYSTEM = """
You are Harborstone Insurance's adaptive task planner.

You execute a real customer request one MCP task at a time.

The plan is allowed to change after an early observation.

Available Harborstone MCP tools:

- get_customer_policies(customer_id: int)

- get_vessel(vessel_id: int)

- check_vessel_eligibility(
    vessel_type: str,
    year_built: int,
    value: float
  )

- estimate_policy_premium_change(
    current_premium: float,
    vessel_type: str,
    vessel_value: float
  )

- get_policy_update_requirements(
    vessel_type: str,
    vessel_value: float
  )

Rules:

1. Never invent a tool.
2. Never pass arguments that a tool does not accept.
3. Use request data for customer_id and new_vessel fields.
4. Use observations from previous tools for policy-specific information.
5. If eligibility is false, do NOT estimate premium.
6. If eligibility is false, change course and gather update requirements,
   then finish with a customer summary.
7. Set done=true only when the request is sufficiently resolved.
"""


async def dynamic_decomposition(
    goal: str,
    llm: BaseChatModel,
    request: dict[str, Any],
    executor,
    trace: RunTrace,
    max_steps: int = 8,
) -> list[tuple[str, Any]]:

    history: list[tuple[str, Any]] = []

    completed: set[str] = set()

    previous_next: str | None = None

    for step in range(max_steps):

        observation = (
            "\n".join(
                f"{task}: "
                f"{json.dumps(result, ensure_ascii=False, default=str)}"
                for task, result in history
            )
            or "None"
        )

        decision = llm.with_structured_output(
            DynamicDecision,
            method="json_schema",
        ).invoke(
            [
                ("system", DYNAMIC_SYSTEM),
                (
                    "human",
                    f"""
Goal:
{goal}

Request data:
{json.dumps(request, ensure_ascii=False)}

Completed tasks:
{sorted(completed)}

Observations:
{observation}

Choose exactly one next MCP task.

If the request is fully resolved:
    done=true

Otherwise:
    done=false

For get_customer_policies:
    use request["customer_id"]

For check_vessel_eligibility:
    use:
        vessel_type = request["new_vessel"]["vessel_type"]
        year_built = request["new_vessel"]["year_built"]
        value = request["new_vessel"]["value"]

For estimate_policy_premium_change:
    use the current premium from the retrieved policy observation.
    Also use:
        vessel_type = request["new_vessel"]["vessel_type"]
        vessel_value = request["new_vessel"]["value"]

For get_policy_update_requirements:
    use:
        vessel_type = request["new_vessel"]["vessel_type"]
        vessel_value = request["new_vessel"]["value"]

Do not pass customer_id to check_vessel_eligibility.

Do not pass policy_id to estimate_policy_premium_change.

Do not pass policy_id to get_policy_update_requirements.
""",
                ),
            ],
            temperature=0.1,
        )

        trace.add_llm_usage(decision)

        if decision.done:
            break

        task_id = decision.task_id.strip()
        tool_name = decision.tool_name.strip()

        if not task_id or not tool_name:
            raise ValueError(
                f"Dynamic planner omitted task_id/tool_name "
                f"at step {step + 1}"
            )

        if task_id in completed:
            raise ValueError(
                f"Dynamic planner repeated task {task_id}"
            )

        # ---------------------------------------------------------
        # Required Week 4 dynamic guardrail:
        # eligibility failure changes the execution course.
        # ---------------------------------------------------------

        ineligible = any(
            isinstance(result, dict)
            and result.get("eligible") is False
            for _, result in history
        )

        if (
            ineligible
            and tool_name == "estimate_policy_premium_change"
        ):
            trace.plan_changes.append(
                {
                    "step": step + 1,
                    "reason": "early ineligible result",
                    "requested_next": task_id,
                    "replacement": "get_policy_update_requirements",
                }
            )

            task_id = (
                f"requirements_after_ineligible_{step + 1}"
            )

            tool_name = "get_policy_update_requirements"

            decision.arguments = {
                "vessel_type": request["new_vessel"]["vessel_type"],
                "vessel_value": request["new_vessel"]["value"],
            }

        elif (
            previous_next is not None
            and previous_next != task_id
        ):
            trace.plan_changes.append(
                {
                    "step": step + 1,
                    "from": previous_next,
                    "to": task_id,
                    "reason": decision.reason,
                }
            )

        # ---------------------------------------------------------
        # Ground arguments from the real request / observations.
        # ---------------------------------------------------------

        if tool_name == "get_customer_policies":

            arguments = {
                "customer_id": request["customer_id"]
            }

        elif tool_name == "check_vessel_eligibility":

            arguments = {
                "vessel_type": request["new_vessel"]["vessel_type"],
                "year_built": request["new_vessel"]["year_built"],
                "value": request["new_vessel"]["value"],
            }

        elif tool_name == "estimate_policy_premium_change":

            current_premium = None

            for _, result in history:
                if isinstance(result, list):
                    for policy in result:
                        if (
                            isinstance(policy, dict)
                            and "premium" in policy
                        ):
                            current_premium = float(
                                policy["premium"]
                            )
                            break

                elif isinstance(result, dict):
                    if "premium" in result:
                        current_premium = float(
                            result["premium"]
                        )

                if current_premium is not None:
                    break

            if current_premium is None:
                raise ValueError(
                    "Cannot estimate premium change before retrieving "
                    "the current policy premium."
                )

            arguments = {
                "current_premium": current_premium,
                "vessel_type": request["new_vessel"]["vessel_type"],
                "vessel_value": request["new_vessel"]["value"],
            }

        elif tool_name == "get_policy_update_requirements":

            arguments = {
                "vessel_type": request["new_vessel"]["vessel_type"],
                "vessel_value": request["new_vessel"]["value"],
            }

        elif tool_name == "get_vessel":

            arguments = decision.arguments

            if "vessel_id" not in arguments:

                vessel_id = None

                for _, result in history:
                    if isinstance(result, list):
                        for policy in result:
                            if (
                                isinstance(policy, dict)
                                and policy.get("vessel_id")
                            ):
                                vessel_id = policy["vessel_id"]
                                break

                    if vessel_id is not None:
                        break

                if vessel_id is None:
                    raise ValueError(
                        "get_vessel requires a vessel_id from a "
                        "previous Harborstone observation."
                    )

                arguments["vessel_id"] = vessel_id

        else:

            raise ValueError(
                f"Dynamic planner selected unsupported MCP tool: "
                f"{tool_name}"
            )

        # ---------------------------------------------------------
        # Execute the selected MCP task.
        # ---------------------------------------------------------

        call = await executor.call_tool(
            tool_name,
            arguments,
        )

        trace.mcp_calls = executor.mcp_calls

        result = call["result"]

        history.append(
            (
                task_id,
                result,
            )
        )

        trace.observations.append(
            {
                "step": step + 1,
                "task_id": task_id,
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "latency_ms": call["latency_ms"],
            }
        )

        completed.add(task_id)

        previous_next = task_id

    return history