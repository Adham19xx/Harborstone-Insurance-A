from __future__ import annotations

import asyncio
import json
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ..models import Plan
from ..integration.trace import RunTrace

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


PLANNER_SYSTEM = """You are the Harborstone Insurance task-decomposition planner.

Your job is to convert the supplied real Harborstone customer request into a
small executable DAG.

Rules:
1. Produce 3-6 operational MCP tasks plus exactly one final synthesis task.
2. Every operational task must call exactly one allowed Harborstone MCP tool.
3. Use dependencies whenever a task needs information produced by an earlier task.
4. Do not invent database fields, tools, or arguments.
5. The MCP server is the source of truth for tool signatures.
6. End with exactly one synthesis node.
7. Preserve the supplied customer goal exactly.
8. For the selected marine policy-update request, use the supplied request data
   for customer_id and new_vessel information.
"""


class PlannedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    instruction: str
    depends_on: list[str]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    kind: str = "mcp"


class GeneratedPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str
    tasks: list[PlannedTask]


def decompose_goal(
    goal: str,
    llm: BaseChatModel,
    request: dict[str, Any],
    trace: RunTrace | None = None,
) -> Plan:
    """
    Decompose the real Harborstone request into an executable DAG.

    The planner is explicitly grounded in the actual MCP tool signatures used
    by the Harborstone server.
    """

    generated = llm.with_structured_output(
        GeneratedPlan,
        method="json_schema",
    ).invoke(
        [
            ("system", PLANNER_SYSTEM),
            (
                "human",
                f"""
Decompose this Harborstone request into an executable dependency-aware DAG.

Request:
{goal!r}

Request data:
{json.dumps(request, ensure_ascii=False)}

The actual available Harborstone MCP tools and signatures are:

1. get_customer_policies(customer_id: int)

2. get_vessel(vessel_id: int)

3. check_vessel_eligibility(
       vessel_type: str,
       year_built: int,
       value: float
   )

4. estimate_policy_premium_change(
       current_premium: float,
       vessel_type: str,
       vessel_value: float
   )

5. get_policy_update_requirements(
       vessel_type: str,
       vessel_value: float
   )

The customer request should be decomposed so that:

- First retrieve the customer's policies using request.customer_id.
- Use the policy result to identify the relevant current policy and its
  current premium when needed.
- Check the new vessel's eligibility using the literal new_vessel fields
  from the request.
- Retrieve coverage information only if the corresponding Harborstone
  coverage tool exists in the available server tools.
- Estimate premium change using current_premium from the retrieved policy
  and the new vessel's type and value.
- Retrieve required policy-update documents using vessel_type and vessel_value.
- End with exactly one synthesis task.

Important argument rules:

get_customer_policies:
    customer_id = request["customer_id"]

check_vessel_eligibility:
    vessel_type = request["new_vessel"]["vessel_type"]
    year_built = request["new_vessel"]["year_built"]
    value = request["new_vessel"]["value"]

estimate_policy_premium_change:
    current_premium must come from the relevant policy returned by
    get_customer_policies.
    vessel_type = request["new_vessel"]["vessel_type"]
    vessel_value = request["new_vessel"]["value"]

get_policy_update_requirements:
    vessel_type = request["new_vessel"]["vessel_type"]
    vessel_value = request["new_vessel"]["value"]

Use references such as:
$t1[0].policy_id
$t1[0].premium

when a later task depends on an earlier task's result.

Do not pass policy_id to tools that do not accept policy_id.

The synthesis task must have:
    kind = "synthesis"
    tool_name = null

Return exactly one terminal synthesis task.
""",
            ),
        ],
        temperature=0.1,
    )

    if trace is not None:
        trace.add_llm_usage(generated)

    payload = generated.model_dump()
    payload["goal"] = goal

    plan = Plan.model_validate(payload)

    terminals = plan.terminal_tasks()

    if len(terminals) != 1:
        raise ValueError(
            f"Expected exactly one terminal synthesis task, found {terminals}"
        )

    terminal = plan.task(terminals[0])

    if terminal.kind != "synthesis":
        raise ValueError(
            "The sole terminal task must be the synthesis task"
        )

    return plan


def _resolve(value: Any, outputs: dict[str, Any]) -> Any:
    """
    Resolve references such as:

        $t1[0].policy_id
        $t1[0].premium

    from previously completed task outputs.
    """

    if isinstance(value, str) and value.startswith("$"):
        expression = value[1:]

        # Split task reference from the remaining path.
        if "." in expression:
            root, remainder = expression.split(".", 1)
        else:
            root, remainder = expression, ""

        current: Any = outputs.get(root)

        if current is None:
            raise KeyError(
                f"Cannot resolve reference {value!r}: "
                f"task {root!r} has no output."
            )

        # Support simple list indexing such as t1[0].
        if remainder:
            parts = remainder.split(".")

            for part in parts:
                if "[" in part and part.endswith("]"):
                    name, index_text = part.split("[", 1)
                    index = int(index_text[:-1])

                    if name:
                        if isinstance(current, dict):
                            current = current[name]
                        else:
                            current = getattr(current, name)

                    current = current[index]

                else:
                    if isinstance(current, dict):
                        current = current[part]
                    else:
                        current = getattr(current, part)

        return current

    if isinstance(value, dict):
        return {
            key: _resolve(item, outputs)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _resolve(item, outputs)
            for item in value
        ]

    return value


def _run_mcp_task(task, outputs, executor, trace):
    """
    Kept for compatibility with the existing planning package.

    Production execution uses execute_plan(), which awaits the async MCP
    executor correctly.
    """

    arguments = _resolve(task.arguments, outputs)

    return arguments, executor.call_tool(
        task.tool_name,
        arguments,
    )


async def execute_plan(
    plan: Plan,
    llm: BaseChatModel,
    executor,
    trace: RunTrace,
) -> dict[str, Any]:
    """
    Execute the DAG in dependency-safe topological batches.

    Independent MCP tasks in the same batch may execute concurrently.
    """

    outputs: dict[str, Any] = {}

    for batch in plan.execution_batches():

        mcp_tasks = [
            plan.task(task_id)
            for task_id in batch
            if plan.task(task_id).kind == "mcp"
        ]

        synthesis_tasks = [
            plan.task(task_id)
            for task_id in batch
            if plan.task(task_id).kind == "synthesis"
        ]

        async def run_one(task):
            arguments = _resolve(task.arguments, outputs)

            call = await executor.call_tool(
                task.tool_name,
                arguments,
            )

            trace.mcp_calls = executor.mcp_calls

            trace.execution.append(
                {
                    "task_id": task.id,
                    "tool": task.tool_name,
                    "arguments": arguments,
                    "result": call["result"],
                    "latency_ms": call["latency_ms"],
                }
            )

            return task.id, call["result"]

        if mcp_tasks:
            results = await asyncio.gather(
                *(run_one(task) for task in mcp_tasks)
            )

            for task_id, result in results:
                outputs[task_id] = result

        for task in synthesis_tasks:

            context = "\n\n".join(
                f"{dep}: "
                f"{json.dumps(outputs[dep], ensure_ascii=False, default=str)}"
                for dep in task.depends_on
            )

            response = llm.invoke(
                [
                    (
                        "system",
                        """
You are Harborstone Insurance's final response synthesizer.

Use only the MCP observations supplied to you.

Provide:
1. Eligibility result.
2. Current coverage information if available.
3. Expected premium impact if available.
4. Required documents/information.
5. Clear next steps.

Do not invent facts that are not present in the observations.
""",
                    ),
                    (
                        "human",
                        f"""
Goal:
{plan.goal}

MCP observations:
{context}

Write the final customer-facing summary.
""",
                    ),
                ],
                temperature=0.2,
            )

            trace.add_llm_usage(response)

            outputs[task.id] = response.content.strip()

            trace.execution.append(
                {
                    "task_id": task.id,
                    "kind": "synthesis",
                    "result": outputs[task.id],
                }
            )

    return outputs


def final_output(
    plan: Plan,
    outputs: dict[str, Any],
) -> str:
    terminals = plan.terminal_tasks()

    if len(terminals) != 1:
        raise ValueError(
            f"Expected exactly one terminal synthesis task, found {terminals}"
        )

    return str(outputs[terminals[0]])