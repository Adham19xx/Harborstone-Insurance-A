import pytest

from planning.models import Plan, Task
from planning.algorithms.decomposition import GeneratedPlan, PlannedTask, decompose_goal, execute_plan, final_output
from planning.integration.trace import RunTrace


class FakeStructured:
    def __init__(self, value):
        self.value = value
    def invoke(self, *_args, **_kwargs):
        return self.value


class FakeLLM:
    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def with_structured_output(self, *_args, **_kwargs):
        return FakeStructured(self.plan)

    def invoke(self, *_args, **_kwargs):
        self.calls += 1
        class Response:
            content = "Final Harborstone summary"
            usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return Response()


class FakeExecutor:
    def __init__(self):
        self.mcp_calls = 0
        self.calls = []

    async def call_tool(self, tool_name, arguments):
        self.mcp_calls += 1
        self.calls.append((tool_name, arguments))
        results = {
            "get_customer_policies": {"current_policy": {"policy_id": 1, "premium": 7500}},
            "get_policy_coverage": {"found": True, "policy": {"policy_id": 1, "premium": 7500}},
            "check_vessel_eligibility": {"eligible": False, "reasons": ["too old"]},
            "estimate_policy_premium_change": {"estimated_increment": 5000},
            "get_policy_update_requirements": {"requirements": ["registration"]},
        }
        return {"tool": tool_name, "arguments": arguments, "result": results[tool_name], "latency_ms": 0.1}


def make_plan():
    return GeneratedPlan(
        goal="policy update goal",
        tasks=[
            PlannedTask(id="t1", instruction="Retrieve current policy", depends_on=[], tool_name="get_customer_policies", arguments={"customer_id": 1}),
            PlannedTask(id="t2", instruction="Review coverage", depends_on=["t1"], tool_name="get_policy_coverage", arguments={"policy_id": "$t1.current_policy.policy_id"}),
            PlannedTask(id="t3", instruction="Check eligibility", depends_on=[], tool_name="check_vessel_eligibility", arguments={"customer_id": 1, "year_built": 2000, "value": 500000}),
            PlannedTask(id="t4", instruction="Estimate premium", depends_on=["t1", "t3"], tool_name="estimate_policy_premium_change", arguments={"policy_id": "$t1.current_policy.policy_id", "vessel_value": 500000}),
            PlannedTask(id="t5", instruction="Get requirements", depends_on=["t1"], tool_name="get_policy_update_requirements", arguments={"policy_id": "$t1.current_policy.policy_id", "vessel_type": "Yacht"}),
            PlannedTask(id="t6", instruction="Synthesize", depends_on=["t2", "t3", "t4", "t5"], kind="synthesis"),
        ],
    )


def test_dag_is_acyclic_and_topological_order_respects_dependencies():
    plan = Plan.model_validate(make_plan().model_dump())
    order = plan.topological_order()
    assert plan.graph.number_of_edges() == 8
    assert order.index("t1") < order.index("t2")
    assert order.index("t3") < order.index("t4")
    assert order.index("t4") < order.index("t6")
    assert order.index("t5") < order.index("t6")


def test_cycle_is_rejected():
    with pytest.raises(ValueError, match="Cycle detected"):
        Plan(goal="cycle", tasks=[
            Task(id="a", instruction="Task A", depends_on=["b"], tool_name="get_customer_policies"),
            Task(id="b", instruction="Task B", depends_on=["a"], tool_name="get_customer_policies"),
        ])


def test_decomposition_first_executes_all_planned_tool_nodes_in_dependency_order():
    fake_llm = FakeLLM(make_plan())
    trace = RunTrace("decomposition_first", "test", "policy update goal")
    plan = decompose_goal("policy update goal", fake_llm, {"customer_id": 1}, trace)
    executor = FakeExecutor()

    import asyncio
    outputs = asyncio.run(execute_plan(plan, fake_llm, executor, trace))

    assert final_output(plan, outputs) == "Final Harborstone summary"
    names = [name for name, _ in executor.calls]
    assert names == [
        "get_customer_policies",
        "check_vessel_eligibility",
        "get_policy_coverage",
        "estimate_policy_premium_change",
        "get_policy_update_requirements",
    ]
    assert trace.mcp_calls == 5
    assert trace.llm_calls == 2
    assert trace.total_tokens == 15
