import pytest

from planning.algorithms.dynamic_decomposition import DynamicDecision, dynamic_decomposition
from planning.integration.trace import RunTrace


class FakeStructured:
    def __init__(self, values):
        self.values = values
        self.index = 0
    def invoke(self, *_args, **_kwargs):
        value = self.values[self.index]
        self.index += 1
        return value


class FakeResponse:
    usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


class FakeLLM:
    def __init__(self):
        self.decisions = [
            DynamicDecision(done=False, task_id="t1", tool_name="get_customer_policies"),
            DynamicDecision(done=False, task_id="t2", tool_name="check_vessel_eligibility"),
            DynamicDecision(done=False, task_id="t3", tool_name="get_policy_update_requirements"),
            DynamicDecision(done=True),
        ]
        self.structured = FakeStructured(self.decisions)
    def with_structured_output(self, *_args, **_kwargs):
        return self.structured


class FakeExecutor:
    def __init__(self):
        self.mcp_calls = 0
        self.calls = []
    async def call_tool(self, tool_name, arguments):
        self.mcp_calls += 1
        self.calls.append(tool_name)
        if tool_name == "get_customer_policies":
            result = {"current_policy": {"policy_id": 1}}
        elif tool_name == "check_vessel_eligibility":
            result = {"eligible": False, "reasons": ["too old"]}
        else:
            result = {"requirements": ["registration"]}
        return {"tool": tool_name, "arguments": arguments, "result": result, "latency_ms": 0.1}


@pytest.mark.asyncio
async def test_dynamic_changes_course_after_early_ineligible_result():
    llm = FakeLLM()
    executor = FakeExecutor()
    trace = RunTrace("dynamic", "test", "policy update goal")
    history = await dynamic_decomposition(
        "policy update goal",
        llm,
        {"customer_id": 1, "new_vessel": {"year_built": 2000, "value": 500000, "vessel_type": "Yacht"}},
        executor,
        trace,
    )
    assert [tool for tool in executor.calls] == [
        "get_customer_policies",
        "check_vessel_eligibility",
        "get_policy_update_requirements",
    ]
    assert "estimate_policy_premium_change" not in executor.calls
    assert history[-1][0] == "t3"


@pytest.mark.asyncio
async def test_dynamic_guardrail_replaces_premium_after_ineligible_observation():
    class BadPlanner(FakeLLM):
        def __init__(self):
            self.decisions = [
                DynamicDecision(done=False, task_id="t1", tool_name="get_customer_policies"),
                DynamicDecision(done=False, task_id="t2", tool_name="check_vessel_eligibility"),
                DynamicDecision(done=False, task_id="t3", tool_name="estimate_policy_premium_change", arguments={"policy_id": 1, "vessel_value": 500000}),
                DynamicDecision(done=True),
            ]
            self.structured = FakeStructured(self.decisions)
    llm = BadPlanner()
    executor = FakeExecutor()
    trace = RunTrace("dynamic", "guardrail", "policy update goal")
    await dynamic_decomposition(
        "policy update goal",
        llm,
        {"customer_id": 1, "new_vessel": {"year_built": 2000, "value": 500000, "vessel_type": "Yacht"}},
        executor,
        trace,
    )
    assert "estimate_policy_premium_change" not in executor.calls
    assert any(change["reason"] == "early ineligible result" for change in trace.plan_changes)
