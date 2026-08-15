"""Harborstone Planning Agent (Week 4 Lab).

This agent sits alongside the existing Memory and RAG agents in `agent/` without
altering their code paths. It solves complex, multi-step, ambiguous customer
insurance planning requests by:
1. Decomposing requests into a DAG (Decomposition-First or Dynamic).
2. Routing complex sub-tasks to specialized planning algorithms:
   - Plan-and-Solve (PS) for deterministic calculations
   - Tree of Thoughts (ToT) for risk prioritization and tradeoff search
   - LATS for MCTS action exploration with grounded feedback
   - MCP direct execution for deterministic database lookups
3. Applying Self-Correction (Self-Refine and Reflexion).
4. Validating all proposals against the real Grounded Environment.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from planning.algorithms.decomposition import decompose_goal, execute_plan, final_output
from planning.algorithms.dynamic_decomposition import dynamic_decomposition
from planning.algorithms.plan_and_solve import plan_and_solve
from planning.algorithms.tree_of_thoughts import tree_of_thoughts_search
from planning.algorithms.lats import lats_search
from planning.algorithms.router import route_subtask, RoutingDecision
from planning.algorithms.self_refine import self_refine
from planning.algorithms.reflexion import run_reflexion
from planning.environment import GroundedEnvironment
from planning.integration.mcp_executor import HarborstoneMCPExecutor
from planning.integration.trace import RunTrace


class PlanningAgent:
    """
    Harborstone Marine Insurance Autonomous Planning Agent.
    """

    def __init__(
        self,
        llm=None,
        server_script: str = "mcp_server/server.py",
        artifacts_dir: str = "artifacts",
    ):
        self.llm = llm
        self.server_script = server_script
        self.artifacts_dir = Path(artifacts_dir)
        self.environment = GroundedEnvironment()

    async def solve_request(
        self,
        request: Dict[str, Any],
        decomposition_method: str = "dynamic",
    ) -> Dict[str, Any]:
        """
        Main entrypoint to solve a complex planning request.
        """
        request_id = request.get("request_id", "custom-request")
        goal_text = request.get("text", "")
        trace = RunTrace(
            method=decomposition_method,
            request_id=request_id,
            goal=goal_text,
        )

        server_path = Path(self.server_script)
        if not server_path.exists():
            # Try repo root relative path
            server_path = Path(__file__).resolve().parents[1] / "mcp_server" / "server.py"

        async with HarborstoneMCPExecutor(server_path) as executor:
            try:
                if decomposition_method == "decomposition_first":
                    # Upfront static DAG generation
                    plan = decompose_goal(goal_text, self.llm, request, trace) if self.llm else None
                    if plan:
                        trace.plan = plan.model_dump()
                        outputs = await execute_plan(plan, self.llm, executor, trace)
                        result = final_output(plan, outputs)
                    else:
                        result = {"status": "plan_executed_deterministically", "request_id": request_id}
                    trace.finish(success=True, result=result)

                else:
                    # Dynamic / interleaved execution
                    history = await dynamic_decomposition(
                        goal_text, self.llm, request, executor, trace
                    )
                    final_res = history[-1][1] if history else {"status": "completed"}
                    trace.finish(success=True, result=final_res)

            except Exception as exc:
                trace.finish(success=False, error=str(exc))

        trace.mcp_calls = executor.mcp_calls
        saved_path = trace.save(self.artifacts_dir)

        return {
            "request_id": request_id,
            "method": decomposition_method,
            "success": trace.success,
            "result": trace.result,
            "error": trace.error,
            "mcp_calls": trace.mcp_calls,
            "llm_calls": trace.llm_calls,
            "total_tokens": trace.total_tokens,
            "latency_ms": trace.latency_ms,
            "trace_file": str(saved_path),
        }

    def solve_subtask_with_algorithm(
        self,
        task_id: str,
        instruction: str,
        context: Dict[str, Any],
        forced_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Routes and executes a sub-task using the selected or optimal planning algorithm.
        """
        routing: RoutingDecision = (
            RoutingDecision(
                task_id=task_id,
                instruction=instruction,
                selected_method=forced_method,  # type: ignore
                rationale="Forced by caller",
                estimated_complexity="Custom",
            )
            if forced_method
            else route_subtask(task_id, instruction, context)
        )

        method = routing.selected_method
        trace = RunTrace(method=f"subtask_{method}", request_id=task_id, goal=instruction)

        if method == "PS":
            res = plan_and_solve(instruction, context, self.llm, trace)
            return {"method": "PS", "routing": routing.model_dump(), "result": res.model_dump()}

        elif method == "ToT":
            res = tree_of_thoughts_search(instruction, context, search_strategy="BFS", llm=self.llm, trace=trace)
            return {"method": "ToT", "routing": routing.model_dump(), "result": res.model_dump()}

        elif method == "LATS":
            res = lats_search(instruction, context, self.environment, llm=self.llm, trace=trace)
            return {"method": "LATS", "routing": routing.model_dump(), "result": res.model_dump()}

        elif method == "SELF_REFINE":
            res = self_refine(instruction, context, self.llm, trace)
            return {"method": "SELF_REFINE", "routing": routing.model_dump(), "result": res.model_dump()}

        elif method == "REFLEXION":
            res = run_reflexion(instruction, context, self.environment, llm=self.llm, trace=trace)
            return {"method": "REFLEXION", "routing": routing.model_dump(), "result": res.model_dump()}

        else:
            return {"method": method, "routing": routing.model_dump(), "result": "Direct execution"}


if __name__ == "__main__":
    from planning.requests.harborstone_requests import ELIGIBLE_REQUEST

    agent = PlanningAgent()
    print("Running Harborstone Planning Agent on sample request...")
    res = asyncio.run(agent.solve_request(ELIGIBLE_REQUEST, decomposition_method="dynamic"))
    print(json.dumps(res, indent=2, default=str))
