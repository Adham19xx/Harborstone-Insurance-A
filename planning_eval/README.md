# Member 1 evaluation

The tests cover DAG construction, cycle rejection, topological/dependency order, real-tool-node execution via an executor boundary, dynamic replanning after an early ineligible observation, and the grounded guardrail that prevents stale premium estimation after an eligibility failure.

`run_harborstone_comparison.py` is the real-request runner. It uses the existing Stdio MCP client path and the existing Harborstone database.
