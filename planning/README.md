# Harborstone Week 4 — Member 1: Decomposition / DAG

This folder is a genuine adaptation of the required reference toolkit:
`AmrSheta22/task_decomposition_and_planning`.

Member 1 owns only the task-decomposition concern:
- decomposition-first planning;
- dynamic/interleaved planning;
- DAG construction and validation;
- cycle rejection at construction time;
- dependency-safe topological execution;
- the real early-result divergence case;
- routing operational nodes to the existing Harborstone MCP tools;
- trace/metrics evidence for these two methods.

The existing Harborstone `mcp_server/` and `db/` are reused. The planner never opens the DB directly.

The source repository's database is Marine Insurance (Customers, Vessels, Policies, Claims, Payments),
so the evaluation request is the marine equivalent of the selected policy-update request: add a newly
purchased vessel to an existing policy. No Auto schema is invented and no DB is rebuilt.

## Reference-toolkit adaptation

`models.py`, `algorithms/decomposition.py`, and `algorithms/dynamic_decomposition.py` preserve the
reference toolkit's Pydantic + NetworkX DAG approach and structured-output planning, then extend the
node schema with Harborstone MCP tool metadata and grounded observations.

The toolkit's JSON artifact idea is reused by `planning/integration/trace.py`; it is not a second
unrelated logging system.

## Run tests

```powershell
python -m pytest planning_eval -v
```

## Real MCP/DB comparison

Start XAMPP MySQL and ensure the existing `harborstone_insurance` database is loaded. Install the
existing project requirements plus the toolkit's Mistral/LangChain dependencies, then set
`MISTRAL_API_KEY` in `.env`.

```powershell
python planning_eval/run_harborstone_comparison.py
```

The default case is deliberately ineligible. This makes the divergence observable: the upfront DAG
contains premium estimation, while the dynamic method observes the real eligibility failure and changes
course to requirements instead of executing the stale premium task.

A second same-type case is available:

```powershell
python planning_eval/run_harborstone_comparison.py --request-id marine-policy-update-eligible-vessel
```

Each run writes JSON traces under `artifacts/` with LLM calls, token usage when the provider reports it,
MCP calls, latency, success, execution/observations, plan changes, and outputs.
