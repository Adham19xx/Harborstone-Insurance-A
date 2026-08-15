# Harborstone Insurance — Agentic AI Platform

An experimental platform built around a fictional marine insurance company, **Harborstone Insurance**. It demonstrates how to combine several core building blocks of an AI agent system on top of a real relational database:

- A **Model Context Protocol (MCP) server** that exposes insurance data and actions safely to an LLM (tools, resources, and prompts) instead of granting direct SQL access.
- Four **RAG (Retrieval-Augmented Generation) architectures** — Naive, Hybrid (dense + BM25), Agentic (multi-hop), and Self-RAG verification — plus an evaluation harness to compare them.
- A multi-layer **agent memory system** — short-term transcript, scratchpad, episodic store, semantic store, and an LLM-driven consolidation layer.
- A **context-window optimization pipeline** (sliding window → observation masking → recursive summarization → zone pruning) with its own metrics and tests.
- An **autonomous decomposition & planning system (Week 4)** — DAG task decomposition (decomposition-first & dynamic), sub-task planning algorithms (Plan-and-Solve, Tree of Thoughts, LATS), self-correction loops (Self-Refine & Reflexion), and a real Grounded Environment.
- A **MySQL schema** modeling customers, vessels, policies, claims, and payments for a marine insurer.

## Repository Structure

```
Harborstone-Insurance-A/
├── server.py                  # MCP server (root copy, uses the official `mcp` package)
├── schema.sql                 # DB schema (root copy, duplicate of db/schema.sql)
├── seed(2).sql                 # Seed data (root copy, duplicate of db/seed(2).sql)
├── requirements.txt.txt       # Root Python dependencies
│
├── mcp_server/
│   ├── server.py               # MCP server (uses the standalone `fastmcp` package)
│   └── requirements.txt.txt
│
├── agent/
│   ├── client.py                # MCP client that drives a full demo session against the server
│   ├── agent_loop.py            # Wires Hybrid RAG + Self-RAG verification into one turn handler
│   ├── planning_agent.py        # Planning Agent coordinating DAGs, PS/ToT/LATS, and Grounded validation
│   └── requirements.txt
│
├── planning/                   # Week 4 Decomposition & Planning System
│   ├── environment.py           # Grounded Environment with real underwriting validation
│   ├── models.py                # Pydantic DAG nodes and Plan schemas with cycle detection
│   ├── algorithms/
│   │   ├── decomposition.py         # Static decomposition-first DAG planner
│   │   ├── dynamic_decomposition.py # Adaptive/interleaved dynamic DAG planner
│   │   ├── plan_and_solve.py        # 2-phase Plan-and-Solve for linear/actuarial math
│   │   ├── tree_of_thoughts.py      # BFS/DFS Tree of Thoughts for risk & tradeoff search
│   │   ├── lats.py                  # 4-phase MCTS Language Agent Tree Search with reflections
│   │   ├── router.py                # Sub-task routing logic dispatching to PS/ToT/LATS
│   │   ├── self_refine.py           # Single-draft 4-point rubric critique & revision
│   │   └── reflexion.py             # Multi-trial loop with episodic verbal reflection buffer
│   ├── integration/
│   │   ├── mcp_executor.py          # Real MCP client adapter
│   │   └── trace.py                 # JSON artifact execution trace logger
│   ├── requests/
│   │   └── harborstone_requests.py  # 15 real marine insurance request fixtures
│   └── README.md
│
├── planning_eval/              # Week 4 Planning Benchmark & Evaluation Harness
│   ├── test_decomposition.py        # DAG acyclicity & topological execution tests
│   ├── test_dynamic_decomposition.py# Dynamic replanning & divergence tests
│   ├── test_planning_algorithms.py  # Unit & integration tests for all planning algorithms
│   ├── run_full_evaluation.py       # Full 15-case benchmark harness & table generator
│   └── run_harborstone_comparison.py# Real MCP live divergence runner
│
├── artifacts/                  # JSON traces & evaluation summary artifacts
│   ├── evaluation_summary.json
│   └── full_comparison_table.md
│
├── rag/
│   ├── vector_store.py          # Chroma-backed persistent vector store
│   ├── embedding.py             # Embedding pipeline wrapper
│   ├── naive_rag.py             # Baseline: vector search + generation
│   ├── hybrid_rag.py            # Dense vector search + BM25, fused with Reciprocal Rank Fusion
│   ├── agentic_rag.py           # Multi-hop retrieval loop
│   └── self_rag_check.py        # Relevance and groundedness verification
│
├── memory/
│   ├── short_term.py            # Rolling, session-scoped conversation buffer
│   ├── scratchpad.py            # Persistent plan / sub-goals / working variables
│   ├── episodic_store.py        # JSON-persisted raw episodes
│   ├── semantic_store.py        # JSON-persisted consolidated facts
│   ├── consolidation.py         # LLM-driven extraction of facts
│   └── router.py                # Scores a memory and decides EPISODIC vs FORGET
│
├── context_eval/
│   ├── sliding_window.py
│   ├── observation_masking.py
│   ├── recursive_summary.py
│   ├── zone_pruning.py
│   ├── evaluator.py
│   ├── metrics.py
│   └── tests.py
│
├── retrieval_eval/
│   ├── evaluation.py
│   └── questions.json
│
└── db/
    ├── schema.sql                 # Customers, Employees, Vessels, Policies, Claims, Payments
    ├── seed(2).sql                 # Sample rows for the above tables
    ├── ERD.png                    # Entity-relationship diagram
    └── README.md.txt
```

---

## Week 4 — Autonomous Decomposition & Planning System

The planning agent handles real, multi-step, ambiguous marine insurance requests (e.g. adding a vessel to an existing hull policy, calculating multi-tier premium adjustments, resolving underwriting constraints, and structuring compliant policyholder notifications).

### Planning Method Comparison Table (15 Real Marine Insurance Cases)

| Planning Concern / Method | Task Success | Avg LLM / Tool Calls | Avg Tokens | Avg Latency | Est Cost/Run | Production Deployment Choice & Justification |
|---|---|---|---|---|---|---|
| **Decomposition-First** | 12/15 (80.0%) | 5.0 | 6,335 | 0.28s | $0.038 | Shipped for static, predictable batch tasks without runtime branching. |
| **Dynamic Decomposition** | **15/15 (100.0%)** | 5.6 | 7,913 | 0.46s | $0.059 | **Shipped as Default Top-Level Planner**: Dynamically reroutes when early tool calls reveal vessel ineligibility. |
| **Plan-and-Solve (PS)** | **15/15 (100.0%)** | 1.0 | 1,450 | 0.12s | $0.010 | **Shipped for Math/Actuarial Sub-Tasks**: Fast, non-branching linear execution for exact premium formulas. |
| **Tree of Thoughts (ToT)** | **15/15 (100.0%)** | 16.0 | 5,400 | 0.39s | $0.038 | **Shipped for Risk Ranking/Tradeoffs**: BFS/DFS search explores candidate deductible and coverage tradeoffs. |
| **LATS (Ungrounded Env)** | 12/15 (80.0%) | 8.0 | 6,900 | 0.52s | $0.052 | *Ablation Baseline*: Demonstrates that ungrounded self-evaluation misses critical underwriting violations. |
| **LATS (Grounded Env)** | **12/15 (80.0%)** | 10.0 | 8,100 | 0.64s | $0.065 | **Shipped for High-Stakes Endorsements**: MCTS exploration guided by real underwriting rules & verbal reflections. |
| **Self-Refine** | **15/15 (100.0%)** | 3.0 | 3,100 | 0.24s | $0.022 | **Shipped for Customer Synthesis**: 1-draft rubric critique for polished policyholder notifications. |
| **Reflexion (Multi-Trial)** | **12/15 (80.0%)** | 4.4 | 5,720 | 0.58s | $0.042 | **Shipped for Complex Multi-Constraint Restructuring**: Episodic memory carries reflections across trials to fix mistakes. |

### Grounded vs. Ungrounded Validation
The Grounded Environment (`planning/environment.py`) replaces superficial LLM self-critique with real actuarial and database validation. In our benchmark contrast:
- **Ungrounded Critic**: Accepts an invalid proposal for an over-age ($750k, 31-year-old) vessel with missing survey as valid (`score = 1.0`).
- **Grounded Engine**: Immediately flags 3 critical violations (Age > 20 years, Missing Independent Marine Surveyor Appraisal, Below-minimum Deductible), rejecting the plan with `score = 0.0`.

---

## Prerequisites & Installation

- Python 3.10+
- MySQL server (via **XAMPP** or standalone)
- pip

```bash
git clone https://github.com/Adham19xx/Harborstone-Insurance-A.git
cd Harborstone-Insurance-A

pip install -r requirements.txt.txt
pip install -r agent/requirements.txt
pip install -r planning/requirements.txt
pip install fastmcp chromadb rank_bm25 tabulate pytest pytest-asyncio networkx
```

## Running the Planning System

1. **Run all automated tests (16/16 pass)**:
   ```bash
   python -m pytest planning_eval -v
   ```

2. **Run full 15-case benchmark comparison**:
   ```bash
   python planning_eval/run_full_evaluation.py
   ```

3. **Run Planning Agent on live MCP server**:
   ```bash
   python planning_eval/run_harborstone_comparison.py
   ```

---

## MCP Server Capabilities

**Resources** (read-only data)
- `harborstone://policies/terms/marine-hull` — static marine hull policy terms and conditions.
- `harborstone://claims/pending-summary` — live query of all claims with `status = 'Pending'`.

**Prompts**
- `draft_claim_investigation(claim_id)` — generates a structured prompt for investigating a specific claim.

**Tools**
- `audit_high_risk_policies` — audits policies with `premium > $8,000`, reporting progress at 20% / 60% / 100%.
- `approve_claim(claim_id, user_role)` — approves a pending claim with role authorization and human-in-the-loop elicitation.
- `switch_user_role(new_role)` — changes the active session role.
- `get_customer_policies(customer_id)` — retrieves customer policy list.
- `check_vessel_eligibility(vessel_type, year_built, value)` — validates vessel underwriting rules.
- `estimate_policy_premium_change(...)` — computes incremental premium.
- `get_policy_update_requirements(...)` — returns required documentation checklist.

---

## License

MIT License.