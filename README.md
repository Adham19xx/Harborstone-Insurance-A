# Harborstone-Insurance-A# Harborstone Insurance — Agentic AI Platform

An experimental platform built around a fictional marine insurance company, **Harborstone Insurance**. It demonstrates how to combine several core building blocks of an AI agent system on top of a real relational database:

- A **Model Context Protocol (MCP) server** that exposes insurance data and actions safely to an LLM (tools, resources, and prompts) instead of granting direct SQL access.
- Four **RAG (Retrieval-Augmented Generation) architectures** — Naive, Hybrid (dense + BM25), Agentic (multi-hop), and Self-RAG verification — plus an evaluation harness to compare them.
- A multi-layer **agent memory system** — short-term transcript, scratchpad, episodic store, semantic store, and an LLM-driven consolidation layer.
- A **context-window optimization pipeline** (sliding window → observation masking → recursive summarization → zone pruning) with its own metrics and tests.
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
│   └── requirements.txt
│
├── rag/
│   ├── vector_store.py          # Chroma-backed persistent vector store
│   ├── embedding.py             # Embedding pipeline wrapper (currently a mock implementation)
│   ├── naive_rag.py             # Baseline: vector search + generation
│   ├── hybrid_rag.py            # Dense vector search + BM25, fused with Reciprocal Rank Fusion
│   ├── agentic_rag.py           # Multi-hop retrieval loop (asks the LLM if context is sufficient)
│   └── self_rag_check.py        # Relevance and groundedness verification
│
├── memory/
│   ├── short_term.py            # Rolling, session-scoped conversation buffer
│   ├── scratchpad.py            # Persistent plan / sub-goals / working variables
│   ├── episodic_store.py        # JSON-persisted raw episodes
│   ├── semantic_store.py        # JSON-persisted consolidated facts (versioned, with expiry & conflicts)
│   ├── consolidation.py         # LLM-driven extraction of facts from episodes into the semantic store
│   └── router.py                # Scores a memory and decides EPISODIC vs FORGET
│
├── context_eval/
│   ├── sliding_window.py
│   ├── observation_masking.py
│   ├── recursive_summary.py
│   ├── zone_pruning.py
│   ├── evaluator.py              # Runs all four stages as one pipeline
│   ├── metrics.py                # Token reduction / latency reporting
│   ├── tests.py                  # Synthetic benchmarks at 100–1000 messages
│   └── README.md
│
├── retrieval_eval/
│   ├── evaluation.py             # Compares Naive / Hybrid / Agentic RAG on sample questions
│   └── questions.json
│
└── db/
    ├── schema.sql                 # Customers, Employees, Vessels, Policies, Claims, Payments
    ├── seed(2).sql                 # Sample rows for the above tables
    ├── ERD.png                    # Entity-relationship diagram
    └── README.md.txt
```

## Prerequisites

- Python 3.10+
- A MySQL server (e.g. via **XAMPP**) — only required for the MCP server and any module that queries `db/`
- pip

## Installation

Install the dependencies declared in the repo, plus a few packages that are imported by the code but currently **missing from the requirements files** (see [Known Issues](#known-issues--suggested-fixes) below):

```bash
git clone https://github.com/Adham19xx/Harborstone-Insurance-A.git
cd Harborstone-Insurance-A

pip install -r requirements.txt.txt
pip install -r agent/requirements.txt

# Additional packages used by rag/, mcp_server/, and retrieval_eval/
# but not currently listed in any requirements file:
pip install fastmcp chromadb rank_bm25 tabulate
```

## Database Setup

The MCP server connects to MySQL with these defaults (matching a stock **XAMPP** install — host `localhost`, user `root`, empty password):

1. Start MySQL/XAMPP.
2. Create the schema:
   ```bash
   mysql -u root -p < db/schema.sql
   ```
3. Load the sample data:
   ```bash
   mysql -u root -p harborstone_insurance < "db/seed(2).sql"
   ```
4. See `db/ERD.png` for the full entity-relationship diagram: `Customers` and `Vessels` feed into `Policies`, which in turn feed into `Claims` and `Payments`.

## Running the Demo

1. Start the MCP server (either copy works — see [Known Issues](#known-issues--suggested-fixes) for the difference):
   ```bash
   python mcp_server/server.py
   ```
2. In a second terminal, run the agent client, which drives a full walkthrough — handshake, reading resources, fetching a prompt template, calling a tool with progress tracking, and testing role-based authorization plus human-in-the-loop confirmation:
   ```bash
   python agent/client.py
   ```

## MCP Server Capabilities

**Resources** (read-only data)
- `harborstone://policies/terms/marine-hull` — static marine hull policy terms and conditions.
- `harborstone://claims/pending-summary` — live query of all claims with `status = 'Pending'`.

**Prompts**
- `draft_claim_investigation(claim_id)` — generates a structured prompt for investigating a specific claim.

**Tools**
- `audit_high_risk_policies` — audits policies with `premium > $8,000`, reporting progress at 20% / 60% / 100%.
- `approve_claim(claim_id, user_role)` — approves a pending claim. Requires the `Claims Officer` or `Manager` role, and requires explicit human confirmation (MCP elicitation) before approving any claim over $10,000.
- `switch_user_role(new_role)` — changes the active session role and pushes a `tools/list_changed` notification to the client.

## RAG Architectures

| Module | Approach |
|---|---|
| `naive_rag.py` | Vector similarity search → generation |
| `hybrid_rag.py` | Dense vector search + BM25 sparse search, fused via Reciprocal Rank Fusion |
| `agentic_rag.py` | Up to 3 retrieval hops; the LLM decides when context is sufficient |
| `self_rag_check.py` | Verifies chunk relevance and answer groundedness after retrieval |

`retrieval_eval/evaluation.py` benchmarks all three architectures against the questions in `retrieval_eval/questions.json` (using a mock LLM) and prints an accuracy / avg-tokens / avg-latency comparison table.

## Memory System

| Layer | Role |
|---|---|
| `short_term.py` | Session-scoped rolling transcript |
| `scratchpad.py` | Persistent plan, sub-goals, and working variables — not affected by transcript pruning |
| `episodic_store.py` | Raw conversation episodes, persisted to JSON |
| `semantic_store.py` | Consolidated facts with versioning, expiry, and conflict flagging |
| `consolidation.py` | Uses an LLM to extract structured facts from unconsolidated episodes |
| `router.py` | Scores a candidate memory (importance, future use, frequency, preference, emotion) to decide `EPISODIC` vs `FORGET`, logging every decision |

## Context Window Optimization

A four-stage pipeline in `context_eval/`, run in this order:

1. **Sliding Window** — keep only the most recent N messages.
2. **Observation Masking** — drop low-signal noise (`"ok"`, `"thanks"`, emoji-only replies, etc.).
3. **Recursive Summarization** — collapse older messages into a single system-role summary once a threshold is exceeded.
4. **Zone-based Pruning** — trim non-protected messages once the total exceeds a max, while always preserving protected roles (e.g. `system`).

`metrics.py` reports token reduction and latency; `tests.py` benchmarks the pipeline on synthetic conversations of 100–1000 messages.

## Known Issues / Suggested Fixes

- **Duplicated files**: `server.py`, `schema.sql`, and `seed(2).sql` at the repo root are exact duplicates of `mcp_server/server.py` (aside from the import shown below) and the files in `db/`. Consider keeping a single source of truth.
- **Two server variants**: `server.py` imports `FastMCP` from the official `mcp.server.fastmcp` package, while `mcp_server/server.py` imports it from the standalone `fastmcp` package. Pick one to avoid confusion.
- **Missing dependencies**: `rag/hybrid_rag.py` needs `rank_bm25`, `rag/vector_store.py` needs `chromadb`, `retrieval_eval/evaluation.py` needs `tabulate`, and `mcp_server/server.py` needs `fastmcp` — none of these appear in any `requirements*.txt` file.
- **Stray file extensions**: `requirements.txt.txt`, `README.md.txt`, and `seed(2).sql` carry extra characters from how they were saved/downloaded. Renaming to `requirements.txt`, `README.md`, and `seed.sql` would clean up the repo.
- **Hardcoded DB credentials**: `get_db_connection()` hardcodes `localhost` / `root` / empty password. Since `.env` is already in `.gitignore`, moving these into environment variables (loaded via `python-dotenv`, already a dependency) is a natural next step.
- **Mock embeddings**: `rag/embedding.py` currently returns placeholder vectors rather than calling a real embedding model — swap in an actual embedding API before relying on retrieval quality.
- **Import style in `context_eval/`**: `evaluator.py` and `tests.py` use bare imports (e.g. `from sliding_window import SlidingWindow`), so scripts in that folder must be run with `context_eval/` as the working directory (or with it added to `sys.path`) rather than imported as `context_eval.evaluator` from the repo root.

## License

No license file is currently included in this repository.