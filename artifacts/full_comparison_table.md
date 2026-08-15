# Harborstone Week 4 — Comprehensive Decomposition & Planning Comparison Table

| Planning Concern / Method | Task Success | Avg LLM / Tool Calls | Avg Tokens | Avg Latency | Est Cost/Run | Justification & Production Placement |
|---|---|---|---|---|---|---|
| **Decomposition-First** | 12/15 (80.0%) | 5.0 | 6,335 | 0.28s | $0.038 | Baseline for purely static, non-branching requests; fails when early tool calls reveal ineligibility. |
| **Dynamic Decomposition** | 15/15 (100.0%) | 5.6 | 7,913 | 0.46s | $0.059 | **Shipped as Default Top-Level Planner**: Reshapes plan after observing early tool failures/ineligibility. |
| **Plan-and-Solve (PS)** | 15/15 (100.0%) | 1.0 | 1,450 | 0.12s | $0.010 | **Shipped for Math/Actuarial Sub-Tasks**: Fast, zero-branching sequential premium/deductible calculation. |
| **Tree of Thoughts (ToT)** | 15/15 (100.0%) | 16.0 | 5,400 | 0.39s | $0.038 | **Shipped for Risk Ranking/Tradeoffs**: BFS/DFS search evaluates candidate coverage & deductible options. |
| **LATS (Ungrounded Env)** | 12/15 (80.0%) | 8.0 | 6,900 | 0.52s | $0.052 | *Ablation Baseline*: Misses underwriting violations due to superficial self-evaluation. |
| **LATS (Grounded Env)** | 12/15 (80.0%) | 10.0 | 8,100 | 0.64s | $0.065 | **Shipped for High-Stakes Endorsements**: MCTS guided by real underwriting rules & reflections. |
| **Self-Refine** | 15/15 (100.0%) | 3.0 | 3,100 | 0.24s | $0.022 | **Shipped for Communication Synthesis**: Fast 1-draft rubric critique for customer notifications. |
| **Reflexion (Multi-Trial)** | 12/15 (80.0%) | 4.4 | 5,720 | 0.58s | $0.042 | **Shipped for Complex Policy Restructuring**: Multi-trial loop carrying verbal reflection memory across attempts. |