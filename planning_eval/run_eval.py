"""
planning_eval/run_eval.py — Evaluation Harness for Harborstone Planning
=====================================================================
Run this script to generate the comparison table required by the PDF.

Usage:
  python -m planning_eval.run_eval

Output:
  - Console: comparison table (accuracy, LLM calls, tokens, latency)
  - artifacts/eval_<timestamp>.json  (full trace per test case)
  - planning_eval/comparison_table.md (markdown table for the README)

The script runs EVERY required method against EVERY applicable test case:
  - Plan-and-Solve   vs Tree of Thoughts vs LATS
  - Self-Refine      vs Reflexion
  - Grounded env     vs ungrounded (random) env  [for TC-09, TC-10]

Do NOT modify test cases between runs (fixed suite rule).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# ── Local imports ──────────────────────────────────────────────────────────
from planning.algorithms.plan_and_solve import plan_and_solve
from planning.algorithms.tree_of_thoughts import tree_of_thoughts
from planning.algorithms.lats import lats
from planning.algorithms.self_refine import reflect_and_refine
from planning.algorithms.reflexion import reflexion
from planning.algorithms.environment import Environment
from planning.algorithms.router import classify_subtask, explain_routing

from planning_eval.test_suite import TEST_SUITE, TestCase

load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup — Anthropic Claude (as specified by the project)
# ---------------------------------------------------------------------------

def get_llm() -> ChatAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found. Add it to your .env file."
        )
    return ChatAnthropic(
        model="claude-3-5-haiku-20241022",   # fast + affordable for eval
        api_key=api_key,
        temperature=0.0,                     # deterministic for reproducibility
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Result recorder
# ---------------------------------------------------------------------------

def _record(
    tc: TestCase,
    method: str,
    success: bool,
    score: float,
    llm_calls: int,
    tokens: int,
    latency: float,
    output: str = "",
    extra: dict | None = None,
) -> dict:
    return {
        "tc_id": tc.id,
        "tc_description": tc.description,
        "method": method,
        "success": success,
        "score": round(score, 3),
        "llm_calls": llm_calls,
        "tokens": tokens,
        "latency_s": round(latency, 3),
        "output_snippet": output[:200],
        **(extra or {}),
    }


# ---------------------------------------------------------------------------
# Run one test case through all applicable methods
# ---------------------------------------------------------------------------

def run_one(tc: TestCase, llm, env: Environment) -> list[dict]:
    rows: list[dict] = []
    ctx_text = json.dumps(tc.request_data, indent=2)

    # ── Plan-and-Solve ────────────────────────────────────────────────────
    try:
        ps_result = plan_and_solve(
            question=tc.request,
            llm=llm,
            task_id=tc.id,
            context=tc.request_data,
        )
        fb = env.evaluate(tc.request, ps_result.solution)
        rows.append(_record(
            tc, "plan_and_solve",
            fb.success, fb.score,
            ps_result.llm_calls, ps_result.tokens_used, ps_result.latency_s,
            ps_result.solution,
        ))
    except Exception as e:
        rows.append(_record(tc, "plan_and_solve", False, 0.0, 0, 0, 0.0, extra={"error": str(e)}))

    # ── Tree of Thoughts ──────────────────────────────────────────────────
    try:
        tot_result = tree_of_thoughts(
            problem=tc.request,
            llm=llm,
            depth=2,
            beam_width=2,
            task_id=tc.id,
            context=tc.request_data,
        )
        fb = env.evaluate(tc.request, tot_result.best_thought.state)
        rows.append(_record(
            tc, "tree_of_thoughts",
            fb.success, fb.score,
            tot_result.llm_calls, tot_result.tokens_used, tot_result.latency_s,
            tot_result.best_thought.state,
        ))
    except Exception as e:
        rows.append(_record(tc, "tree_of_thoughts", False, 0.0, 0, 0, 0.0, extra={"error": str(e)}))

    # ── LATS (grounded) ───────────────────────────────────────────────────
    try:
        lats_result = lats(
            task=tc.request,
            llm=llm,
            environment=env,
            max_iterations=4,
            task_id=tc.id,
            context=tc.request_data,
        )
        rows.append(_record(
            tc, "lats_grounded",
            lats_result.success, lats_result.best_score,
            lats_result.llm_calls, lats_result.tokens_used, lats_result.latency_s,
            lats_result.output,
        ))
    except Exception as e:
        rows.append(_record(tc, "lats_grounded", False, 0.0, 0, 0, 0.0, extra={"error": str(e)}))

    # ── LATS (ungrounded — random env, for grounding contrast) ────────────
    try:
        import random as _random
        from planning.algorithms.environment import Environment as _Env, EnvironmentFeedback as _EF

        class _RandomEnv(_Env):
            """Fake environment identical to the toolkit's default."""
            def evaluate(self, task, output, **kwargs) -> _EF:
                score = _random.uniform(0.4, 1.0)
                return _EF(
                    score=score,
                    success=score >= self.success_threshold,
                    message="UNGROUNDED: random score (toolkit default)",
                    grounding_source="random.uniform (toolkit fake)",
                )

        ungrounded_env = _RandomEnv()
        lats_ung = lats(
            task=tc.request,
            llm=llm,
            environment=ungrounded_env,
            max_iterations=4,
            task_id=tc.id,
            context=tc.request_data,
        )
        grounded_env_for_ung = env
        final_fb = grounded_env_for_ung.evaluate(tc.request, lats_ung.output)
        rows.append(_record(
            tc, "lats_ungrounded",
            final_fb.success, final_fb.score,   # score using GROUNDED env to show the difference
            lats_ung.llm_calls, lats_ung.tokens_used, lats_ung.latency_s,
            lats_ung.output,
            extra={"note": "MCTS used random feedback; final score measured by grounded env"},
        ))
    except Exception as e:
        rows.append(_record(tc, "lats_ungrounded", False, 0.0, 0, 0, 0.0, extra={"error": str(e)}))

    # ── Self-Refine ───────────────────────────────────────────────────────
    try:
        # First, get a draft from PS to refine
        ps_for_refine = plan_and_solve(question=tc.request, llm=llm, context=tc.request_data)
        sr_result = reflect_and_refine(
            goal=tc.request,
            draft=ps_for_refine.solution,
            llm=llm,
            task_id=tc.id,
            context=ctx_text,
        )
        fb = env.evaluate(tc.request, sr_result.revised)
        rows.append(_record(
            tc, "self_refine",
            fb.success, fb.score,
            sr_result.llm_calls + 2, sr_result.tokens_used, sr_result.latency_s,
            sr_result.revised,
            extra={"grounded_issues_before": sr_result.grounded_issues},
        ))
    except Exception as e:
        rows.append(_record(tc, "self_refine", False, 0.0, 0, 0, 0.0, extra={"error": str(e)}))

    # ── Reflexion ─────────────────────────────────────────────────────────
    try:
        ref_result = reflexion(
            task=tc.request,
            llm=llm,
            environment=env,
            max_trials=3,
            memory_size=3,
            task_id=tc.id,
            context=ctx_text,
        )
        rows.append(_record(
            tc, "reflexion",
            ref_result.success, env.evaluate(tc.request, ref_result.output).score,
            ref_result.llm_calls, ref_result.tokens_used, ref_result.latency_s,
            ref_result.output,
            extra={
                "trials": len(ref_result.trials),
                "episodic_memory": ref_result.memory,
            },
        ))
    except Exception as e:
        rows.append(_record(tc, "reflexion", False, 0.0, 0, 0, 0.0, extra={"error": str(e)}))

    return rows


# ---------------------------------------------------------------------------
# Markdown table builder
# ---------------------------------------------------------------------------

def _build_markdown_table(all_rows: list[dict]) -> str:
    header = (
        "| TC | Method | Success | Score | LLM Calls | Tokens | Latency (s) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows_md = ""
    for r in all_rows:
        ok = "✅" if r["success"] else "❌"
        rows_md += (
            f"| {r['tc_id']} | `{r['method']}` | {ok} | {r['score']:.3f} "
            f"| {r['llm_calls']} | {r['tokens']} | {r['latency_s']:.2f} |\n"
        )
    return header + rows_md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Harborstone Planning Evaluation Harness")
    print("=" * 70)

    llm = get_llm()
    env = Environment(success_threshold=0.65)

    all_rows: list[dict] = []

    for tc in TEST_SUITE:
        print(f"\n▶  {tc.id}: {tc.description[:60]}…")
        rows = run_one(tc, llm, env)
        all_rows.extend(rows)
        for r in rows:
            status = "✅" if r["success"] else "❌"
            print(f"   {status}  {r['method']:25s}  score={r['score']:.3f}  "
                  f"calls={r['llm_calls']}  tokens={r['tokens']}  "
                  f"lat={r['latency_s']:.1f}s")

    # Save JSON trace
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = artifacts_dir / f"eval_{ts}.json"
    trace_path.write_text(json.dumps(all_rows, indent=2, default=str), encoding="utf-8")
    print(f"\n✓ Trace saved to {trace_path}")

    # Save markdown comparison table
    md_table = _build_markdown_table(all_rows)
    md_path = Path("planning_eval") / "comparison_table.md"
    md_path.parent.mkdir(exist_ok=True)
    md_path.write_text(
        "# Harborstone Planning — Method Comparison Table\n\n"
        + md_table + "\n\n"
        + "> Generated by `python -m planning_eval.run_eval`\n",
        encoding="utf-8",
    )
    print(f"✓ Comparison table saved to {md_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(md_table)


if __name__ == "__main__":
    main()
