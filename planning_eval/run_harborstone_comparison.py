from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planning.algorithms.decomposition import decompose_goal, execute_plan, final_output
from planning.algorithms.dynamic_decomposition import dynamic_decomposition
from planning.integration.mcp_executor import HarborstoneMCPExecutor
from planning.integration.trace import RunTrace
from planning.requests.harborstone_requests import REAL_REQUESTS


def build_llm() -> ChatMistralAI:
    load_dotenv(ROOT / ".env")
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY is missing. Add it to .env before the real evaluation run.")
    return ChatMistralAI(api_key=key, model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"), max_retries=2)


async def run_case(request: dict) -> dict:
    llm = build_llm()
    results = {}
    server = ROOT / "mcp_server" / "server.py"
    async with HarborstoneMCPExecutor(server) as executor:
        trace = RunTrace("decomposition_first", request["request_id"], request["text"])
        try:
            plan = decompose_goal(request["text"], llm, request, trace)
            trace.plan = plan.model_dump()
            outputs = await execute_plan(plan, llm, executor, trace)
            trace.mcp_calls = executor.mcp_calls
            trace.finish(True, final_output(plan, outputs))
        except Exception as exc:
            trace.mcp_calls = executor.mcp_calls
            trace.finish(False, error=str(exc))
        path = trace.save(ROOT / "artifacts")
        results["decomposition_first"] = {"trace": str(path), **trace.to_dict()}

    llm = build_llm()
    async with HarborstoneMCPExecutor(server) as executor:
        trace = RunTrace("dynamic", request["request_id"], request["text"])
        try:
            history = await dynamic_decomposition(request["text"], llm, request, executor, trace)
            trace.mcp_calls = executor.mcp_calls
            trace.finish(True, history[-1][1] if history else None)
        except Exception as exc:
            trace.mcp_calls = executor.mcp_calls
            trace.finish(False, error=str(exc))
        path = trace.save(ROOT / "artifacts")
        results["dynamic"] = {"trace": str(path), **trace.to_dict()}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", choices=[r["request_id"] for r in REAL_REQUESTS], default=REAL_REQUESTS[0]["request_id"])
    args = parser.parse_args()
    request = next(r for r in REAL_REQUESTS if r["request_id"] == args.request_id)
    results = asyncio.run(run_case(request))
    comparison = {
        "request_id": args.request_id,
        "same_request_type": True,
        "methods": {
            method: {
                "success": data["success"],
                "mcp_calls": data["mcp_calls"],
                "llm_calls": data["llm_calls"],
                "total_tokens": data["total_tokens"],
                "latency_ms": data["latency_ms"],
                "trace": data["trace"],
                "plan_changes": data["plan_changes"],
            }
            for method, data in results.items()
        },
        "dynamic_diverged": bool(results["dynamic"]["plan_changes"]),
    }
    out = ROOT / "artifacts" / f"comparison-{args.request_id}.json"
    out.write_text(json.dumps(comparison, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    print(f"Comparison artifact: {out}")


if __name__ == "__main__":
    main()
