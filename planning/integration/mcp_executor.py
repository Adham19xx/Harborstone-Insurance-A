from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class HarborstoneMCPExecutor:
    """Real MCP adapter; the planner never accesses Harborstone's DB directly."""

    def __init__(self, server_script: str | Path = "mcp_server/server.py") -> None:
        self.server_script = Path(server_script)
        self._stdio = None
        self._session = None
        self.mcp_calls = 0
        self.call_trace: list[dict[str, Any]] = []

    async def __aenter__(self) -> "HarborstoneMCPExecutor":
        params = StdioServerParameters(command=sys.executable, args=[str(self.server_script)], env=None)
        self._stdio = stdio_client(params)
        read, write = await self._stdio.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            await self._session.__aexit__(exc_type, exc, tb)
        if self._stdio is not None:
            await self._stdio.__aexit__(exc_type, exc, tb)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("HarborstoneMCPExecutor must be used as an async context manager")
        started = time.perf_counter()
        self.mcp_calls += 1
        result = await self._session.call_tool(tool_name, arguments=arguments)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        texts: list[str] = []
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            if text is not None:
                texts.append(text)
        raw = "\n".join(texts)
        try:
            value: Any = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            value = raw
        payload = {"tool": tool_name, "arguments": arguments, "result": value, "latency_ms": latency_ms}
        self.call_trace.append(payload)
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool {tool_name} returned an error: {raw}")
        return payload
