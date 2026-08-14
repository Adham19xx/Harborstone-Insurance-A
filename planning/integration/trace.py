from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import time


@dataclass
class RunTrace:
    """Trace format based on the required reference toolkit's JSON artifacts."""

    method: str
    request_id: str
    goal: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    success: bool = False
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_usage_available: bool = False
    mcp_calls: int = 0
    latency_ms: float = 0.0
    plan: dict[str, Any] | None = None
    execution: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    plan_changes: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None
    error: str | None = None
    _started_perf: float = field(default_factory=time.perf_counter, repr=False)

    def add_llm_usage(self, response: Any) -> None:
        self.llm_calls += 1
        usage = getattr(response, "usage_metadata", None)
        if not isinstance(usage, dict):
            return
        self.token_usage_available = True
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.total_tokens = self.input_tokens + self.output_tokens

    def finish(self, success: bool, result: Any = None, error: str | None = None) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.latency_ms = round((time.perf_counter() - self._started_perf) * 1000, 2)
        self.success = success
        self.result = result
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload.pop("_started_perf", None)
        return payload

    def save(self, root: str | Path) -> Path:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = path / f"{self.method}-{self.request_id}-{stamp}.json"
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return out
