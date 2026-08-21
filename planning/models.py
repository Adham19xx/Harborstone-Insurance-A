from __future__ import annotations

from typing import Any
import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, model_validator


ALLOWED_MCP_TOOLS = {
    "get_customer_policies",
    "get_policy_coverage",
    "get_vessel",
    "check_vessel_eligibility",
    "estimate_policy_premium_change",
    "get_policy_update_requirements",
    "apply_cancellation_rules",
}



class Task(BaseModel):
    """A validated DAG node, extended from the reference toolkit for real MCP execution."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$")
    instruction: str = Field(min_length=5)
    depends_on: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    kind: str = Field(default="mcp", pattern=r"^(mcp|synthesis)$")

    @model_validator(mode="after")
    def validate_tool(self) -> "Task":
        if self.kind == "mcp":
            if self.tool_name not in ALLOWED_MCP_TOOLS:
                raise ValueError(f"Unsupported Harborstone MCP tool: {self.tool_name!r}")
        elif self.tool_name is not None:
            raise ValueError("Synthesis tasks must not declare tool_name")
        return self


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = Field(min_length=5)
    tasks: list[Task] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_dag(self) -> "Plan":
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Task ids must be unique")
        known = set(ids)
        for task in self.tasks:
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(f"{task.id} has unknown dependencies: {sorted(missing)}")
            if task.id in task.depends_on:
                raise ValueError(f"{task.id} cannot depend on itself")
        if not nx.is_directed_acyclic_graph(self.graph):
            cycle = nx.find_cycle(self.graph)
            blocked = sorted({node for edge in cycle for node in edge[:2]})
            raise ValueError(f"Cycle detected; blocked tasks: {blocked}")
        return self

    @property
    def graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(task.id for task in self.tasks)
        graph.add_edges_from(
            (dependency, task.id)
            for task in self.tasks
            for dependency in task.depends_on
        )
        return graph

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self.graph))

    def execution_batches(self) -> list[list[str]]:
        return [sorted(generation) for generation in nx.topological_generations(self.graph)]

    def task(self, task_id: str) -> Task:
        return next(task for task in self.tasks if task.id == task_id)

    def terminal_tasks(self) -> list[str]:
        return [node for node, degree in self.graph.out_degree if degree == 0]

# Re-export EnvironmentFeedback from environment.py to avoid duplicate definitions
from .environment import EnvironmentFeedback

