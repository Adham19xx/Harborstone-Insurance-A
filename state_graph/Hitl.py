from __future__ import annotations

from typing import Optional

from state_graph.models import (
    GraphState,
    GraphStatus,
    HITLRequest,
    GraphEventType,
)
from state_graph.checkpointing import CheckpointManager


class HITLManager:
    """
    Shared Human-in-the-Loop manager for Harborstone State Graphs.

    Responsibilities:
    - Create a real HITL request.
    - Pause the graph.
    - Persist the paused state using checkpointing.
    - Accept an explicit human decision.
    - Resume the graph after the decision.
    - Persist the resumed state.

    The manager does NOT decide business outcomes.
    The graph decides what APPROVE / REJECT means.
    """

    def __init__(self, checkpoint_manager: CheckpointManager) -> None:
        self.checkpoints = checkpoint_manager

    # ==========================================================
    # REQUEST HUMAN APPROVAL
    # ==========================================================

    def request_approval(
        self,
        state: GraphState,
        *,
        reason: str,
        prompt: str,
        allowed_actions: Optional[list[str]] = None,
        request_id: Optional[str] = None,
    ) -> GraphState:
        """
        Pause a graph and create a persistent HITL request.

        This represents a genuine human-in-the-loop pause:
        the graph cannot continue until an external human
        submits a decision.
        """

        if state.status == GraphStatus.COMPLETED:
            raise ValueError("Cannot request HITL for a completed graph.")

        if state.status == GraphStatus.FAILED:
            raise ValueError(
                "Cannot request HITL while the graph is FAILED."
            )

        if state.hitl_request is not None:
            if state.hitl_request.status == "PENDING":
                raise ValueError(
                    "A pending HITL request already exists for this run."
                )

        actions = allowed_actions or ["APPROVE", "REJECT"]

        request = HITLRequest(
            request_id=request_id or f"{state.run_id}-hitl-{state.checkpoint_version + 1}",
            run_id=state.run_id,
            graph_name=state.graph_name,
            current_state=state.current_state,
            reason=reason,
            prompt=prompt,
            status="PENDING",
            allowed_actions=actions,
        )

        state.hitl_request = request
        state.status = GraphStatus.PAUSED_FOR_HITL

        state.add_event(
            GraphEventType.HITL_REQUESTED,
            from_state=state.current_state,
            to_state=state.current_state,
            message="Human approval is required before the graph can continue.",
            metadata={
                "request_id": request.request_id,
                "reason": reason,
                "allowed_actions": actions,
            },
        )

        # Persist the PAUSED state.
        self.checkpoints.save_checkpoint(state)

        return state

    # ==========================================================
    # RESOLVE HUMAN DECISION
    # ==========================================================

    def resolve(
        self,
        state: GraphState,
        *,
        decision: str,
        decided_by: str,
        decision_notes: str = "",
    ) -> GraphState:
        """
        Resolve an existing HITL request.

        The human must explicitly provide one of the actions
        allowed by the HITL request.

        After the decision:
            PAUSED_FOR_HITL -> RUNNING

        The graph itself is responsible for deciding the next
        business state.
        """

        if state.status != GraphStatus.PAUSED_FOR_HITL:
            raise ValueError(
                "Graph is not paused for HITL."
            )

        if state.hitl_request is None:
            raise ValueError(
                "No HITL request exists for this graph run."
            )

        request = state.hitl_request

        if request.status != "PENDING":
            raise ValueError(
                f"HITL request is already {request.status}."
            )

        decision = decision.upper().strip()

        allowed_actions = {
            action.upper()
            for action in request.allowed_actions
        }

        if decision not in allowed_actions:
            raise ValueError(
                f"Invalid HITL decision '{decision}'. "
                f"Allowed actions: {sorted(allowed_actions)}"
            )

        if not decided_by.strip():
            raise ValueError(
                "A human decision must identify who made the decision."
            )

        # Store the human decision.
        request.status = "RESOLVED"
        request.decision = decision
        request.decided_by = decided_by
        request.decision_notes = decision_notes

        state.add_event(
            GraphEventType.HITL_RESOLVED,
            from_state=state.current_state,
            to_state=state.current_state,
            message=f"Human decision received: {decision}",
            metadata={
                "request_id": request.request_id,
                "decision": decision,
                "decided_by": decided_by,
                "decision_notes": decision_notes,
            },
        )

        # The graph can now continue.
        state.status = GraphStatus.RUNNING

        state.add_event(
            GraphEventType.RESUME,
            from_state=state.current_state,
            to_state=state.current_state,
            message="Graph resumed after human decision.",
            metadata={
                "request_id": request.request_id,
            },
        )

        # Persist the resumed state.
        self.checkpoints.save_checkpoint(state)

        return state

    # ==========================================================
    # READ PENDING REQUEST
    # ==========================================================

    @staticmethod
    def get_pending_request(
        state: GraphState,
    ) -> Optional[HITLRequest]:
        """
        Return the pending HITL request for a graph run.

        Returns None if there is no pending request.
        """

        if state.hitl_request is None:
            return None

        if state.hitl_request.status != "PENDING":
            return None

        return state.hitl_request