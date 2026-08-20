from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GraphStatus(str, Enum):
    """
    Lifecycle status of a state-graph run.

    These values describe whether a run is actively executing,
    waiting for an external event, paused for human input,
    failed, or finished.
    """

    RUNNING = "RUNNING"
    WAITING = "WAITING"
    PAUSED_FOR_HITL = "PAUSED_FOR_HITL"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class GraphEventType(str, Enum):
    """
    Types of meaningful events recorded in the graph history.
    """

    TRANSITION = "TRANSITION"
    WAIT = "WAIT"
    RESUME = "RESUME"
    HITL_REQUESTED = "HITL_REQUESTED"
    HITL_RESOLVED = "HITL_RESOLVED"
    FAILURE = "FAILURE"
    COMPLETED = "COMPLETED"


class GraphEvent(BaseModel):
    """
    One recorded event in a graph run.

    The event history gives the Platform a structured timeline
    of the run and also makes resume/debugging easier.
    """

    event_type: GraphEventType
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HITLRequest(BaseModel):
    """
    Generic Human-in-the-Loop request.

    Member 1 owns the pause/state mechanism.
    The Platform/Admin side can later display this request and
    submit a decision through the backend.
    """

    request_id: str
    run_id: str
    graph_name: str
    current_state: str

    reason: str
    prompt: str

    status: str = "PENDING"

    allowed_actions: List[str] = Field(
        default_factory=lambda: ["APPROVE", "REJECT"]
    )

    decision: Optional[str] = None
    decided_by: Optional[str] = None
    decision_notes: Optional[str] = None


class GraphState(BaseModel):
    """
    Shared durable state format for every Harborstone State Graph.

    Important:
    - `current_state` identifies the exact graph node/state.
    - `status` identifies the run lifecycle.
    - `data` stores graph-specific business data.
    - `history` stores meaningful transitions/events.
    - `checkpoint_version` increments whenever a new checkpoint
      is successfully persisted.
    """

    run_id: str
    graph_name: str

    current_state: str = "START"
    status: GraphStatus = GraphStatus.RUNNING

    customer_id: Optional[int] = None
    policy_id: Optional[int] = None
    claim_id: Optional[int] = None
    vessel_id: Optional[int] = None

    data: Dict[str, Any] = Field(default_factory=dict)

    history: List[GraphEvent] = Field(default_factory=list)

    hitl_request: Optional[HITLRequest] = None

    checkpoint_version: int = 0

    error_message: Optional[str] = None

    def add_event(
        self,
        event_type: GraphEventType,
        *,
        from_state: Optional[str] = None,
        to_state: Optional[str] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add one structured event to the run history.
        """

        self.history.append(
            GraphEvent(
                event_type=event_type,
                from_state=from_state,
                to_state=to_state,
                message=message,
                metadata=metadata or {},
            )
        )

    def transition_to(
        self,
        next_state: str,
        *,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Move the graph from its current state to the next state.

        Checkpointing itself is intentionally NOT performed here.
        The graph runner/checkpoint layer will persist the state
        after this meaningful transition.
        """

        previous_state = self.current_state
        self.current_state = next_state

        self.add_event(
            GraphEventType.TRANSITION,
            from_state=previous_state,
            to_state=next_state,
            message=message,
            metadata=metadata,
        )

    def mark_waiting(
        self,
        *,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark the run as waiting for an external event or response.
        """

        self.status = GraphStatus.WAITING

        self.add_event(
            GraphEventType.WAIT,
            from_state=self.current_state,
            to_state=self.current_state,
            message=message,
            metadata=metadata,
        )

    def mark_running(
        self,
        *,
        message: Optional[str] = None,
    ) -> None:
        """
        Mark a previously waiting or paused run as active again.
        """

        previous_status = self.status
        self.status = GraphStatus.RUNNING

        self.add_event(
            GraphEventType.RESUME,
            from_state=self.current_state,
            to_state=self.current_state,
            message=message or f"Run resumed from {previous_status.value}",
        )

    def mark_failed(
        self,
        error_message: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark the run as failed.

        Member 2 can later connect this state to the Failure/Ticket
        system without changing the shared state format.
        """

        self.status = GraphStatus.FAILED
        self.error_message = error_message

        self.add_event(
            GraphEventType.FAILURE,
            from_state=self.current_state,
            to_state=self.current_state,
            message=error_message,
            metadata=metadata,
        )

    def mark_completed(
        self,
        *,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark the graph run as successfully completed.
        """

        self.status = GraphStatus.COMPLETED

        self.add_event(
            GraphEventType.COMPLETED,
            from_state=self.current_state,
            to_state=self.current_state,
            message=message,
            metadata=metadata,
        )