from __future__ import annotations

from typing import Any, Dict, Optional

from state_graph.models import GraphState, GraphStatus, GraphEventType
from state_graph.checkpointing import CheckpointManager
from state_graph.Hitl import HITLManager
from state_graph.llm_tools import run_task_decomposition, call_mcp_tool_sync, run_constrained_react


class HighValueVehicleAdditionGraph:
    """
    Graph 3:
    High-Value Vehicle Addition to an Existing Auto Insurance Policy.

    This workflow demonstrates:
    - Stateful execution
    - External waiting for documents
    - External eligibility decisions
    - Cycles for incomplete documents
    - HITL approval for high-value vehicles
    - Failure handling with ticket creation
    - Checkpoint-based recovery
    """

    GRAPH_NAME = "HIGH_VALUE_VEHICLE_ADDITION"

    MAX_DOCUMENT_ATTEMPTS = 3

    DEFAULT_VALUE_THRESHOLD = 100000.0
    DEFAULT_PREMIUM_THRESHOLD = 5000.0

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self.checkpoints = checkpoint_manager
        self.hitl_manager = HITLManager(checkpoint_manager)

    # ==========================================================
    # START
    # ==========================================================

    def start(
        self,
        state: GraphState,
        vehicle_details: Dict[str, Any],
    ) -> GraphState:
        """
        Start a new vehicle-addition workflow.

        The graph validates the customer/policy request,
        stores the vehicle information, and moves toward
        eligibility checking.
        """

        state.graph_name = self.GRAPH_NAME

        if state.current_state != "START":
            raise ValueError(
                f"Graph must start from START, got {state.current_state}"
            )

        state.data["vehicle"] = vehicle_details

        self._transition(
            state,
            "VERIFY_CUSTOMER",
            "Verifying the customer requesting the policy update.",
        )
        self._checkpoint(state)

        self._transition(
            state,
            "GET_EXISTING_POLICY",
            "Loading the customer's existing auto insurance policy.",
        )
        self._checkpoint(state)

        self._transition(
            state,
            "COLLECT_VEHICLE_DETAILS",
            "Recording the new vehicle details.",
        )
        
        # Use Task Decomposition LLM pattern to extract complex vehicle data
        extracted_details = run_task_decomposition(
            "Extract structured vehicle information from raw customer input",
            str(vehicle_details)
        )
        state.data["extracted_vehicle_details"] = extracted_details

        self._checkpoint(state)

        self._transition(
            state,
            "CHECK_ELIGIBILITY",
            "Checking whether the vehicle is eligible for coverage.",
        )
        
        # Use MCP tool to check eligibility
        mcp_eligibility = call_mcp_tool_sync(
            "check_vessel_eligibility",
            {
                "vessel_type": vehicle_details.get("type", "Yacht"),
                "year_built": vehicle_details.get("year", 2018),
                "value": vehicle_details.get("value", 100000.0)
            }
        )
        state.data["mcp_eligibility_result"] = mcp_eligibility

        self._checkpoint(state)

        return state

    # ==========================================================
    # ELIGIBILITY
    # ==========================================================

    def evaluate_eligibility(
        self,
        state: GraphState,
        eligibility_result: Dict[str, Any],
    ) -> GraphState:
        """
        Process the result of an external eligibility check.

        The branch is based on the actual external result,
        not an LLM-generated guess.
        """

        if state.current_state != "CHECK_ELIGIBILITY":
            raise ValueError(
                f"Expected CHECK_ELIGIBILITY, got {state.current_state}"
            )

        eligible = bool(eligibility_result.get("eligible"))

        state.data["eligibility_result"] = eligibility_result

        if not eligible:
            self._transition(
                state,
                "INELIGIBLE",
                "External eligibility check determined that the vehicle is not eligible.",
            )

            state.data["final_decision"] = "REJECTED"

            self._transition(
                state,
                "CLOSE",
                "Vehicle addition request rejected because the vehicle is ineligible.",
            )

            state.mark_completed(
                message="Vehicle addition rejected due to eligibility rules."
            )

            self._checkpoint(state)
            return state

        self._transition(
            state,
            "REQUEST_VEHICLE_DOCUMENTS",
            "Vehicle is eligible; required documents must be collected.",
        )
        self._checkpoint(state)

        self._transition(
            state,
            "AWAITING_DOCUMENTS",
            "Waiting for the customer to submit the required vehicle documents.",
        )

        state.mark_waiting(
            message=(
                "Waiting for external vehicle documents such as "
                "proof of ownership, registration, and valuation."
            ),
            metadata={
                "wait_reason": "VEHICLE_DOCUMENTS",
                "external_event_required": True,
            },
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # RESUME AFTER DOCUMENT SUBMISSION
    # ==========================================================

    def resume_with_documents(
        self,
        state: GraphState,
        documents: Dict[str, Any],
    ) -> GraphState:
        """
        Resume the graph when the customer submits documents.

        Incomplete documents create a genuine cycle:

        REQUEST_VEHICLE_DOCUMENTS
                ↓
        AWAITING_DOCUMENTS
                ↓
        VALIDATE_DOCUMENTS
                ↓
        REQUEST_VEHICLE_DOCUMENTS
                ↺
        """

        if state.current_state != "AWAITING_DOCUMENTS":
            raise ValueError(
                "Documents can only be submitted while the graph "
                "is AWAITING_DOCUMENTS."
            )

        if state.status != GraphStatus.WAITING:
            raise ValueError(
                f"Graph is not waiting for documents. "
                f"Current status: {state.status.value}"
            )

        state.mark_running(
            message="Vehicle documents received from the customer."
        )

        state.data["documents"] = documents

        self._checkpoint(state)

        self._transition(
            state,
            "VALIDATE_DOCUMENTS",
            "Validating the submitted vehicle documents.",
        )
        self._checkpoint(state)

        # Use Constrained ReAct LLM Agent to validate the authenticity and completeness of documents
        llm_validation = run_constrained_react(
            "Validate vehicle documents including proof of ownership, registration, and valuation",
            {"evidence": documents}
        )
        complete = llm_validation.get("is_valid", False)
        state.data["document_validation_reasoning"] = llm_validation.get("reasoning", "")

        if not complete:
            attempts = state.data.get("document_attempts", 0) + 1
            state.data["document_attempts"] = attempts

            if attempts >= self.MAX_DOCUMENT_ATTEMPTS:
                self._transition(
                    state,
                    "CLOSE",
                    "Maximum document submission attempts reached.",
                )

                state.data["final_decision"] = "REJECTED"

                state.mark_completed(
                    message=(
                        "Vehicle addition rejected because required "
                        "documents were not provided."
                    )
                )

                self._checkpoint(state)
                return state

            # Cycle back to requesting documents.
            self._transition(
                state,
                "REQUEST_VEHICLE_DOCUMENTS",
                "Submitted documents are incomplete; additional documents are required.",
            )
            self._checkpoint(state)

            self._transition(
                state,
                "AWAITING_DOCUMENTS",
                "Waiting for the missing vehicle documents.",
            )

            state.mark_waiting(
                message="Additional vehicle documents are required.",
                metadata={
                    "wait_reason": "INCOMPLETE_DOCUMENTS",
                    "attempt": attempts,
                },
            )

            self._checkpoint(state)

            return state

        # Documents are complete.
        self._transition(
            state,
            "VALUATION_PREMIUM_REVIEW",
            "Documents are complete; reviewing vehicle value and premium impact.",
        )
        self._checkpoint(state)

        return state

    # ==========================================================
    # VALUATION / PREMIUM REVIEW
    # ==========================================================

    def evaluate_valuation(
        self,
        state: GraphState,
        valuation_result: Dict[str, Any],
        premium_result: Dict[str, Any],
        value_threshold: float = DEFAULT_VALUE_THRESHOLD,
        premium_threshold: float = DEFAULT_PREMIUM_THRESHOLD,
    ) -> GraphState:
        """
        Process external valuation and premium results.

        High-value or high-premium cases require human approval.
        """

        if state.current_state != "VALUATION_PREMIUM_REVIEW":
            raise ValueError(
                f"Expected VALUATION_PREMIUM_REVIEW, "
                f"got {state.current_state}"
            )

        vehicle_value = float(
            valuation_result.get("vehicle_value", 0)
        )

        premium_change = float(
            premium_result.get("additional_premium", 0)
        )

        state.data["valuation_result"] = valuation_result
        state.data["premium_result"] = premium_result

        state.data["approval_thresholds"] = {
            "vehicle_value_threshold": value_threshold,
            "premium_change_threshold": premium_threshold,
        }

        high_value = vehicle_value > value_threshold
        high_premium = premium_change > premium_threshold

        state.data["high_value"] = high_value
        state.data["high_premium_change"] = high_premium

        if high_value or high_premium:
            self._transition(
                state,
                "HITL_APPROVAL",
                "Vehicle value or premium impact exceeds the automatic approval threshold.",
            )

            reason_parts = []

            if high_value:
                reason_parts.append(
                    f"vehicle value ${vehicle_value:,.2f} "
                    f"exceeds ${value_threshold:,.2f}"
                )

            if high_premium:
                reason_parts.append(
                    f"premium increase ${premium_change:,.2f} "
                    f"exceeds ${premium_threshold:,.2f}"
                )

            return self.hitl_manager.request_approval(
                state,
                reason="; ".join(reason_parts),
                prompt=(
                    "Admin approval is required before adding this "
                    "high-value vehicle to the policy."
                ),
                request_id=f"{state.run_id}-vehicle-approval",
            )

        # Normal case: no HITL required.
        self._transition(
            state,
            "APPLY_POLICY_UPDATE",
            "Vehicle does not require additional human approval.",
        )
        self._checkpoint(state)

        return state

    # ==========================================================
    # HITL
    # ==========================================================

    def resolve_hitl(
        self,
        state: GraphState,
        decision: str,
        decided_by: str,
        notes: str = "",
    ) -> GraphState:
        """
        Resolve the administrative approval.

        This is a genuine pause/resume boundary.
        """

        if state.current_state != "HITL_APPROVAL":
            raise ValueError(
                "HITL can only be resolved from HITL_APPROVAL."
            )

        self.hitl_manager.resolve(
            state,
            decision=decision,
            decided_by=decided_by,
            decision_notes=notes,
        )

        decision = decision.upper()

        if decision == "REJECT":
            state.data["final_decision"] = "REJECTED"

            self._transition(
                state,
                "CLOSE",
                "Admin rejected the high-value vehicle addition.",
            )

            state.mark_completed(
                message="Vehicle addition rejected by administrator."
            )

            self._checkpoint(state)
            return state

        # APPROVE
        self._transition(
            state,
            "APPLY_POLICY_UPDATE",
            "Admin approved the high-value vehicle addition.",
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # APPLY POLICY UPDATE
    # ==========================================================

    def apply_policy_update(
        self,
        state: GraphState,
        update_result: Dict[str, Any],
    ) -> GraphState:
        """
        Apply the actual policy update.

        An external/database failure can be passed to handle_failure().
        """

        if state.current_state != "APPLY_POLICY_UPDATE":
            raise ValueError(
                f"Expected APPLY_POLICY_UPDATE, got {state.current_state}"
            )

        state.data["policy_update_result"] = update_result

        self._transition(
            state,
            "CLOSE",
            "Policy update was successfully applied.",
        )

        state.data["final_decision"] = "APPROVED"

        state.mark_completed(
            message="High-value vehicle successfully added to the policy."
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # FAILURE / TICKET
    # ==========================================================

    def handle_failure(
        self,
        state: GraphState,
        error_message: str,
        ticket_id: str,
    ) -> GraphState:
        """
        Handle an unexpected failure.

        The failure is different from HITL:
        - HITL = expected business pause.
        - Failure = unexpected technical/external problem.

        The latest successful checkpoint remains available for recovery.
        """

        state.data["failure_ticket_id"] = ticket_id

        state.mark_failed(
            error_message,
            metadata={
                "ticket_id": ticket_id,
                "failure_type": "UNEXPECTED_FAILURE",
            },
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # RECOVERY
    # ==========================================================

    def recover(
        self,
        run_id: str,
    ) -> Optional[GraphState]:
        """
        Recover the graph from the latest durable checkpoint.
        """

        state = self.checkpoints.load_latest_checkpoint(run_id)

        if state is None:
            return None

        if state.status == GraphStatus.FAILED:
            state.mark_running(
                message="Run recovered from the latest checkpoint."
            )

            self._checkpoint(state)

        return state

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _transition(
        self,
        state: GraphState,
        next_state: str,
        message: str,
    ) -> None:
        state.transition_to(
            next_state,
            message=message,
        )

    def _checkpoint(
        self,
        state: GraphState,
    ) -> None:
        self.checkpoints.save_checkpoint(state)

    @staticmethod
    def _documents_are_complete(
        documents: Dict[str, Any],
    ) -> bool:
        """
        Required documents for adding a high-value vehicle.
        """

        required_documents = {
            "proof_of_ownership",
            "vehicle_registration",
            "valuation_report",
        }

        submitted = {
            key
            for key, value in documents.items()
            if value
        }

        return required_documents.issubset(submitted)