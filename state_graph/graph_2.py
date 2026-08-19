from __future__ import annotations

from typing import Any, Dict, Optional

from state_graph.models import (
    GraphState,
    GraphStatus,
    GraphEventType,
)
from state_graph.checkpointing import CheckpointManager
from state_graph.Hitl import HITLManager
from state_graph.llm_tools import run_tree_of_thoughts, call_mcp_tool_sync


class PolicyCancellationGraph:
    """
    Graph 2:
    Policy Cancellation / Renewal Retention Workflow.

    Harborstone Insurance workflow:

        START
          ↓
        VERIFY_CUSTOMER
          ↓
        GET_POLICY
          ↓
        CHECK_CANCELLATION_RULES
          ↓
        CALCULATE_FINANCIAL_IMPACT
          ↓
        PRESENT_OPTIONS
          ↓
        AWAITING_CUSTOMER_DECISION
             │
             ├── CONFIRM_CANCEL
             │       ↓
             │   REVIEW_CANCELLATION
             │       ↓
             │   Refund threshold?
             │       ├── NO  → APPLY_CANCELLATION → CLOSE
             │       └── YES → HITL_APPROVAL
             │
             ├── KEEP_POLICY → CLOSE
             │
             ├── MODIFY_POLICY
             │       ↓
             │   GENERATE_REVISED_TERMS
             │       ↓
             │   AWAITING_CUSTOMER_REVIEW
             │       ├── ACCEPT → APPLY_UPDATE → CLOSE
             │       └── REJECT → PRESENT_OPTIONS ↺
             │
             └── NO_RESPONSE
                     ↓
                  FOLLOW_UP
                     ↓
             AWAITING_CUSTOMER_DECISION ↺

    Unexpected failures are persisted and can be recovered
    from the latest checkpoint.
    """

    GRAPH_NAME = "POLICY_CANCELLATION_RETENTION"

    MAX_NO_RESPONSE_FOLLOWUPS = 3
    MAX_MODIFICATION_CYCLES = 3

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        semantic_store: Optional[Any] = None,
    ) -> None:
        self.checkpoints = checkpoint_manager
        self.semantic_store = semantic_store
        self.hitl_manager = HITLManager(checkpoint_manager)

    # ==========================================================
    # START
    # ==========================================================

    def start(self, state: GraphState) -> GraphState:
        """
        Start the cancellation/retention workflow.

        The workflow intentionally stops at
        AWAITING_CUSTOMER_DECISION.
        """

        state.graph_name = self.GRAPH_NAME

        if state.current_state != "START":
            raise ValueError(
                f"Graph must start from START, "
                f"got {state.current_state}"
            )

        # ------------------------------------------------------
        # 1. Verify customer
        # ------------------------------------------------------

        self._transition(
            state,
            "VERIFY_CUSTOMER",
            "Verifying the customer requesting policy cancellation.",
        )
        self._checkpoint(state)

        # ------------------------------------------------------
        # 2. Load policy
        # ------------------------------------------------------

        self._transition(
            state,
            "GET_POLICY",
            "Loading the active insurance policy.",
        )
        self._checkpoint(state)

        # ------------------------------------------------------
        # 3. Check cancellation rules
        # ------------------------------------------------------

        self._transition(
            state,
            "CHECK_CANCELLATION_RULES",
            "Checking policy cancellation and renewal rules.",
        )
        
        # Use MCP Tool to apply cancellation rules
        mcp_rules = call_mcp_tool_sync(
            "apply_cancellation_rules",
            {"policy_id": getattr(state, "policy_id", 0)}
        )
        
        # Use RAG (Semantic Store) to retrieve the official cancellation policy text
        policy_context = None
        if self.semantic_store:
            try:
                policy_context = self.semantic_store.retrieve(
                    query="What are the cancellation and refund rules for marine policies?"
                )
            except Exception:
                pass
                
        state.data["cancellation_rules"] = mcp_rules
        state.data["cancellation_policy_context"] = policy_context

        self._checkpoint(state)

        # ------------------------------------------------------
        # 4. Calculate financial impact
        # ------------------------------------------------------

        self._transition(
            state,
            "CALCULATE_FINANCIAL_IMPACT",
            "Calculating refund or outstanding balance.",
        )
        self._checkpoint(state)

        # ------------------------------------------------------
        # 5. Present customer options
        # ------------------------------------------------------

        self._transition(
            state,
            "PRESENT_OPTIONS",
            "Presenting cancellation and retention options.",
        )
        
        # Use Tree of Thoughts LLM pattern to evaluate and present the best retention options
        tot_options = run_tree_of_thoughts(
            "Determine the best retention options for the customer based on their history",
            {"customer_id": getattr(state, "customer_id", 0), "policy_id": getattr(state, "policy_id", 0)}
        )
        state.data["generated_retention_options"] = tot_options
        
        self._checkpoint(state)

        # ------------------------------------------------------
        # 6. REAL EXTERNAL WAIT
        # ------------------------------------------------------

        self._transition(
            state,
            "AWAITING_CUSTOMER_DECISION",
            "Waiting for the customer's decision.",
        )

        state.mark_waiting(
            message=(
                "Waiting for the customer to choose whether "
                "to cancel, keep, or modify the policy."
            ),
            metadata={
                "wait_reason": "CUSTOMER_DECISION",
                "external_event_required": True,
                "allowed_decisions": [
                    "CONFIRM_CANCEL",
                    "KEEP_POLICY",
                    "MODIFY_POLICY",
                ],
            },
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # CUSTOMER DECISION
    # ==========================================================

    def receive_customer_decision(
        self,
        state: GraphState,
        decision: str,
    ) -> GraphState:
        """
        Resume the workflow when the customer responds.

        This is the external event that releases the
        AWAITING_CUSTOMER_DECISION state.
        """

        if state.current_state != "AWAITING_CUSTOMER_DECISION":
            raise ValueError(
                "Customer decision can only be received while "
                "the graph is AWAITING_CUSTOMER_DECISION."
            )

        if state.status != GraphStatus.WAITING:
            raise ValueError(
                "Graph is not waiting for a customer decision. "
                f"Current status: {state.status.value}"
            )

        decision = decision.upper()

        state.mark_running(
            message="Customer decision received."
        )

        state.data["customer_decision"] = decision

        self._checkpoint(state)

        # ======================================================
        # EXTERNAL BRANCH
        # ======================================================

        if decision == "CONFIRM_CANCEL":

            self._transition(
                state,
                "REVIEW_CANCELLATION",
                "Customer confirmed policy cancellation.",
            )
            self._checkpoint(state)

            return state

        if decision == "KEEP_POLICY":

            self._transition(
                state,
                "CLOSE",
                "Customer chose to keep the existing policy.",
            )

            state.data["final_decision"] = "POLICY_RETAINED"

            state.mark_completed(
                message="Existing policy retained."
            )

            self._checkpoint(state)

            return state

        if decision == "MODIFY_POLICY":

            self._transition(
                state,
                "GENERATE_REVISED_TERMS",
                "Customer requested modified policy terms.",
            )

            self._checkpoint(state)

            return state

        if decision == "NO_RESPONSE":

            return self._handle_no_response(state)

        raise ValueError(
            "Unsupported customer decision. "
            "Expected CONFIRM_CANCEL, KEEP_POLICY, "
            "MODIFY_POLICY, or NO_RESPONSE."
        )

    # ==========================================================
    # NO RESPONSE / FOLLOW-UP CYCLE
    # ==========================================================

    def _handle_no_response(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Handle a customer who has not responded.

        This creates the real waiting cycle:

            AWAITING_CUSTOMER_DECISION
                    ↓
                FOLLOW_UP
                    ↓
            AWAITING_CUSTOMER_DECISION
                    ↺
        """

        attempts = (
            state.data.get("no_response_followups", 0) + 1
        )

        state.data["no_response_followups"] = attempts

        if attempts >= self.MAX_NO_RESPONSE_FOLLOWUPS:

            self._transition(
                state,
                "CLOSE",
                "Maximum customer follow-ups reached.",
            )

            state.data["final_decision"] = "NO_RESPONSE"

            state.mark_completed(
                message=(
                    "Workflow closed after maximum "
                    "customer follow-ups."
                )
            )

            self._checkpoint(state)

            return state

        self._transition(
            state,
            "FOLLOW_UP",
            "Customer did not respond; sending follow-up.",
        )

        self._checkpoint(state)

        self._transition(
            state,
            "AWAITING_CUSTOMER_DECISION",
            "Waiting for customer response after follow-up.",
        )

        state.mark_waiting(
            message="Waiting for customer response after follow-up.",
            metadata={
                "wait_reason": "CUSTOMER_DECISION",
                "followup_attempt": attempts,
            },
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # CANCELLATION REVIEW
    # ==========================================================

    def review_cancellation(
        self,
        state: GraphState,
        cancellation_result: Dict[str, Any],
        refund_threshold: float = 5000.0,
    ) -> GraphState:
        """
        Review cancellation eligibility and financial impact.

        cancellation_result should contain external/policy results,
        for example:

            {
                "allowed": True,
                "refund_amount": 2500.00
            }
        """

        if state.current_state != "REVIEW_CANCELLATION":
            raise ValueError(
                f"Expected REVIEW_CANCELLATION, "
                f"got {state.current_state}"
            )

        allowed = bool(
            cancellation_result.get("allowed", False)
        )

        refund_amount = float(
            cancellation_result.get("refund_amount", 0)
        )

        state.data["cancellation_result"] = cancellation_result
        state.data["refund_amount"] = refund_amount
        state.data["refund_threshold"] = refund_threshold

        # ------------------------------------------------------
        # Cancellation not allowed
        # ------------------------------------------------------

        if not allowed:

            self._transition(
                state,
                "CLOSE",
                "Cancellation is not allowed under policy rules.",
            )

            state.data["final_decision"] = (
                "CANCELLATION_REJECTED"
            )

            state.mark_completed(
                message=(
                    "Cancellation rejected because "
                    "policy rules do not allow it."
                )
            )

            self._checkpoint(state)

            return state

        # ======================================================
        # HITL BRANCH
        # ======================================================

        if refund_amount > refund_threshold:

            self._transition(
                state,
                "HITL_APPROVAL",
                "Refund exceeds the human approval threshold.",
            )

            return self.hitl_manager.request_approval(
                state,
                reason=(
                    f"Cancellation refund "
                    f"${refund_amount:,.2f} exceeds "
                    f"threshold ${refund_threshold:,.2f}."
                ),
                prompt="Senior policy officer approval is required.",
                request_id=f"{state.run_id}-cancellation",
            )

        # ------------------------------------------------------
        # Normal cancellation
        # ------------------------------------------------------

        self._transition(
            state,
            "APPLY_CANCELLATION",
            "Cancellation is allowed without human approval.",
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # HITL RESOLUTION
    # ==========================================================

    def resolve_hitl(
        self,
        state: GraphState,
        decision: str,
        decided_by: str,
        notes: str = "",
    ) -> GraphState:
        """
        Resolve a human approval request.
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

        # ======================================================
        # HUMAN BRANCH
        # ======================================================

        if decision == "APPROVE":

            self._transition(
                state,
                "APPLY_CANCELLATION",
                "Human approved the cancellation refund.",
            )

            self._checkpoint(state)

            return state

        # Human rejected the cancellation.

        self._transition(
            state,
            "CLOSE",
            "Human rejected the cancellation request.",
        )

        state.data["final_decision"] = (
            "CANCELLATION_REJECTED"
        )

        state.mark_completed(
            message="Cancellation rejected by human reviewer."
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # APPLY CANCELLATION
    # ==========================================================

    def apply_cancellation(
        self,
        state: GraphState,
        update_result: Optional[Dict[str, Any]] = None,
    ) -> GraphState:
        """
        Apply the cancellation using the policy/update layer.

        update_result represents the external policy-system result.
        """

        if state.current_state != "APPLY_CANCELLATION":
            raise ValueError(
                f"Expected APPLY_CANCELLATION, "
                f"got {state.current_state}"
            )

        if update_result:
            state.data["cancellation_update"] = update_result

        self._transition(
            state,
            "CLOSE",
            "Policy cancellation successfully applied.",
        )

        state.data["final_decision"] = "CANCELLED"

        state.mark_completed(
            message="Policy cancellation completed."
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # MODIFIED POLICY TERMS
    # ==========================================================

    def generate_revised_terms(
        self,
        state: GraphState,
        revised_terms: Dict[str, Any],
    ) -> GraphState:
        """
        Generate and present revised policy terms.
        """

        if state.current_state != "GENERATE_REVISED_TERMS":
            raise ValueError(
                "Expected GENERATE_REVISED_TERMS."
            )

        cycles = (
            state.data.get("modification_cycles", 0) + 1
        )

        state.data["modification_cycles"] = cycles
        state.data["revised_terms"] = revised_terms

        self._transition(
            state,
            "AWAITING_CUSTOMER_REVIEW",
            "Waiting for customer to review revised policy terms.",
        )

        state.mark_waiting(
            message=(
                "Waiting for customer to accept or reject "
                "the revised policy terms."
            ),
            metadata={
                "wait_reason": "CUSTOMER_REVIEW",
                "modification_cycle": cycles,
            },
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # CUSTOMER REVIEW OF MODIFIED TERMS
    # ==========================================================

    def receive_terms_decision(
        self,
        state: GraphState,
        decision: str,
    ) -> GraphState:
        """
        Receive the customer's decision about revised terms.
        """

        if state.current_state != "AWAITING_CUSTOMER_REVIEW":
            raise ValueError(
                "Customer terms decision can only be received "
                "from AWAITING_CUSTOMER_REVIEW."
            )

        if state.status != GraphStatus.WAITING:
            raise ValueError(
                "Graph is not waiting for customer review."
            )

        decision = decision.upper()

        state.mark_running(
            message="Customer reviewed the revised policy terms."
        )

        state.data["terms_decision"] = decision

        self._checkpoint(state)

        # ------------------------------------------------------
        # ACCEPT
        # ------------------------------------------------------

        if decision == "ACCEPT":

            self._transition(
                state,
                "APPLY_UPDATE",
                "Customer accepted the revised policy terms.",
            )

            self._checkpoint(state)

            return state

        # ------------------------------------------------------
        # REJECT → cycle back to options
        # ------------------------------------------------------

        if decision == "REJECT":

            cycles = state.data.get(
                "modification_cycles",
                0,
            )

            if cycles >= self.MAX_MODIFICATION_CYCLES:

                self._transition(
                    state,
                    "CLOSE",
                    "Maximum policy modification cycles reached.",
                )

                state.data["final_decision"] = (
                    "RETENTION_FAILED"
                )

                state.mark_completed(
                    message=(
                        "Workflow closed after maximum "
                        "policy modification cycles."
                    )
                )

                self._checkpoint(state)

                return state

            self._transition(
                state,
                "PRESENT_OPTIONS",
                "Customer rejected revised terms; "
                "presenting options again.",
            )

            self._checkpoint(state)

            self._transition(
                state,
                "AWAITING_CUSTOMER_DECISION",
                "Waiting for the customer's next decision.",
            )

            state.mark_waiting(
                message=(
                    "Waiting for another customer decision "
                    "after revised terms were rejected."
                ),
                metadata={
                    "wait_reason": "CUSTOMER_DECISION",
                    "modification_cycle": cycles,
                },
            )

            self._checkpoint(state)

            return state

        raise ValueError(
            "Terms decision must be ACCEPT or REJECT."
        )

    # ==========================================================
    # APPLY POLICY UPDATE
    # ==========================================================

    def apply_update(
        self,
        state: GraphState,
        update_result: Optional[Dict[str, Any]] = None,
    ) -> GraphState:
        """
        Apply the modified policy.
        """

        if state.current_state != "APPLY_UPDATE":
            raise ValueError(
                f"Expected APPLY_UPDATE, "
                f"got {state.current_state}"
            )

        if update_result:
            state.data["policy_update"] = update_result

        self._transition(
            state,
            "CLOSE",
            "Modified policy successfully applied.",
        )

        state.data["final_decision"] = "POLICY_MODIFIED"

        state.mark_completed(
            message="Policy successfully modified and retained."
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # FAILURE
    # ==========================================================

    def handle_failure(
        self,
        state: GraphState,
        error_message: str,
        ticket_id: str,
    ) -> GraphState:
        """
        Handle an unexpected technical/external failure.

        This is different from HITL:
            HITL    = expected business decision.
            FAILURE = unexpected system/external problem.
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
        Recover the latest durable checkpoint.

        The workflow does not restart from START.
        """

        return self.checkpoints.load_latest_checkpoint(run_id)

    # ==========================================================
    # RESUME AFTER FAILURE
    # ==========================================================

    def resume_after_failure(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Resume a failed workflow from its latest checkpoint.
        """

        if state.status != GraphStatus.FAILED:
            raise ValueError(
                "Only FAILED runs can be resumed."
            )

        state.mark_running(
            message=(
                "Failure resolved; resuming from "
                "the latest checkpoint."
            )
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