from __future__ import annotations

from typing import Any, Dict, Optional

from state_graph.models import (
    GraphState,
    GraphStatus,
    GraphEventType,
)
from state_graph.checkpointing import CheckpointManager
from state_graph.Hitl import HITLManager
from state_graph.llm_tools import run_constrained_react, call_mcp_tool_sync


class AutoInsuranceClaimGraph:
    """
    Graph 1:
    Auto Insurance Claim Investigation & Settlement.

    Harborstone Insurance workflow:

        START
          ↓
        VALIDATE_CLAIM
          ↓
        CHECK_POLICY_COVERAGE
          ↓
        REQUEST_EVIDENCE
          ↓
        AWAITING_EVIDENCE
          ↓
        VALIDATE_EVIDENCE
          ├── incomplete → REQUEST_EVIDENCE ↺
          └── complete
                  ↓
             ASSESS_CLAIM
                  ↓
          CALCULATE_SETTLEMENT
                  ↓
             ┌────┴────┐
             ↓         ↓
        normal      high amount
             ↓         ↓
           CLOSE    HITL_APPROVAL
                       ↓
                    PAUSED
                       ↓
                ADMIN DECISION
                   ↓       ↓
               APPROVE    REJECT
                  ↓          ↓
                CLOSE      CLOSE

    Unexpected failures:
        FAILURE
           ↓
        FAILED
           ↓
        Ticket created
           ↓
        Checkpoint preserved
           ↓
        Recovery / Resume
    """

    GRAPH_NAME = "AUTO_INSURANCE_CLAIM"

    # Prevent an infinite evidence-request cycle.
    MAX_EVIDENCE_ATTEMPTS = 3

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
        Start a new auto-insurance claim investigation.

        The graph intentionally stops in AWAITING_EVIDENCE
        instead of completing the workflow in one execution.
        """

        state.graph_name = self.GRAPH_NAME

        if state.current_state != "START":
            raise ValueError(
                f"Graph must start from START, "
                f"got {state.current_state}"
            )

        # ------------------------------------------------------
        # 1. Validate claim
        # ------------------------------------------------------

        self._transition(
            state,
            "VALIDATE_CLAIM",
            "Validating the submitted auto insurance claim.",
        )
        self._checkpoint(state)

        # ------------------------------------------------------
        # 2. Check policy coverage
        # ------------------------------------------------------

        self._transition(
            state,
            "CHECK_POLICY_COVERAGE",
            "Checking whether the claim is covered by the policy.",
        )
        self._checkpoint(state)

        # ------------------------------------------------------
        # 3. Request evidence
        # ------------------------------------------------------

        self._transition(
            state,
            "REQUEST_EVIDENCE",
            "Requesting accident evidence from the customer.",
        )
        self._checkpoint(state)

        # ------------------------------------------------------
        # 4. Real external wait
        # ------------------------------------------------------

        self._transition(
            state,
            "AWAITING_EVIDENCE",
            "Waiting for external claim evidence.",
        )

        state.mark_waiting(
            message=(
                "Waiting for external evidence such as accident "
                "photos, police report, or repair report."
            ),
            metadata={
                "wait_reason": "CUSTOMER_EVIDENCE",
                "external_event_required": True,
            },
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # RESUME AFTER EXTERNAL EVIDENCE
    # ==========================================================

    def resume_with_evidence(
        self,
        state: GraphState,
        evidence: Dict[str, Any],
    ) -> GraphState:
        """
        Resume the graph after an external evidence event.

        The graph does not assume that evidence will arrive.
        It remains WAITING until an external event occurs.
        """

        if state.current_state != "AWAITING_EVIDENCE":
            raise ValueError(
                "Evidence can only be submitted while "
                "the graph is AWAITING_EVIDENCE."
            )

        if state.status != GraphStatus.WAITING:
            raise ValueError(
                "Graph is not waiting for evidence. "
                f"Current status: {state.status.value}"
            )

        # Resume from the external event.
        state.mark_running(
            message="External claim evidence received."
        )

        state.data["evidence"] = evidence

        self._checkpoint(state)

        # ------------------------------------------------------
        # Validate evidence
        # ------------------------------------------------------

        self._transition(
            state,
            "VALIDATE_EVIDENCE",
            "Validating the received claim evidence.",
        )

        self._checkpoint(state)

        # Use Constrained ReAct Agent to validate evidence instead of static checks
        llm_result = run_constrained_react(
            "Validate claim evidence documents to ensure all required items are present",
            {"evidence": evidence}
        )
        complete = llm_result.get("is_valid", False)
        
        # Save LLM reasoning to state
        state.data["validation_reasoning"] = llm_result.get("reasoning", "")

        # ======================================================
        # EXTERNAL BRANCH
        # ======================================================

        if not complete:

            attempts = (
                state.data.get("evidence_attempts", 0) + 1
            )

            state.data["evidence_attempts"] = attempts

            # --------------------------------------------------
            # Cycle safety limit
            # --------------------------------------------------

            if attempts >= self.MAX_EVIDENCE_ATTEMPTS:

                self._transition(
                    state,
                    "CLOSE",
                    "Maximum evidence attempts reached.",
                )

                state.data["final_decision"] = "REJECTED"

                state.mark_completed(
                    message=(
                        "Claim rejected because required evidence "
                        f"was incomplete after "
                        f"{self.MAX_EVIDENCE_ATTEMPTS} attempts."
                    )
                )

                self._checkpoint(state)

                return state

            # --------------------------------------------------
            # Cycle:
            #
            # VALIDATE_EVIDENCE
            #       ↓
            # REQUEST_EVIDENCE
            #       ↓
            # AWAITING_EVIDENCE
            #       ↓
            # VALIDATE_EVIDENCE
            # --------------------------------------------------

            self._transition(
                state,
                "REQUEST_EVIDENCE",
                "Evidence is incomplete; additional evidence is required.",
            )

            self._checkpoint(state)

            self._transition(
                state,
                "AWAITING_EVIDENCE",
                "Waiting for the missing claim evidence.",
            )

            state.mark_waiting(
                message="Additional evidence is required.",
                metadata={
                    "wait_reason": "INCOMPLETE_EVIDENCE",
                    "attempt": attempts,
                },
            )

            self._checkpoint(state)

            return state

        # ------------------------------------------------------
        # Evidence is complete
        # ------------------------------------------------------

        self._transition(
            state,
            "ASSESS_CLAIM",
            "Evidence is complete. Assessing the claim.",
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # CLAIM ASSESSMENT
    # ==========================================================

    def assess_claim(
        self,
        state: GraphState,
        claim_details: Dict[str, Any],
    ) -> GraphState:
        """
        Assess the claim using an external policy/coverage result.

        The coverage branch should be based on deterministic
        policy information rather than an LLM inventing the result.

        semantic_store can optionally provide policy context,
        but the final `covered` value comes from the supplied
        claim/policy result.
        """

        if state.current_state != "ASSESS_CLAIM":
            raise ValueError(
                f"Expected ASSESS_CLAIM, "
                f"got {state.current_state}"
            )

        # ------------------------------------------------------
        # Optional policy context retrieval
        # ------------------------------------------------------

        policy_context = None

        if self.semantic_store is not None:
            try:
                policy_context = self.semantic_store.retrieve(
                    query=(
                        f"Auto insurance coverage for "
                        f"{claim_details.get('type', 'claim')}"
                    )
                )
            except Exception:
                # RAG/semantic retrieval must not silently decide
                # the business outcome.
                policy_context = None

        # ------------------------------------------------------
        # External/deterministic coverage result
        # ------------------------------------------------------

        # Use MCP Tool to fetch active policies for the customer
        mcp_policies = call_mcp_tool_sync(
            "get_customer_policies",
            {"customer_id": getattr(state, "customer_id", 1)}
        )

        covered = len(mcp_policies) > 0

        state.data["claim_assessment"] = {
            "covered": covered,
            "details": claim_details,
            "policy_context_available": policy_context is not None,
            "mcp_policy_result": mcp_policies
        }

        # ======================================================
        # BRANCH: COVERED / NOT COVERED
        # ======================================================

        if not covered:

            self._transition(
                state,
                "CLOSE",
                "Claim is not covered by the policy.",
            )

            state.data["final_decision"] = "REJECTED"

            state.mark_completed(
                message="Claim rejected because it is not covered."
            )

            self._checkpoint(state)

            return state

        # ------------------------------------------------------
        # Covered → settlement
        # ------------------------------------------------------

        self._transition(
            state,
            "CALCULATE_SETTLEMENT",
            "Claim is covered; calculating settlement amount.",
        )

        self._checkpoint(state)

        return state

    # ==========================================================
    # SETTLEMENT CALCULATION
    # ==========================================================

    def calculate_settlement(
        self,
        state: GraphState,
        settlement_amount: float,
        approval_threshold: float = 10000.0,
    ) -> GraphState:
        """
        Calculate the settlement amount.

        Amounts above the threshold cannot be approved
        automatically and require human approval.
        """

        if state.current_state != "CALCULATE_SETTLEMENT":
            raise ValueError(
                "Expected CALCULATE_SETTLEMENT, "
                f"got {state.current_state}"
            )

        if settlement_amount <= 0:
            raise ValueError(
                "Settlement amount must be positive."
            )

        state.data["settlement_amount"] = settlement_amount
        state.data["approval_threshold"] = approval_threshold

        # ======================================================
        # HITL BRANCH
        # ======================================================

        if settlement_amount > approval_threshold:

            self._transition(
                state,
                "HITL_APPROVAL",
                "Settlement exceeds the automatic approval threshold.",
            )

            return self.hitl_manager.request_approval(
                state,
                reason=(
                    f"Settlement amount "
                    f"${settlement_amount:,.2f} exceeds "
                    f"threshold ${approval_threshold:,.2f}."
                ),
                prompt="Senior claims officer approval is required.",
                request_id=f"{state.run_id}-settlement",
            )

        # ------------------------------------------------------
        # Normal automatic approval
        # ------------------------------------------------------

        self._transition(
            state,
            "CLOSE",
            "Settlement is below the approval threshold.",
        )

        state.data["final_decision"] = "APPROVED"

        state.mark_completed(
            message="Claim automatically approved and closed."
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
        Resume the graph after a real human decision.
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

            state.data["final_decision"] = "APPROVED"

            self._transition(
                state,
                "CLOSE",
                "Human approved the settlement.",
            )

            state.mark_completed(
                message="Claim approved by human reviewer."
            )

        else:

            state.data["final_decision"] = "REJECTED"

            self._transition(
                state,
                "CLOSE",
                "Human rejected the settlement.",
            )

            state.mark_completed(
                message="Claim rejected by human reviewer."
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
        Handle an unexpected failure.

        Failure is different from HITL:
        - HITL = expected business pause.
        - Failure = unexpected technical/external problem.

        The current state is preserved through checkpointing,
        allowing later recovery.
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

    def recover(self, run_id: str) -> Optional[GraphState]:
        """
        Recover the latest durable checkpoint.

        This does NOT restart the graph from START.
        It restores the exact last persisted state.
        """

        state = self.checkpoints.load_latest_checkpoint(run_id)

        if state is None:
            return None

        return state

    # ==========================================================
    # RESUME AFTER FAILURE
    # ==========================================================

    def resume_after_failure(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Resume execution from the last checkpoint after
        the external/technical failure has been resolved.

        The graph does not restart from START.
        """

        if state.status != GraphStatus.FAILED:
            raise ValueError(
                "Only FAILED runs can be resumed through "
                "resume_after_failure()."
            )

        state.mark_running(
            message=(
                "Failure resolved; resuming from the "
                "latest checkpoint."
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

    @staticmethod
    def _evidence_is_complete(
        evidence: Dict[str, Any],
    ) -> bool:
        """
        Required evidence for the demo claim workflow:
            - accident photos
            - police report
            - repair report
        """

        required = {
            "accident_photos",
            "police_report",
            "repair_report",
        }

        provided = {
            key
            for key, value in evidence.items()
            if value
        }

        return required.issubset(provided)