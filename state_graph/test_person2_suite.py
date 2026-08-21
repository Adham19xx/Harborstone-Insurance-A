"""
Comprehensive Person 2 Test Suite for Harborstone State Graphs,
Failure/Ticket Lifecycle, MCP/LLM Integration, and Recovery.
"""
import pytest
from typing import Any, Dict
from uuid import uuid4

from state_graph.models import GraphState, GraphStatus
from state_graph.checkpointing import CheckpointManager
from state_graph.failure_tickets import TicketManager, FailureTicket, TicketStatus, FailureType
from state_graph.Hitl import HITLManager
from state_graph.graph_1 import AutoInsuranceClaimGraph
from state_graph.graph_2 import PolicyCancellationGraph
from state_graph.graph_3 import HighValueVehicleAdditionGraph
from state_graph.llm_tools import (
    inject_llm_failure,
    inject_mcp_failure,
    set_injected_mcp_executor,
    run_constrained_react,
    run_tree_of_thoughts,
    run_task_decomposition,
)
from rag.policy_retriever import PolicyRAGRetriever


# ---------------------------------------------------------------------------
# Test Fixtures & Mock LLMs
# ---------------------------------------------------------------------------
class MockLLM:
    """Mock LLM returning structured outputs for state graph testing."""
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.invocations = []

    def invoke(self, messages, **kwargs):
        if self.should_fail:
            raise RuntimeError("Injected LLM invocation failure")
        self.invocations.append(messages)
        class MockResp:
            content = "Validated evidence strictly against provided constraints."
        return MockResp()

    def with_structured_output(self, schema, **kwargs):
        if self.should_fail:
            raise RuntimeError("Injected LLM structured output failure")
        class FakeStructured:
            def invoke(s, messages, **kw):
                self.invocations.append(messages)
                schema_name = getattr(schema, "__name__", str(schema))
                if "ThoughtCandidates" in schema_name:
                    class ThoughtObj:
                        candidates = ["Option A: 10% loyalty discount", "Option B: Increase deductible"]
                    return ThoughtObj()
                elif "ThoughtEvaluation" in schema_name:
                    class EvalObj:
                        score = 0.9
                        rationale = "Compliant"
                    return EvalObj()
                elif "VehicleDecompositionResult" in schema_name:
                    class VehicleObj:
                        vehicle_type = "Yacht"
                        declared_value = 250000.0
                        year_built = 2023
                        manufacturer = "Azimut"
                        model = "Grande"
                        subtasks = ["verify", "check_eligibility"]
                        def model_dump(self):
                            return {
                                "vehicle_type": self.vehicle_type,
                                "declared_value": self.declared_value,
                                "year_built": self.year_built,
                                "manufacturer": self.manufacturer,
                                "model": self.model,
                                "subtasks": self.subtasks,
                            }
                    return VehicleObj()
                return None
        return FakeStructured()


@pytest.fixture(autouse=True)
def clean_injections():
    inject_llm_failure(None)
    inject_mcp_failure(None)
    set_injected_mcp_executor(None)
    yield
    inject_llm_failure(None)
    inject_mcp_failure(None)
    set_injected_mcp_executor(None)


# ===========================================================================
# 1. TICKET LIFECYCLE TESTS
# ===========================================================================
def test_ticket_lifecycle_transitions():
    tm = TicketManager(in_memory=True)
    ticket = tm.create_ticket(
        run_id="run-100",
        graph_name="AUTO_INSURANCE_CLAIM",
        failed_node="VALIDATE_EVIDENCE",
        failure_type=FailureType.MCP_FAILURE,
        error_message="Connection refused to MCP socket",
        checkpoint_version=3,
        metadata={"step": "mcp_call"},
    )

    assert ticket.status == TicketStatus.OPEN
    assert ticket.recovery_attempts == 0
    assert ticket.failed_node == "VALIDATE_EVIDENCE"
    assert ticket.failure_type == FailureType.MCP_FAILURE
    assert ticket.checkpoint_version == 3

    # Transition to INVESTIGATING
    investigating_ticket = tm.start_investigation(ticket.ticket_id)
    assert investigating_ticket.status == TicketStatus.INVESTIGATING
    assert investigating_ticket.recovery_attempts == 1

    # Transition to RESOLVED
    resolved_ticket = tm.resolve_ticket(ticket.ticket_id, resolution_note="MCP connection restored")
    assert resolved_ticket.status == TicketStatus.RESOLVED
    assert resolved_ticket.resolution_note == "MCP connection restored"


# ===========================================================================
# 2. MCP FAILURE SIMULATION & TICKET CREATION
# ===========================================================================
def test_mcp_failure_handling_and_ticket_creation():
    cm = CheckpointManager(in_memory=True)
    tm = TicketManager(in_memory=True)
    graph = AutoInsuranceClaimGraph(checkpoint_manager=cm, ticket_manager=tm)

    state = GraphState(run_id=f"test-mcp-fail-{uuid4()}", graph_name=graph.GRAPH_NAME)
    cm.create_run(state)
    graph.start(state)

    # Simulate MCP failure
    inject_mcp_failure("get_customer_policies")
    state.transition_to("ASSESS_CLAIM", message="Assessing claim")
    cm.save_checkpoint(state)

    failed_state = None
    try:
        from state_graph.llm_tools import call_mcp_tool_sync
        call_mcp_tool_sync("get_customer_policies", {"customer_id": 1})
    except Exception as exc:
        failed_state = graph.handle_failure(
            state,
            error_message=str(exc),
            failure_type=FailureType.MCP_FAILURE,
            failed_node="ASSESS_CLAIM",
        )

    assert failed_state is not None
    assert failed_state.status == GraphStatus.FAILED
    assert "failure_ticket_id" in failed_state.data

    ticket = tm.get_ticket(failed_state.data["failure_ticket_id"])
    assert ticket is not None
    assert ticket.status == TicketStatus.OPEN
    assert ticket.run_id == state.run_id
    assert ticket.failed_node == "ASSESS_CLAIM"
    assert ticket.failure_type == FailureType.MCP_FAILURE
    assert ticket.checkpoint_version == 6



# ===========================================================================
# 3. LLM FAILURE SIMULATION
# ===========================================================================
def test_llm_failure_handling_and_recovery():
    cm = CheckpointManager(in_memory=True)
    tm = TicketManager(in_memory=True)
    graph = PolicyCancellationGraph(checkpoint_manager=cm, ticket_manager=tm)

    state = GraphState(run_id=f"test-llm-fail-{uuid4()}", graph_name=graph.GRAPH_NAME, customer_id=1, policy_id=101)
    cm.create_run(state)

    # Simulate LLM failure in ToT
    try:
        inject_llm_failure("tot_generation")
        run_tree_of_thoughts("Generate options", {"policy_id": 101})
    except Exception as exc:
        failed_state = graph.handle_failure(
            state,
            error_message=str(exc),
            failure_type=FailureType.LLM_FAILURE,
            failed_node="PRESENT_OPTIONS",
        )

    assert failed_state.status == GraphStatus.FAILED
    ticket = tm.get_ticket(failed_state.data["failure_ticket_id"])
    assert ticket.status == TicketStatus.OPEN
    assert ticket.failure_type == FailureType.LLM_FAILURE

    # Recover
    inject_llm_failure(None)
    resumed_state = graph.resume_after_failure(failed_state)
    assert resumed_state.status == GraphStatus.RUNNING
    resumed_ticket = tm.get_ticket(ticket.ticket_id)
    assert resumed_ticket.status == TicketStatus.INVESTIGATING
    assert resumed_ticket.recovery_attempts == 1


# ===========================================================================
# 4. RECOVERY RESUMES FROM CHECKPOINT (NOT FROM START)
# ===========================================================================
def test_recovery_resumes_from_checkpoint_not_start():
    cm = CheckpointManager(in_memory=True)
    tm = TicketManager(in_memory=True)
    graph = AutoInsuranceClaimGraph(checkpoint_manager=cm, ticket_manager=tm)

    state = GraphState(run_id=f"test-resume-chk-{uuid4()}", graph_name=graph.GRAPH_NAME)
    cm.create_run(state)
    graph.start(state)

    assert state.current_state == "AWAITING_EVIDENCE"

    # Advance with evidence to VALIDATE_EVIDENCE -> ASSESS_CLAIM
    graph.resume_with_evidence(
        state,
        evidence={"accident_photos": "photo.jpg", "police_report": "report.pdf", "repair_report": "repair.pdf"}
    )
    assert state.current_state == "ASSESS_CLAIM"
    chk_ver_at_assess = state.checkpoint_version

    # Simulate crash / failure at ASSESS_CLAIM
    failed_state = graph.handle_failure(
        state,
        error_message="Network glitch during settlement computation",
        failure_type=FailureType.UNEXPECTED_ERROR,
    )
    assert failed_state.status == GraphStatus.FAILED

    # Recover state
    recovered_state = graph.recover(state.run_id)
    assert recovered_state is not None
    assert recovered_state.current_state == "ASSESS_CLAIM"
    assert recovered_state.checkpoint_version >= chk_ver_at_assess
    assert recovered_state.current_state != "START"

    # Resume
    resumed_state = graph.resume_after_failure(recovered_state)
    assert resumed_state.status == GraphStatus.RUNNING
    assert resumed_state.current_state == "ASSESS_CLAIM"


# ===========================================================================
# 5. PERSISTENT RECOVERY SIMULATION (PROCESS RESTART)
# ===========================================================================
def test_persistent_recovery_across_fresh_manager_instances():
    cm1 = CheckpointManager(in_memory=True)
    tm1 = TicketManager(in_memory=True)
    graph1 = HighValueVehicleAdditionGraph(checkpoint_manager=cm1, ticket_manager=tm1)

    run_id = f"test-restart-{uuid4()}"
    state = GraphState(run_id=run_id, graph_name=graph1.GRAPH_NAME)
    cm1.create_run(state)
    graph1.start(state, vehicle_details={"type": "Yacht", "value": 350000.0, "year": 2024})

    assert state.current_state == "CHECK_ELIGIBILITY"
    latest_v = state.checkpoint_version

    # Simulate total crash: create new manager and graph instances sharing store
    cm2 = CheckpointManager(in_memory=True)
    cm2._runs_memory = cm1._runs_memory
    cm2._checkpoints_memory = cm1._checkpoints_memory

    tm2 = TicketManager(in_memory=True)
    tm2._memory_store = tm1._memory_store

    graph2 = HighValueVehicleAdditionGraph(checkpoint_manager=cm2, ticket_manager=tm2)
    restored_state = graph2.recover(run_id)

    assert restored_state is not None
    assert restored_state.run_id == run_id
    assert restored_state.current_state == "CHECK_ELIGIBILITY"
    assert restored_state.checkpoint_version == latest_v
    assert "vehicle" in restored_state.data


# ===========================================================================
# 6. RECOVERY FAILURE REPEATED
# ===========================================================================
def test_recovery_failure_increments_attempts_and_preserves_state():
    cm = CheckpointManager(in_memory=True)
    tm = TicketManager(in_memory=True)
    graph = AutoInsuranceClaimGraph(checkpoint_manager=cm, ticket_manager=tm)

    state = GraphState(run_id=f"test-repeat-fail-{uuid4()}", graph_name=graph.GRAPH_NAME)
    cm.create_run(state)
    graph.start(state)

    # First failure
    graph.handle_failure(state, "Initial error", failure_type=FailureType.MCP_FAILURE)
    ticket_id = state.data["failure_ticket_id"]

    # First recovery attempt
    graph.resume_after_failure(state)
    t1 = tm.get_ticket(ticket_id)
    assert t1.status == TicketStatus.INVESTIGATING
    assert t1.recovery_attempts == 1

    # Second failure during recovery
    graph.handle_failure(state, "Secondary failure during retry", failure_type=FailureType.MCP_FAILURE)
    tm.record_recovery_failure(ticket_id, "Secondary failure during retry")
    
    # Second recovery attempt
    graph.resume_after_failure(state)
    t2 = tm.get_ticket(ticket_id)
    assert t2.status == TicketStatus.INVESTIGATING
    assert t2.recovery_attempts == 2


# ===========================================================================
# 7. GRAPH TECHNIQUE INVOCATION TESTS
# ===========================================================================
def test_graph_1_invokes_constrained_react_and_rag():
    cm = CheckpointManager(in_memory=True)
    mock_llm = MockLLM()
    rag = PolicyRAGRetriever()
    graph = AutoInsuranceClaimGraph(checkpoint_manager=cm, rag_retriever=rag, llm=mock_llm)

    state = GraphState(run_id=f"test-g1-tech-{uuid4()}", graph_name=graph.GRAPH_NAME)
    cm.create_run(state)
    graph.start(state)

    # 1. Constrained ReAct invoked during evidence validation
    graph.resume_with_evidence(
        state,
        evidence={"accident_photos": "photo.jpg", "police_report": "report.pdf", "repair_report": "repair.pdf"}
    )
    assert "validation_reasoning" in state.data
    assert "verified_evidence" in state.data
    assert len(state.data["verified_evidence"]) == 3

    # 2. RAG invoked during claim assessment
    graph.assess_claim(state, claim_details={"type": "collision", "amount": 4000.0})
    assert state.data["claim_assessment"]["policy_context_available"] is True


def test_graph_2_invokes_tot_and_rag():
    cm = CheckpointManager(in_memory=True)
    mock_llm = MockLLM()
    rag = PolicyRAGRetriever()
    graph = PolicyCancellationGraph(checkpoint_manager=cm, rag_retriever=rag, llm=mock_llm)

    state = GraphState(run_id=f"test-g2-tech-{uuid4()}", graph_name=graph.GRAPH_NAME, customer_id=1, policy_id=101)
    cm.create_run(state)
    graph.start(state)

    # 1. RAG policy context retrieved
    assert state.data.get("cancellation_policy_context") is not None

    # 2. Tree of Thoughts generated retention options
    assert "generated_retention_options" in state.data
    options = state.data["generated_retention_options"]
    assert isinstance(options, list)
    assert len(options) >= 2


def test_graph_3_invokes_task_decomposition_and_constrained_react():
    cm = CheckpointManager(in_memory=True)
    mock_llm = MockLLM()
    graph = HighValueVehicleAdditionGraph(checkpoint_manager=cm, llm=mock_llm)

    state = GraphState(run_id=f"test-g3-tech-{uuid4()}", graph_name=graph.GRAPH_NAME)
    cm.create_run(state)

    # 1. Task Decomposition invoked in start()
    graph.start(state, vehicle_details={"text": "Add 2024 Azimut luxury yacht valued at $650,000 to policy"})
    assert "extracted_vehicle_details" in state.data
    extracted = state.data["extracted_vehicle_details"]
    assert "vehicle_type" in extracted

    # Advance to documents request
    graph.evaluate_eligibility(state, eligibility_result={"eligible": True, "reasons": []})
    assert state.current_state == "AWAITING_DOCUMENTS"

    # 2. Constrained ReAct invoked in resume_with_documents()
    graph.resume_with_documents(
        state,
        documents={
            "proof_of_ownership": "invoice.pdf",
            "vehicle_registration": "reg.pdf",
            "valuation_report": "survey.pdf",
        }
    )
    assert "document_validation_reasoning" in state.data
    assert "verified_documents" in state.data
    assert len(state.data["verified_documents"]) == 3



# ===========================================================================
# 8. HITL PAUSE PRESERVED (NOT CONVERTED TO FAILURE)
# ===========================================================================
def test_hitl_pause_is_not_treated_as_failure():
    cm = CheckpointManager(in_memory=True)
    tm = TicketManager(in_memory=True)
    graph = AutoInsuranceClaimGraph(checkpoint_manager=cm, ticket_manager=tm)

    state = GraphState(run_id=f"test-hitl-{uuid4()}", graph_name=graph.GRAPH_NAME)
    cm.create_run(state)
    graph.start(state)

    graph.resume_with_evidence(
        state,
        evidence={"accident_photos": "photo.jpg", "police_report": "report.pdf", "repair_report": "repair.pdf"}
    )
    graph.assess_claim(state, claim_details={"type": "collision", "amount": 15000.0})
    
    # Calculate settlement with high amount -> triggers HITL pause
    graph.calculate_settlement(state, settlement_amount=15000.0)

    assert state.status == GraphStatus.PAUSED_FOR_HITL
    assert state.hitl_request is not None
    assert state.data["settlement_amount"] == 15000.0

    # Verify NO failure tickets were created for HITL pause
    tickets = [t for t in tm._memory_store.values() if t.run_id == state.run_id]
    assert len(tickets) == 0

    # Resolve HITL cleanly
    graph.resolve_hitl(state, decision="APPROVE", decided_by="Claims_Manager_1")
    assert state.status == GraphStatus.COMPLETED
    assert state.data["final_decision"] == "APPROVED"


