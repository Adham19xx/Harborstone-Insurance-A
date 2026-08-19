from uuid import uuid4

from state_graph.models import GraphState
from state_graph.checkpointing import CheckpointManager


def main():
    run_id = f"test-run-{uuid4()}"

    manager = CheckpointManager()

    # Start a new graph run
    state = GraphState(
        run_id=run_id,
        graph_name="marine_claim",
        claim_id=3,
    )

    print("Creating run...")
    manager.create_run(state)

    print(
        f"Created checkpoint version: "
        f"{state.checkpoint_version}"
    )

    # Simulate a meaningful transition
    state.transition_to(
        "LOAD_CLAIM",
        message="Claim loading started",
    )

    manager.save_checkpoint(state)

    print(
        f"Saved checkpoint version: "
        f"{state.checkpoint_version}"
    )

    # Simulate another meaningful transition
    state.transition_to(
        "INVESTIGATE",
        message="Claim investigation started",
    )

    manager.save_checkpoint(state)

    print(
        f"Saved checkpoint version: "
        f"{state.checkpoint_version}"
    )

    # Simulate crash/restart:
    # The original state object is no longer used.
    print("\nSimulating application restart...")

    restored_state = manager.load_latest_checkpoint(run_id)

    if restored_state is None:
        print("No checkpoint found.")
        return

    print("\nRestored state:")
    print(f"Run ID: {restored_state.run_id}")
    print(f"Graph: {restored_state.graph_name}")
    print(f"Current state: {restored_state.current_state}")
    print(f"Status: {restored_state.status.value}")
    print(f"Checkpoint version: {restored_state.checkpoint_version}")


if __name__ == "__main__":
    main()