from __future__ import annotations

import json
from typing import Optional

try:
    import mysql.connector
except ImportError:
    mysql = None


from state_graph.models import GraphState


class CheckpointManager:
    """
    Persist and restore State Graph runs using the existing
    Harborstone Insurance MySQL database, with transparent
    in-memory storage fallback for local/unit testing.
    """

    def __init__(
        self,
        host: str = "localhost",
        user: str = "root",
        password: str = "",
        database: str = "harborstone_insurance",
        in_memory: bool = False,
    ) -> None:
        self.in_memory = in_memory
        self.db_config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
        }
        self._runs_memory: dict[str, dict] = {}
        self._checkpoints_memory: dict[str, list[dict]] = {}

    def _get_connection(self):
        if self.in_memory:
            return None
        try:
            return mysql.connector.connect(**self.db_config)
        except Exception:
            return None

    def create_run(self, state: GraphState) -> None:
        """
        Create the initial GraphRuns record and save version 1.
        """
        self._runs_memory[state.run_id] = {
            "run_id": state.run_id,
            "graph_name": state.graph_name,
            "customer_id": state.customer_id,
            "policy_id": state.policy_id,
            "claim_id": state.claim_id,
            "vessel_id": state.vessel_id,
            "current_state": state.current_state,
            "status": state.status.value,
            "checkpoint_version": 0,
        }
        self._checkpoints_memory.setdefault(state.run_id, [])

        connection = self._get_connection()
        if connection is not None:
            try:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    INSERT INTO GraphRuns (
                        run_id,
                        graph_name,
                        customer_id,
                        policy_id,
                        claim_id,
                        vessel_id,
                        current_state,
                        status,
                        checkpoint_version
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    """,
                    (
                        state.run_id,
                        state.graph_name,
                        state.customer_id,
                        state.policy_id,
                        state.claim_id,
                        state.vessel_id,
                        state.current_state,
                        state.status.value,
                        0,
                    ),
                )
                connection.commit()
                cursor.close()
                connection.close()
            except Exception:
                pass

        self.save_checkpoint(state)

    def save_checkpoint(self, state: GraphState) -> int:
        """
        Save a full snapshot of the current graph state.
        """
        # Memory bookkeeping
        run_mem = self._runs_memory.get(state.run_id)
        if run_mem is None:
            run_mem = {
                "run_id": state.run_id,
                "graph_name": state.graph_name,
                "checkpoint_version": 0,
            }
            self._runs_memory[state.run_id] = run_mem

        next_version = run_mem.get("checkpoint_version", 0) + 1
        run_mem["checkpoint_version"] = next_version
        run_mem["current_state"] = state.current_state
        run_mem["status"] = state.status.value

        state.checkpoint_version = next_version
        state_json = state.model_dump_json()

        self._checkpoints_memory.setdefault(state.run_id, []).append(
            {
                "checkpoint_version": next_version,
                "current_state": state.current_state,
                "status": state.status.value,
                "state_json": state_json,
            }
        )

        connection = self._get_connection()
        if connection is not None:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute(
                    """
                    SELECT checkpoint_version
                    FROM GraphRuns
                    WHERE run_id = %s
                    FOR UPDATE
                    """,
                    (state.run_id,),
                )
                run = cursor.fetchone()
                if run is not None:
                    db_next_version = int(run["checkpoint_version"]) + 1
                    state.checkpoint_version = db_next_version
                    state_json = state.model_dump_json()

                    cursor.execute(
                        """
                        INSERT INTO GraphCheckpoints (
                            run_id,
                            checkpoint_version,
                            current_state,
                            status,
                            state_json
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            state.run_id,
                            db_next_version,
                            state.current_state,
                            state.status.value,
                            state_json,
                        ),
                    )

                    cursor.execute(
                        """
                        UPDATE GraphRuns
                        SET
                            customer_id = %s,
                            policy_id = %s,
                            claim_id = %s,
                            vessel_id = %s,
                            current_state = %s,
                            status = %s,
                            checkpoint_version = %s
                        WHERE run_id = %s
                        """,
                        (
                            state.customer_id,
                            state.policy_id,
                            state.claim_id,
                            state.vessel_id,
                            state.current_state,
                            state.status.value,
                            db_next_version,
                            state.run_id,
                        ),
                    )
                    connection.commit()
                    next_version = db_next_version
                cursor.close()
                connection.close()
            except Exception:
                pass

        return next_version

    def load_latest_checkpoint(
        self,
        run_id: str,
    ) -> Optional[GraphState]:
        """
        Load the newest durable checkpoint for a graph run.
        """
        connection = self._get_connection()
        if connection is not None:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute(
                    """
                    SELECT state_json
                    FROM GraphCheckpoints
                    WHERE run_id = %s
                    ORDER BY checkpoint_version DESC
                    LIMIT 1
                    """,
                    (run_id,),
                )
                checkpoint = cursor.fetchone()
                cursor.close()
                connection.close()
                if checkpoint is not None:
                    raw_state = checkpoint["state_json"]
                    state_data = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
                    return GraphState.model_validate(state_data)
            except Exception:
                pass

        # Memory store fallback
        chk_list = self._checkpoints_memory.get(run_id, [])
        if not chk_list:
            return None
        latest = sorted(chk_list, key=lambda c: c["checkpoint_version"])[-1]
        raw_state = latest["state_json"]
        state_data = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
        return GraphState.model_validate(state_data)

    def load_checkpoint(
        self,
        run_id: str,
        checkpoint_version: int,
    ) -> Optional[GraphState]:
        """
        Load one specific checkpoint version.
        """
        connection = self._get_connection()
        if connection is not None:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute(
                    """
                    SELECT state_json
                    FROM GraphCheckpoints
                    WHERE run_id = %s
                      AND checkpoint_version = %s
                    """,
                    (run_id, checkpoint_version),
                )
                checkpoint = cursor.fetchone()
                cursor.close()
                connection.close()
                if checkpoint is not None:
                    raw_state = checkpoint["state_json"]
                    state_data = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
                    return GraphState.model_validate(state_data)
            except Exception:
                pass

        chk_list = self._checkpoints_memory.get(run_id, [])
        for chk in chk_list:
            if chk["checkpoint_version"] == checkpoint_version:
                raw_state = chk["state_json"]
                state_data = json.loads(raw_state) if isinstance(raw_state, str) else raw_state
                return GraphState.model_validate(state_data)
        return None

    def run_exists(self, run_id: str) -> bool:
        """
        Check whether a graph run already exists.
        """
        if run_id in self._runs_memory:
            return True

        connection = self._get_connection()
        if connection is not None:
            try:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    SELECT 1
                    FROM GraphRuns
                    WHERE run_id = %s
                    LIMIT 1
                    """,
                    (run_id,),
                )
                exists = cursor.fetchone() is not None
                cursor.close()
                connection.close()
                return exists
            except Exception:
                pass

        return False