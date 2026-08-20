from __future__ import annotations

import json
from typing import Optional

import mysql.connector

from state_graph.models import GraphState


class CheckpointManager:
    """
    Persist and restore State Graph runs using the existing
    Harborstone Insurance MySQL database.

    The manager stores:
    1. A summary of the current run in GraphRuns.
    2. A full immutable snapshot after every meaningful transition
       in GraphCheckpoints.
    """

    def __init__(
        self,
        host: str = "localhost",
        user: str = "root",
        password: str = "",
        database: str = "harborstone_insurance",
    ) -> None:
        self.db_config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
        }

    def _get_connection(self):
        """
        Open a new connection to the existing Harborstone database.
        """

        return mysql.connector.connect(**self.db_config)

    def create_run(self, state: GraphState) -> None:
        """
        Create the initial GraphRuns record.

        This method should be called once when a new graph run starts.
        The initial state is also saved as checkpoint version 1.
        """

        connection = self._get_connection()

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

        except Exception:
            connection.rollback()
            raise

        finally:
            cursor.close()
            connection.close()

        self.save_checkpoint(state)

    def save_checkpoint(self, state: GraphState) -> int:
        """
        Save a full snapshot of the current graph state.

        A new checkpoint version is created every time this method
        succeeds. The GraphRuns row is updated to point to the
        newest version.

        Returns:
            The new checkpoint version.
        """

        connection = self._get_connection()

        try:
            cursor = connection.cursor(dictionary=True)

            # Lock the current run row while calculating
            # the next checkpoint version.
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

            if run is None:
                raise ValueError(
                    f"Graph run '{state.run_id}' does not exist. "
                    "Call create_run() before saving checkpoints."
                )

            next_version = int(run["checkpoint_version"]) + 1

            # Keep the in-memory state synchronized with the
            # checkpoint that is about to be persisted.
            state.checkpoint_version = next_version

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
                    next_version,
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
                    next_version,
                    state.run_id,
                ),
            )

            connection.commit()

            return next_version

        except Exception:
            connection.rollback()
            raise

        finally:
            cursor.close()
            connection.close()

    def load_latest_checkpoint(
        self,
        run_id: str,
    ) -> Optional[GraphState]:
        """
        Load the newest durable checkpoint for a graph run.

        This is the method used after a crash/restart to restore
        the graph from its last persisted state.
        """

        connection = self._get_connection()

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

            if checkpoint is None:
                return None

            raw_state = checkpoint["state_json"]

            # mysql-connector may return JSON as either
            # a string or a Python object depending on configuration.
            if isinstance(raw_state, str):
                state_data = json.loads(raw_state)
            else:
                state_data = raw_state

            return GraphState.model_validate(state_data)

        finally:
            cursor.close()
            connection.close()

    def load_checkpoint(
        self,
        run_id: str,
        checkpoint_version: int,
    ) -> Optional[GraphState]:
        """
        Load one specific checkpoint version.

        Useful for debugging or future rollback functionality.
        """

        connection = self._get_connection()

        try:
            cursor = connection.cursor(dictionary=True)

            cursor.execute(
                """
                SELECT state_json
                FROM GraphCheckpoints
                WHERE run_id = %s
                  AND checkpoint_version = %s
                """,
                (
                    run_id,
                    checkpoint_version,
                ),
            )

            checkpoint = cursor.fetchone()

            if checkpoint is None:
                return None

            raw_state = checkpoint["state_json"]

            if isinstance(raw_state, str):
                state_data = json.loads(raw_state)
            else:
                state_data = raw_state

            return GraphState.model_validate(state_data)

        finally:
            cursor.close()
            connection.close()

    def run_exists(self, run_id: str) -> bool:
        """
        Check whether a graph run already exists.
        """

        connection = self._get_connection()

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

            return cursor.fetchone() is not None

        finally:
            cursor.close()
            connection.close()