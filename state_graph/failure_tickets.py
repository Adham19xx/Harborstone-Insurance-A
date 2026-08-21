from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    import mysql.connector
except ImportError:
    mysql = None

from pydantic import BaseModel, ConfigDict, Field


class TicketStatus(str, Enum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"


class FailureType(str, Enum):
    MCP_FAILURE = "MCP_FAILURE"
    LLM_FAILURE = "LLM_FAILURE"
    RAG_FAILURE = "RAG_FAILURE"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


class FailureTicket(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticket_id: str = Field(default_factory=lambda: f"ticket-{uuid4()}")
    run_id: str
    graph_name: str
    failed_node: str
    failure_type: FailureType = FailureType.UNEXPECTED_ERROR
    error_message: str
    checkpoint_version: int
    status: TicketStatus = TicketStatus.OPEN
    recovery_attempts: int = 0
    resolution_note: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def mark_investigating(self) -> None:
        self.status = TicketStatus.INVESTIGATING
        self.recovery_attempts += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_resolved(self, resolution_note: str = "Resolved successfully") -> None:
        self.status = TicketStatus.RESOLVED
        self.resolution_note = resolution_note
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def record_recovery_failure(self, error_message: str) -> None:
        self.status = TicketStatus.INVESTIGATING
        self.error_message = error_message
        self.updated_at = datetime.now(timezone.utc).isoformat()


class TicketManager:
    """
    Manages persistent failure tickets across State Graph runs.
    Supports MySQL persistence with transparent in-memory fallback for testing.
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
        self._memory_store: Dict[str, FailureTicket] = {}

    def _get_connection(self):
        if self.in_memory:
            return None
        return mysql.connector.connect(**self.db_config)

    def create_ticket(
        self,
        run_id: str,
        graph_name: str,
        failed_node: str,
        failure_type: FailureType | str,
        error_message: str,
        checkpoint_version: int,
        ticket_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FailureTicket:
        if isinstance(failure_type, str):
            try:
                failure_type = FailureType(failure_type)
            except ValueError:
                failure_type = FailureType.UNEXPECTED_ERROR

        existing = self.get_ticket(ticket_id) if ticket_id else None
        current_attempts = existing.recovery_attempts if existing else 0
        current_status = existing.status if existing and existing.status != TicketStatus.RESOLVED else TicketStatus.OPEN


        ticket = FailureTicket(
            ticket_id=ticket_id or f"ticket-{uuid4()}",
            run_id=run_id,
            graph_name=graph_name,
            failed_node=failed_node,
            failure_type=failure_type,
            error_message=error_message,
            checkpoint_version=checkpoint_version,
            status=current_status,
            recovery_attempts=current_attempts,
            metadata=metadata or {},
        )


        self._save_ticket(ticket)
        return ticket

    def _save_ticket(self, ticket: FailureTicket) -> None:
        self._memory_store[ticket.ticket_id] = ticket.model_copy()

        if self.in_memory:
            return

        try:
            conn = self._get_connection()
            if conn is None:
                return
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO FailureTickets (
                    ticket_id, run_id, graph_name, failed_node,
                    failure_type, error_message, checkpoint_version,
                    status, recovery_attempts, resolution_note, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    status = VALUES(status),
                    recovery_attempts = VALUES(recovery_attempts),
                    resolution_note = VALUES(resolution_note),
                    error_message = VALUES(error_message),
                    metadata = VALUES(metadata),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    ticket.ticket_id,
                    ticket.run_id,
                    ticket.graph_name,
                    ticket.failed_node,
                    ticket.failure_type.value
                    if isinstance(ticket.failure_type, FailureType)
                    else str(ticket.failure_type),
                    ticket.error_message,
                    ticket.checkpoint_version,
                    ticket.status.value
                    if isinstance(ticket.status, TicketStatus)
                    else str(ticket.status),
                    ticket.recovery_attempts,
                    ticket.resolution_note,
                    json.dumps(ticket.metadata, default=str),
                ),
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception:
            pass

    def get_ticket(self, ticket_id: str) -> Optional[FailureTicket]:
        if self.in_memory or ticket_id in self._memory_store:
            ticket = self._memory_store.get(ticket_id)
            if ticket is not None:
                return ticket.model_copy()

        try:
            conn = self._get_connection()
            if conn is None:
                return self._memory_store.get(ticket_id)
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM FailureTickets WHERE ticket_id = %s",
                (ticket_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if not row:
                return self._memory_store.get(ticket_id)

            meta = row.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            elif not isinstance(meta, dict):
                meta = {}

            ticket = FailureTicket(
                ticket_id=row["ticket_id"],
                run_id=row["run_id"],
                graph_name=row["graph_name"],
                failed_node=row["failed_node"],
                failure_type=FailureType(row["failure_type"])
                if row["failure_type"] in FailureType._value2member_map_
                else FailureType.UNEXPECTED_ERROR,
                error_message=row["error_message"],
                checkpoint_version=row["checkpoint_version"],
                status=TicketStatus(row["status"])
                if row["status"] in TicketStatus._value2member_map_
                else TicketStatus.OPEN,
                recovery_attempts=row["recovery_attempts"],
                resolution_note=row.get("resolution_note"),
                metadata=meta,
                created_at=str(row.get("created_at")),
                updated_at=str(row.get("updated_at")),
            )
            self._memory_store[ticket.ticket_id] = ticket.model_copy()
            return ticket
        except Exception:
            return self._memory_store.get(ticket_id)

    def get_latest_ticket_for_run(self, run_id: str) -> Optional[FailureTicket]:
        candidates = [t for t in self._memory_store.values() if t.run_id == run_id]
        if candidates and self.in_memory:
            return sorted(candidates, key=lambda t: t.checkpoint_version, reverse=True)[0].model_copy()

        try:
            conn = self._get_connection()
            if conn is None:
                return candidates[-1].model_copy() if candidates else None
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM FailureTickets WHERE run_id = %s ORDER BY checkpoint_version DESC, created_at DESC LIMIT 1",
                (run_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            if row:
                return self.get_ticket(row["ticket_id"])
        except Exception:
            pass

        return candidates[-1].model_copy() if candidates else None

    def start_investigation(self, ticket_id: str) -> Optional[FailureTicket]:
        ticket = self.get_ticket(ticket_id)
        if ticket is None:
            return None
        ticket.mark_investigating()
        self._save_ticket(ticket)
        return ticket

    def resolve_ticket(
        self, ticket_id: str, resolution_note: str = "Resolved successfully"
    ) -> Optional[FailureTicket]:
        ticket = self.get_ticket(ticket_id)
        if ticket is None:
            return None
        ticket.mark_resolved(resolution_note=resolution_note)
        self._save_ticket(ticket)
        return ticket

    def record_recovery_failure(
        self, ticket_id: str, error_message: str
    ) -> Optional[FailureTicket]:
        ticket = self.get_ticket(ticket_id)
        if ticket is None:
            return None
        ticket.record_recovery_failure(error_message)
        self._save_ticket(ticket)
        return ticket
