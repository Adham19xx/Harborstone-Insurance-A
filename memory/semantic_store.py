import json
import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SemanticStore:
    """
    Semantic Memory Store: Stores consolidated facts, rules, and attributes.
    Only written to by the Consolidation Layer.
    Supports versioning, updates, and expiration.
    """

    def __init__(self, file_path: str = "db/semantic_memory.json"):
        self.file_path = file_path
        self.facts: List[Dict[str, Any]] = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load active and historical facts from disk to persist across sessions."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    else:
                        logger.warning("Invalid data format in semantic memory file. Resetting to empty list.")
                        return []
            except json.JSONDecodeError:
                logger.error("JSON decode error in semantic memory. Returning empty list.")
                return []
            except Exception as e:
                logger.error(f"Unexpected error while loading semantic memory: {e}")
                return []
        return []

    def _save_data(self) -> None:
        """Save semantic memory data to the JSON file."""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.facts, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save semantic memory: {e}")

    def add_or_update_fact(
        self,
        entity: str,
        attribute: str,
        value: Any,
        source_episode_id: str,
        expires_at: Optional[str] = None
    ) -> None:
        """
        Add or update a fact with automatic versioning.
        Old facts are marked as superseded rather than being deleted.
        """

        # Input Validation
        if not entity or not isinstance(entity, str):
            raise ValueError("entity must be a non-empty string")

        if not attribute or not isinstance(attribute, str):
            raise ValueError("attribute must be a non-empty string")

        if not source_episode_id or not isinstance(source_episode_id, str):
            raise ValueError("source_episode_id must be a non-empty string")

        # Retrieve active facts for the same entity and attribute
        existing_facts = [
            f for f in self.facts
            if f.get("entity") == entity
            and f.get("attribute") == attribute
            and f.get("status") == "active"
        ]

        new_version = 1

        if existing_facts:
            # Select latest version
            latest_fact = max(existing_facts, key=lambda x: x.get("version", 0))

            # Supersede previous version
            latest_fact["status"] = "superseded"
            new_version = latest_fact.get("version", 1) + 1

        # Create new fact entry
        new_fact = {
            "fact_id": f"fact_{uuid.uuid4().hex}",
            "entity": entity,
            "attribute": attribute,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "source_episode": source_episode_id,
            "version": new_version,
            "expires_at": expires_at,
            "status": "active"
        }

        self.facts.append(new_fact)
        self._save_data()

        logger.info(f"Fact added/updated: {new_fact['fact_id']} (v{new_version})")

    def get_active_facts(self, entity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve active facts while enforcing expiration checks.
        """

        valid_facts = []
        now = datetime.now()

        updated = False

        for fact in self.facts:
            if fact.get("status") != "active":
                continue

            # Handle expiration logic
            expires_at = fact.get("expires_at")
            if expires_at:
                try:
                    expiry_time = datetime.fromisoformat(expires_at)
                    if expiry_time < now:
                        fact["status"] = "expired"
                        updated = True
                        continue
                except Exception:
                    logger.warning(f"Invalid expires_at format in fact: {fact.get('fact_id')}")

            if entity and fact.get("entity") != entity:
                continue

            valid_facts.append(fact)

        if updated:
            self._save_data()

        return valid_facts

    def flag_conflict(
        self,
        fact_id_1: str,
        fact_id_2: str,
        resolution_note: str
    ) -> None:
        """
        Explicitly log a conflict between two facts.
        """

        found_ids = {fact_id_1, fact_id_2}
        updated = False

        for fact in self.facts:
            if fact.get("fact_id") in found_ids:
                fact["status"] = "conflicted"
                fact["conflict_note"] = resolution_note
                updated = True

        if not updated:
            logger.warning("No matching facts found to mark conflict.")
        else:
            logger.info(f"Conflict flagged between {fact_id_1} and {fact_id_2}")

        self._save_data()

    def get_all_facts(self) -> List[Dict[str, Any]]:
        """Return all recorded facts regardless of status."""
        return self.facts

    def get_facts_by_entity(self, entity: str) -> List[Dict[str, Any]]:
        """Retrieve all facts associated with a specific entity."""
        return [
            f for f in self.facts
            if f.get("entity") == entity
        ]