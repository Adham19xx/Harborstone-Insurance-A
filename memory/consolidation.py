import logging
import json
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConsolidationLayer:
    """
    Consolidation Layer: The background brain of the memory system.
    Reads unconsolidated episodes, extracts semantic facts via an LLM,
    resolves conflicts, and updates the Semantic Memory.
    """

    def __init__(self, episodic_store, semantic_store, llm_client=None):
        """
        Initializes the Consolidation Layer with access to both memory stores.

        Args:
            episodic_store: Instance of EpisodicStore
            semantic_store: Instance of SemanticStore
            llm_client: LLM client (must have method generate(prompt: str) -> str)
        """
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.llm_client = llm_client

    def run_consolidation_pass(self) -> None:
        """
        Main method to process unconsolidated episodes.
        """
        unconsolidated_episodes = self.episodic_store.get_unconsolidated_episodes()

        if not unconsolidated_episodes:
            logger.info("No new unconsolidated episodes found.")
            return

        logger.info(f"Found {len(unconsolidated_episodes)} episodes to process.")

        for episode in unconsolidated_episodes:
            episode_id = episode.get("episode_id")
            messages = episode.get("messages", [])

            try:
                extracted_facts = self._extract_facts_from_messages(messages)

                for fact in extracted_facts:
                    self._process_and_store_fact(
                        entity=fact.get("entity"),
                        attribute=fact.get("attribute"),
                        value=fact.get("value"),
                        source_episode_id=episode_id,
                        expires_at=fact.get("expires_at")
                    )

                self.episodic_store.mark_as_consolidated(episode_id)
                logger.info(f"Episode consolidated: {episode_id}")

            except Exception as e:
                logger.error(f"Error processing episode {episode_id}: {e}")

    def _extract_facts_from_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Uses LLM to extract structured facts from transcript messages.
        """

        if not messages or not self.llm_client:
            return []

        try:
            prompt = f"""
Extract structured facts from the following conversation.
Return ONLY valid JSON array matching this exact format:
[
  {{
    "entity": "string",
    "attribute": "string",
    "value": "any",
    "expires_at": null
  }}
]
Conversation:
{json.dumps(messages, ensure_ascii=False)}
"""
            response = self.llm_client.generate(prompt)

            # Clean markdown code formatting if returned by the LLM
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                cleaned_response = re.sub(r"^```(?:json)?\s*", "", cleaned_response)
                cleaned_response = re.sub(r"\s*```$", "", cleaned_response)

            parsed = json.loads(cleaned_response)

            if isinstance(parsed, list):
                return parsed
            else:
                logger.warning("LLM returned invalid format (not a list).")
                return []

        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON.")
            return []

        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return []

    def _process_and_store_fact(
        self,
        entity: str,
        attribute: str,
        value: Any,
        source_episode_id: str,
        expires_at: Optional[str] = None
    ) -> None:
        """
        Handles conflict detection and storing facts in SemanticStore.

        FIXES applied vs. the original version:
          1. `flag_conflict` used to be called with the SAME fact_id for both
             fact_id_1 and fact_id_2 (the new fact didn't exist yet at that
             point), which meant no real conflict link was ever recorded.
             We now create the new fact FIRST, look up its real fact_id, and
             only then flag the conflict between the OLD fact and the NEW
             fact.
          2. If the freshly extracted value is identical to the existing
             active value, we skip storing a redundant duplicate version
             (the original code always called add_or_update_fact, even when
             nothing had actually changed).
        """

        if not entity or not attribute:
            logger.warning("Invalid fact (missing entity or attribute). Skipping.")
            return

        active_facts = self.semantic_store.get_active_facts(entity=entity)

        existing_fact = next(
            (f for f in active_facts if f.get("attribute") == attribute),
            None
        )

        # FIX 2: nothing actually changed -> don't create a redundant version
        if existing_fact and existing_fact.get("value") == value:
            logger.info(
                f"No change for {entity}.{attribute} (value already '{value}'). "
                "Skipping re-store."
            )
            return

        old_fact_id = existing_fact.get("fact_id") if existing_fact else None
        old_value = existing_fact.get("value") if existing_fact else None

        # Store the new version first (SemanticStore handles superseding +
        # versioning of the previous active fact internally).
        self.semantic_store.add_or_update_fact(
            entity=entity,
            attribute=attribute,
            value=value,
            source_episode_id=source_episode_id,
            expires_at=expires_at
        )

        if existing_fact:
            logger.info(
                f"Conflict detected for {entity}.{attribute}: "
                f"'{old_value}' -> '{value}'"
            )

            # FIX 1: look up the fact we just created so we can pass its
            # real fact_id, instead of reusing the old fact's id twice.
            updated_facts = self.semantic_store.get_facts_by_entity(entity)
            new_fact = max(
                (f for f in updated_facts if f.get("attribute") == attribute),
                key=lambda f: f.get("version", 0),
                default=None
            )

            if new_fact and old_fact_id:
                resolution_note = (
                    f"Conflict detected: attribute '{attribute}' value changed from "
                    f"'{old_value}' to '{value}' via episode {source_episode_id}"
                )
                self.semantic_store.flag_conflict(
                    fact_id_1=old_fact_id,
                    fact_id_2=new_fact["fact_id"],
                    resolution_note=resolution_note
                )
            else:
                logger.warning(
                    f"Could not resolve both fact IDs for {entity}.{attribute}; "
                    "skipping conflict flag."
                )