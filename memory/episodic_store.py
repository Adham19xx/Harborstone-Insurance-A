import json
import os
import uuid
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class EpisodicStore:
    """
    Episodic Memory Store: Records temporal events and raw interactions.
    This acts as the source of truth for the Consolidation layer.
    """
    def __init__(self, file_path: str = "db/episodic_memory.json"):
        self.file_path = file_path
        self.episodes: List[Dict[str, Any]] = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load previous episodes from disk to persist across sessions."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    else:
                        logger.warning("Invalid data format in episodic memory file. Resetting to empty list.")
                        return []
            except json.JSONDecodeError:
                logger.error("JSON decode error. Returning empty episodic memory.")
                return []
            except Exception as e:
                logger.error(f"Unexpected error while loading episodic memory: {e}")
                return []
        return []

    def _save_data(self) -> None:
        """Save episodic memory data to the JSON file."""
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.episodes, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save episodic memory: {e}")

    def add_episode(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a new episode (group of messages) to memory.
        This method is invoked by the promote-or-drop Router.
        """

        # Input Validation
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id must be a non-empty string")

        if not messages or not isinstance(messages, list):
            raise ValueError("messages must be a non-empty list")

        # Create a new episode object
        episode = {
            "episode_id": f"ep_{uuid.uuid4().hex}",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "metadata": metadata if isinstance(metadata, dict) else {},
            "consolidated": False
        }

        self.episodes.append(episode)
        self._save_data()

        logger.info(f"New episode added: {episode['episode_id']}")

    def get_unconsolidated_episodes(self) -> List[Dict[str, Any]]:
        """Retrieve episodes that have not yet been consolidated into semantic facts."""
        return [
            ep for ep in self.episodes
            if not ep.get("consolidated", False)
        ]

    def mark_as_consolidated(self, episode_id: str) -> None:
        """Mark an episode as consolidated once processed by the consolidation layer."""

        found = False

        for ep in self.episodes:
            if ep.get("episode_id") == episode_id:
                ep["consolidated"] = True
                found = True
                logger.info(f"Episode marked as consolidated: {episode_id}")
                break

        if not found:
            logger.warning(f"Episode not found: {episode_id}")

        self._save_data()

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """Return all recorded episodes."""
        return self.episodes

    def get_episodes_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all episodes associated with a specific session_id."""
        return [
            ep for ep in self.episodes
            if ep.get("session_id") == session_id
        ]