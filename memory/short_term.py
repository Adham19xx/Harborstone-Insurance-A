from typing import List, Dict, Any, Optional


class ShortTermMemory:
    """
    A rolling message buffer to store the ongoing conversation transcript.
    This is intentionally separated from the agent's scratchpad.

    This memory is session-scoped (i.e., it resets when the session ends).
    """

    def __init__(self) -> None:
        # List to store the conversation transcript
        self.transcript: List[Dict[str, Any]] = []

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new message to the transcript.

        Args:
            role: "user" | "assistant" | "system"
            content: message text
            metadata: optional additional info (timestamps, tags, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }

        self.transcript.append(message)

    def get_transcript(self) -> List[Dict[str, Any]]:
        """
        Retrieve full conversation transcript.
        """
        return self.transcript

    def get_recent_turns(self, n: int) -> List[Dict[str, Any]]:
        """
        Get the last N messages from the transcript.

        Args:
            n: number of recent messages to return
        """
        if n <= 0:
            return []

        return self.transcript[-n:] if n < len(self.transcript) else self.transcript

    def prune_transcript(self, keep_last_n: int) -> List[Dict[str, Any]]:
        """
        Trim the transcript to keep only the last N messages.

        Returns:
            List of removed (overflowed) messages, which can later
            be sent to long-term memory (episodic store).
        """
        if keep_last_n <= 0:
            overflowed = self.transcript
            self.transcript = []
            return overflowed

        if keep_last_n >= len(self.transcript):
            return []

        overflowed_messages = self.transcript[:-keep_last_n]
        self.transcript = self.transcript[-keep_last_n:]

        return overflowed_messages

    def clear(self) -> None:
        """
        Clear the entire short-term memory (end of session).
        """
        self.transcript.clear()

    def __len__(self) -> int:
        """
        Return number of stored messages.
        """
        return len(self.transcript)