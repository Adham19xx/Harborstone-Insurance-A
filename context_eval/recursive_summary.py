from typing import List, Any


class RecursiveSummarizer:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages

    def summarize(self, messages: List[Any]) -> List[Any]:
        if len(messages) <= self.max_messages:
            return messages

        old_messages = messages[:-self.max_messages]
        recent_messages = messages[-self.max_messages:]

        summary = self._build_summary(old_messages)

        return [
            {
                "role": "system",
                "content": summary
            }
        ] + recent_messages

    def _build_summary(self, messages: List[Any]) -> str:
        lines = []

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                role = "unknown"
                content = str(msg)

            content = content.replace("\n", " ").strip()

            if len(content) > 120:
                content = content[:120] + "..."

            lines.append(f"{role}: {content}")

        return (
            "Summary of previous conversation:\n"
            + "\n".join(lines)
        )