import re
from typing import List, Any


class ObservationMasking:
    def __init__(self):
        self.patterns = [
            r"^ok$",
            r"^okay$",
            r"^thanks$",
            r"^thank you$",
            r"^hi$",
            r"^hello$",
            r"^typing\.\.\.$",
            r"^\.+$",
            r"^[👍👌😂❤️]+$",
        ]

    def _is_noise(self, text: str) -> bool:
        text = text.strip().lower()

        if len(text) == 0:
            return True

        for pattern in self.patterns:
            if re.match(pattern, text):
                return True

        return False

    def mask(self, messages: List[Any]) -> List[Any]:
        filtered = []

        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = str(msg)

            if not self._is_noise(content):
                filtered.append(msg)

        return filtered