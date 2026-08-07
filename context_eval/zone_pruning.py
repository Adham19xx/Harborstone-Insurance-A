from typing import List, Dict, Any


class ZonePruner:
    def __init__(
        self,
        max_messages: int = 30,
        protected_roles=None
    ):
        self.max_messages = max_messages
        self.protected_roles = protected_roles or ["system"]

    def prune(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(messages) <= self.max_messages:
            return messages

        protected = []
        normal = []

        for msg in messages:
            role = msg.get("role", "")

            if role in self.protected_roles:
                protected.append(msg)
            else:
                normal.append(msg)

        remaining = max(
            self.max_messages - len(protected),
            0
        )

        normal = normal[-remaining:]

        return protected + normal

    def split_zones(self, messages: List[Dict[str, Any]]):
        zones = {
            "system": [],
            "memory": [],
            "chat": [],
            "tool": []
        }

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                zones["system"].append(msg)

            elif role == "memory":
                zones["memory"].append(msg)

            elif role == "tool":
                zones["tool"].append(msg)

            else:
                zones["chat"].append(msg)

        return zones