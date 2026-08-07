import json
from datetime import datetime


class MemoryRouter:
    def __init__(self, episodic_store=None, log_file="router_logs.json"):
        self.episodic_store = episodic_store
        self.log_file = log_file

    def route(self, memory):
        """
        memory example:
        {
            "content": "...",
            "importance": 0.8,
            "future_use": True,
            "frequency": 2,
            "user_preference": False,
            "emotion": False
        }
        """

        importance = memory.get("importance", 0)
        future_use = memory.get("future_use", False)
        frequency = memory.get("frequency", 1)
        user_preference = memory.get("user_preference", False)
        emotion = memory.get("emotion", False)

        score = 0

        score += importance * 5

        if future_use:
            score += 2

        if user_preference:
            score += 2

        if emotion:
            score += 1

        score += min(frequency, 5)

        if score >= 8:
            decision = "EPISODIC"

            if self.episodic_store is not None:
                self.episodic_store.add(memory)

        else:
            decision = "FORGET"

        self._log(memory, decision, score)

        return decision

    def _log(self, memory, decision, score):
        log = {
            "timestamp": str(datetime.now()),
            "memory": memory.get("content", ""),
            "importance": memory.get("importance", 0),
            "future_use": memory.get("future_use", False),
            "frequency": memory.get("frequency", 1),
            "user_preference": memory.get("user_preference", False),
            "emotion": memory.get("emotion", False),
            "score": score,
            "decision": decision,
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log, ensure_ascii=False))
                f.write("\n")
        except Exception:
            pass


if __name__ == "__main__":

    router = MemoryRouter()

    samples = [
        {
            "content": "My birthday is May 8",
            "importance": 0.95,
            "future_use": True,
            "frequency": 3,
            "user_preference": True,
            "emotion": False,
        },
        {
            "content": "Hello",
            "importance": 0.02,
            "future_use": False,
            "frequency": 1,
            "user_preference": False,
            "emotion": False,
        },
    ]

    for memory in samples:
        print(router.route(memory))