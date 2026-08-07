import time


class ContextMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.original_messages = 0
        self.final_messages = 0
        self.original_tokens = 0
        self.final_tokens = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def estimate_tokens(self, messages):
        total = 0

        for msg in messages:
            if isinstance(msg, dict):
                text = msg.get("content", "")
            else:
                text = str(msg)

            total += max(1, len(text.split()))

        return total

    def calculate(self, original, processed):
        self.original_messages = len(original)
        self.final_messages = len(processed)

        self.original_tokens = self.estimate_tokens(original)
        self.final_tokens = self.estimate_tokens(processed)

        latency = 0
        if self.start_time is not None and self.end_time is not None:
            latency = round(self.end_time - self.start_time, 4)

        reduction = 0
        if self.original_tokens > 0:
            reduction = round(
                (
                    (self.original_tokens - self.final_tokens)
                    / self.original_tokens
                )
                * 100,
                2,
            )

        return {
            "original_messages": self.original_messages,
            "processed_messages": self.final_messages,
            "original_tokens": self.original_tokens,
            "processed_tokens": self.final_tokens,
            "token_reduction_percent": reduction,
            "latency_seconds": latency,
        }