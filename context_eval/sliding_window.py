from collections import deque
from typing import List, Any


class SlidingWindow:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add(self, message: Any):
        self.buffer.append(message)

    def extend(self, messages: List[Any]):
        self.buffer.extend(messages)

    def get_context(self) -> List[Any]:
        return list(self.buffer)

    def prune(self) -> List[Any]:
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)