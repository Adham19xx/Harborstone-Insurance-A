import time
from sliding_window import SlidingWindow
from observation_masking import ObservationMasking
from recursive_summary import RecursiveSummarizer
from zone_pruning import ZonePruner


class ContextEvaluator:
    def __init__(
        self,
        window_size=10,
        max_summary_messages=20,
        max_context_messages=30
    ):
        self.window = SlidingWindow(window_size)
        self.masking = ObservationMasking()
        self.summary = RecursiveSummarizer(max_summary_messages)
        self.pruner = ZonePruner(max_context_messages)

    def evaluate(self, messages):
        start = time.time()

        self.window.clear()
        self.window.extend(messages)

        context = self.window.get_context()
        context = self.masking.mask(context)
        context = self.summary.summarize(context)
        context = self.pruner.prune(context)

        latency = round(time.time() - start, 4)

        return {
            "context": context,
            "metrics": {
                "messages": len(context),
                "latency": latency
            }
        }

    def process(self, messages):
        return self.evaluate(messages)["context"]