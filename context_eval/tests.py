from evaluator import ContextEvaluator
from metrics import ContextMetrics


def build_messages(count):
    messages = []

    for i in range(count):
        messages.append(
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"This is message number {i}"
            }
        )

    return messages


def run_test(message_count):
    evaluator = ContextEvaluator()
    metrics = ContextMetrics()

    original = build_messages(message_count)

    metrics.start()
    result = evaluator.evaluate(original)
    metrics.stop()

    report = metrics.calculate(
        original,
        result["context"]
    )

    print("=" * 60)
    print(f"TEST: {message_count} Messages")
    print("=" * 60)

    for key, value in report.items():
        print(f"{key}: {value}")

    print()


if __name__ == "__main__":
    run_test(100)
    run_test(300)
    run_test(500)
    run_test(1000)