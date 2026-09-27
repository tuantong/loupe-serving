import threading

from loupe.batcher import BatchedBackend


class Fake:
    label_prefix = ""

    def __init__(self):
        self.calls = []

    def _prompt_ids(self, messages, thinking):
        return [len(messages[0]["content"])]

    def slice_logits_from_ids(self, prompts, slice_ids):
        self.calls.append(len(prompts))
        return [[float(p[0]) + i for i in ids] for p, ids in zip(prompts, slice_ids)]


def test_concurrent_calls_share_one_forward_pass():
    fake = Fake()
    batched = BatchedBackend(fake, window_ms=50, max_batch=8)
    out = {}

    def call(n):
        out[n] = batched.slice_logits([[{"role": "user", "content": "x" * n}]], [[0, 1]])

    threads = [threading.Thread(target=call, args=(n,)) for n in range(1, 5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == {n: [[float(n), float(n) + 1]] for n in range(1, 5)}
    assert max(fake.calls) > 1


def test_errors_propagate_to_callers():
    class Broken(Fake):
        def slice_logits_from_ids(self, prompts, slice_ids):
            raise RuntimeError("boom")

    batched = BatchedBackend(Broken(), window_ms=1)
    try:
        batched.slice_logits([[{"role": "user", "content": "a"}]], [[0]])
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")
