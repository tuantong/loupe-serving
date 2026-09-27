"""Cross-request batching: concurrent decide() calls share one forward pass."""

import threading
import time

from loupe.prompt import Messages


class _Job:
    def __init__(self, prompt: list[int], slice_ids: list[int]):
        self.prompt = prompt
        self.slice_ids = slice_ids
        self.done = threading.Event()
        self.result: list[float] | None = None
        self.error: BaseException | None = None


class BatchedBackend:
    """Wraps a backend exposing slice_logits_from_ids; collects prompts from
    concurrent callers for up to window_ms and runs them in one batch."""

    def __init__(self, backend, window_ms: float = 8.0, max_batch: int = 16):
        self.backend = backend
        self.window = window_ms / 1000.0
        self.max_batch = max_batch
        self._pending: list[_Job] = []
        self._cv = threading.Condition()
        self._gpu = threading.Lock()
        threading.Thread(target=self._loop, daemon=True, name="loupe-batcher").start()

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def slice_logits(self, batch: list[Messages], slice_ids: list[list[int]]) -> list[list[float]]:
        jobs = [_Job(self.backend._prompt_ids(m, False), ids) for m, ids in zip(batch, slice_ids)]
        with self._cv:
            self._pending.extend(jobs)
            self._cv.notify()
        for job in jobs:
            job.done.wait()
            if job.error is not None:
                raise job.error
        return [job.result for job in jobs]

    def think_then_slice(self, *args, **kwargs):
        with self._gpu:
            return self.backend.think_then_slice(*args, **kwargs)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._pending:
                    self._cv.wait()
                deadline = time.monotonic() + self.window
                while len(self._pending) < self.max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)
                jobs, self._pending = self._pending[: self.max_batch], self._pending[self.max_batch :]
            try:
                with self._gpu:
                    rows = self.backend.slice_logits_from_ids([j.prompt for j in jobs], [j.slice_ids for j in jobs])
                for job, row in zip(jobs, rows):
                    job.result = row
            except BaseException as exc:
                for job in jobs:
                    job.error = exc
            for job in jobs:
                job.done.set()
