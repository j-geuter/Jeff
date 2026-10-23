"""Batching for the server (on by default; --batching off serves one question at a time).

The questions of all requests that are waiting are answered together. A worker thread takes the waiting questions (up
to max_batch questions and max_tokens padded input tokens, a cap measured from the GPU's free memory) and runs their
decision passes as one forward pass (Decider.hidden_batch). It never waits for more requests to arrive: a question that
finds the server idle is answered at once, alone. Questions that reason:
- with the vLLM engine, a second thread hands all waiting reasonings to vLLM as one batch while the worker goes on
  answering other questions; the read passes after the reasonings join the worker's next batch;
- with the transformers engine, the worker writes them one after another, after answering the rest of its batch.
A request is answered when all its questions are. With one request at a time (as JevBench sends them) every batch
holds one question and the answers are exactly those of --batching off; with several questions in a batch, the padding
changes the probabilities by numerical noise only (about the size of the effect of --compile).
"""
from __future__ import annotations

import collections
import queue
import threading
import time

from .prompting import MAX_OPTIONS


class Request:
    """One request's questions in flight; done is set when all are answered (or one failed: error)."""

    def __init__(self, keys: list[str], deadline: float | None):
        self.keys, self.left, self.deadline = list(keys), len(keys), deadline
        self.answers, self.tokens, self.generated, self.thought, self.seconds = {}, 0, 0, 0, 0.0
        self.error = None
        self.done = threading.Event()
        self.lock = threading.Lock()


class Job:
    """One question: stage "first" (the one-pass answer), "read" (after its reasoning) or "wide" (> 16 options)."""

    def __init__(self, request: Request, key: str, question: dict, row: dict):
        self.request, self.key, self.question, self.row = request, key, question, row
        self.ids, self.k, self.stage, self.input_tokens = None, 0, "first", 0
        self.first, self.prompt, self.gen = None, None, None
        self.admitted = self.counted = False       # reasoning admitted by the limits / its output counted


class Batcher:
    def __init__(self, model, max_batch: int = 32, max_tokens: int | None = None):
        self.model, self.decider = model, model.decider
        self.max_batch = int(max_batch)
        if self.max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        self.max_tokens = int(max_tokens) if max_tokens else self.decider.batch_tokens()
        self.jobs, self.reasonings = queue.Queue(), queue.Queue()
        self.backlog = collections.deque()        # taken from jobs but not yet run (the worker's own)
        self.limits = threading.Lock()            # the reasoner's counts (both threads update them)
        reasoner = self.decider.reasoner
        self.parallel = reasoner is not None and reasoner.generator is not None
        threading.Thread(target=self._work, daemon=True, name="jeff-decide").start()
        if self.parallel:
            threading.Thread(target=self._reason, daemon=True, name="jeff-reason").start()

    def submit(self, state, questions: dict) -> Request:
        """Answer one request; blocks until it is answered. Invalid questions raise at once (before anything runs);
        a failure while answering is raised here too."""
        reasoner = self.decider.reasoner
        rows = [(key, question, self.model.row(state, question, key)) for key, question in questions.items()]
        request = Request([key for key, _, _ in rows], reasoner.request_deadline() if reasoner is not None else None)
        for key, question, row in rows:
            self.jobs.put(Job(request, key, question, row))
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request

    # -- the worker: decision passes ---------------------------------------------------------------------------------

    def _ready(self, job: Job) -> bool:
        """Tokenize a new question (False: it failed or was answered on the spot)."""
        if job.stage != "first" or job.ids is not None:
            return True
        try:
            if len(job.row.get("options") or []) > MAX_OPTIONS:
                self._wide(job)
                return False
            job.row, job.ids, _ = self.decider.prepare(job.row)
            job.k, job.input_tokens = len(job.row["options"]), len(job.ids)
            return True
        except Exception as error:                # noqa: BLE001 - reported to the request
            self._fail(job, error)
            return False

    def _next(self, block: bool) -> Job | None:
        try:
            return self.jobs.get(block=block)
        except queue.Empty:
            return None

    def _take(self) -> list[Job]:
        """The next batch. All waiting questions are taken and sorted by length; a batch holds questions of similar
        length (padding wastes compute: the batch costs batch size x longest input), up to max_batch questions and
        max_tokens padded tokens. The rest wait in the backlog, which is answered before new questions are taken."""
        if not self.backlog:
            waiting = [self._next(block=True)]
            while True:
                job = self._next(block=False)
                if job is None:
                    break
                waiting.append(job)
            ready = [job for job in waiting if self._ready(job)]
            self.backlog.extend(sorted(ready, key=lambda job: len(job.ids)))
            if not self.backlog:
                return []
        batch, width, real = [], 0, 0
        while self.backlog and len(batch) < self.max_batch:
            job = self.backlog[0]
            wider, more = max(width, len(job.ids)), real + len(job.ids)
            if batch and (wider * (len(batch) + 1) > self.max_tokens or wider * (len(batch) + 1) > max(1.25 * more, more + 256)):
                break
            batch.append(self.backlog.popleft())
            width, real = wider, more
        return batch

    def _work(self) -> None:
        torch = self.decider._torch
        while True:
            batch = self._take()
            if not batch:
                continue
            try:
                with torch.inference_mode():
                    start = time.perf_counter()
                    self.decider.cudnn_for(max(len(job.ids) for job in batch))
                    hs = self.decider.hidden_batch([job.ids for job in batch])
                    if self.decider.device.type == "cuda":
                        torch.cuda.synchronize(self.decider.device)
                    share = (time.perf_counter() - start) / len(batch)
                    self._after(batch, hs, share)
            except Exception as error:            # noqa: BLE001 - e.g. out of GPU memory: fail this batch only
                for job in batch:
                    self._release(job)
                    self._fail(job, error)

    def _after(self, batch: list[Job], hs, share: float) -> None:
        reasoner, inline = self.decider.reasoner, []
        for i, job in enumerate(batch):
            h = hs[i: i + 1]
            if job.stage == "read":
                probs, _ = reasoner.read(h, job.k, job.row["qtype"], job.first)
                self._count(job, len(job.gen))
                self._answer(job, probs, len(job.gen), True, share)
                continue
            probs, logits = self.decider.first(h, job.k)
            with self.limits:
                wants = reasoner is not None and reasoner.wants(job.row["qtype"], probs, job.request.deadline)
                job.admitted = wants
            if reasoner is not None and not wants:
                self._count(job, 0)
            if not wants:
                self._answer(job, probs, 0, False, share)
            elif self.parallel:                   # vLLM writes it; the read pass comes back to this worker
                job.first, job.prompt, job.stage = logits, reasoner.prompt(job.ids), "write"
                self.reasonings.put(job)
            else:
                inline.append((job, logits))
        for job, logits in inline:                # transformers: after the rest of the batch is answered
            start = time.perf_counter()
            try:
                probs, _, generated = reasoner.think(job.ids, job.k, job.row["qtype"], logits)
            except Exception as error:            # noqa: BLE001
                self._release(job)
                self._fail(job, error)
                continue
            self._count(job, generated)
            self._answer(job, probs, generated, True, share + time.perf_counter() - start)

    def _wide(self, job: Job) -> None:
        """More than 16 options: answered alone, in groups (Decider.decide_wide; these questions do not reason)."""
        with self.decider._torch.inference_mode():
            probs, _, tokens, seconds = self.decider.decide_wide(job.row)
        job.input_tokens = tokens
        self._answer(job, probs, 0, False, seconds)

    # -- the reasoning thread (vLLM) ---------------------------------------------------------------------------------

    def _reason(self) -> None:
        reasoner = self.decider.reasoner
        while True:
            batch = [self.reasonings.get()]
            while True:
                try:
                    batch.append(self.reasonings.get_nowait())
                except queue.Empty:
                    break
            try:
                gens = reasoner.write([job.prompt for job in batch])
            except Exception as error:            # noqa: BLE001
                for job in batch:
                    self._release(job)
                    self._fail(job, error)
                continue
            for job, gen in zip(batch, gens):
                job.gen, job.ids, job.stage = gen, reasoner.read_ids(job.prompt, gen), "read"
                self.jobs.put(job)

    # -- results -----------------------------------------------------------------------------------------------------

    def _count(self, job: Job, generated: int) -> None:
        """The question is answered: count it (and its output) for the reasoning limits, once."""
        with self.limits:
            if not job.counted:
                self.decider.reasoner.count(generated, job.admitted)
                job.counted = True

    def _release(self, job: Job) -> None:
        """A failed question: release the budget a reasoning held."""
        if self.decider.reasoner is not None and job.admitted:
            self._count(job, 0)

    def _answer(self, job: Job, probs: list[float], generated: int, thought: bool, seconds: float) -> None:
        answer = self.model.answer(job.question["type"], [o["id"] for o in job.row["options"]], probs)
        request = job.request
        with request.lock:
            request.answers[job.key] = answer
            request.tokens += job.input_tokens
            request.generated += generated
            request.thought += int(thought)
            request.seconds += seconds
            request.left -= 1
            if request.left == 0:
                request.done.set()

    def _fail(self, job: Job, error: Exception) -> None:
        request = job.request
        with request.lock:
            if request.error is None:
                request.error = error
            request.done.set()
