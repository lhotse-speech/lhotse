import threading
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Generator, Iterable


def parallel_map(
    fn: Callable,
    *iterables: Iterable,
    num_jobs: int = 1,
    queue_size: int = 5000,
    threads: bool = False,
) -> Generator:
    """
    Works like Python's ``map``, but parallelizes the execution of ``fn`` over ``num_jobs``
    subprocesses or threads.

    Under the hood, it spawns ``num_jobs`` producer jobs that put their results on a queue.
    The current thread becomes a consumer thread and this generator yields items from the queue
    to the caller, as they become available.

    Example::

        >>> for root in parallel_map(math.sqrt, range(1000), num_jobs=4):
        ...     print(root)

    :param fn: function/callable to execute on each element.
    :param iterables: one of more iterables (one for each parameter of ``fn``).
    :param num_jobs: the number of parallel jobs.
    :param queue_size: max number of result items stored in memory.
        Decreasing this number might save more memory when the downstream processing is slower than
        the producer jobs.
    :param threads: whether to use threads instead of processes for producers (false by default).
    :return: a generator over results from ``fn`` applied to each item of ``iterables``.
    """
    thread = SubmitterThread(
        fn, *iterables, num_jobs=num_jobs, queue_size=queue_size, threads=threads
    )
    thread.start()
    q = thread.queue

    while thread.is_alive() or not q.empty():
        try:
            yield q.get(block=True, timeout=0.1).result()
        except queue.Empty:
            # Hit the timeout but thread is still running, try again.
            # This is needed to avoid hanging at the end when nothing else
            # shows up in the queue, but the thread didn't shutdown yet.
            continue

    thread.join()


class SubmitterThread(threading.Thread):
    def __init__(
        self,
        fn: Callable,
        *iterables,
        num_jobs: int = 1,
        queue_size: int = 10000,
        threads: bool = False,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.num_jobs = num_jobs
        self.iterables = iterables
        self.queue = queue.Queue(maxsize=queue_size)
        self.use_threads = threads

    def run(self) -> None:
        executor = ProcessPoolExecutor if self.use_threads else ThreadPoolExecutor
        with executor(self.num_jobs) as ex:
            for args in zip(*self.iterables):
                future = ex.submit(self.fn, *args)
                self.queue.put(future, block=True)
