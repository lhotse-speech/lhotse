import multiprocessing
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Callable, Dict, Generator, Iterable, Optional, Type

from tqdm.auto import tqdm


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
        executor = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        with executor(self.num_jobs) as ex:
            for args in zip(*self.iterables):
                future = ex.submit(self.fn, *args)
                self.queue.put(future, block=True)


class ParallelExecutor:
    """
    A class which uses ProcessPoolExecutor to parallelize the execution of a callable class.
    The instances of the runner class are instantiated separately in each worker process.

    Example::

        >>> class MyRunner:
        ...     def __init__(self):
        ...         self.name = name
        ...     def __call__(self, x):
        ...         return f'processed: {x}'
        ...
        >>> runner = ParallelExecutor(MyRunner, num_jobs=4)
        >>> for item in runner(range(10)):
        ...     print(item)

    If the __init__ method of the callable class accepts parameters except for `self`,
    use `functools.partial` or similar method to obtain a proper initialization function:

        >>> class MyRunner:
        ...     def __init__(self, name):
        ...         self.name = name
        ...     def __call__(self, x):
        ...         return f'{self.name}: {x}'
        ...
        >>> runner = ParallelExecutor(partial(MyRunner, name='my_name'), num_jobs=4)
        >>> for item in runner(range(10)):
        ...     print(item)


    The initialization function will be called separately for each worker process. Steps like loading a
    PyTorch model instance to the selected device should be done inside the initialization function.
    """

    _runners: Dict[Optional[int], Callable] = {}

    def __init__(
        self,
        init_fn: Callable[[], Callable],
        num_jobs: int = 1,
        verbose: bool = False,
        description: str = "Processing",
    ):
        """
        Instantiate a parallel executor.

        :param init_fn: A function which returns a runner object (e.g. a class) that will be instantiated
            in each worker process.
        :param num_jobs: The number of parallel jobs to run. Defaults to 1 (no parallelism).
        :param verbose: Whether to show a progress bar.
        :param description: The description to show in the progress bar.
        """
        self._make_runner = init_fn
        self.num_jobs = num_jobs
        self.verbose = verbose
        self.description = description

    def _init_runner(self):
        pid = multiprocessing.current_process().pid
        self._runners[pid] = self._make_runner()

    def _process(self, *args, **kwargs):
        pid = multiprocessing.current_process().pid
        runner = self._runners[pid]
        return runner(*args, **kwargs)

    def __call__(self, items: Iterable, **kwargs) -> Generator:
        if self.num_jobs == 1:
            runner = self._make_runner()
            for item in tqdm(items, desc=self.description, disable=not self.verbose):
                yield runner(item, **kwargs)

        else:
            pool = ProcessPoolExecutor(
                max_workers=self.num_jobs,
                initializer=self._init_runner,
                mp_context=multiprocessing.get_context("spawn"),
            )

            with pool as executor:
                try:
                    res = executor.map(partial(self._process, **kwargs), items)
                    for item in tqdm(
                        res,
                        desc=self.description,
                        total=len(items),
                        disable=not self.verbose,
                    ):
                        yield item
                except KeyboardInterrupt as exc:  # pragma: no cover
                    pool.shutdown(wait=False)
                    if self.verbose:
                        print("Interrupted by the user.")
                    raise exc
                except Exception as exc:  # pragma: no cover
                    pool.shutdown(wait=False)
                    raise RuntimeError(
                        "Parallel processing failed. Please report this issue."
                    ) from exc
                finally:
                    self._runners.clear()
