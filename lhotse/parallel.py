import threading
from concurrent.futures import ProcessPoolExecutor
from queue import Queue


def parallel_map(fn, iterable, num_jobs: int = 1):
    # thread = threading.Thread(target=submitter, args=(fn, iterable, queue, num_jobs))
    thread = SubmitterThread(fn, iterable, num_jobs)
    thread.run()
    queue = thread.queue

    while thread.is_alive() or not queue.empty():
        yield queue.get(block=True).result()

    thread.join()


class SubmitterThread(threading.Thread):
    def __init__(self, fn, iterable, num_jobs) -> None:
        super().__init__()
        self.fn = fn
        self.num_jobs = num_jobs
        self.iterable = iterable
        self.queue = Queue(maxsize=5000)

    def run(self) -> None:
        with ProcessPoolExecutor(self.num_jobs) as ex:
            for item in self.iterable:
                future = ex.submit(self.fn, item)
                self.queue.put(future, block=True)
