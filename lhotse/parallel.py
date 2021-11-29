import threading
import queue
from concurrent.futures import ProcessPoolExecutor


def parallel_map(fn, *iterables, num_jobs: int = 1, queue_size: int = 5000):
    thread = SubmitterThread(fn, *iterables, num_jobs=num_jobs, queue_size=queue_size)
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
        self, fn, *iterables, num_jobs: int = 1, queue_size: int = 5000
    ) -> None:
        super().__init__()
        self.fn = fn
        self.num_jobs = num_jobs
        self.iterables = iterables
        self.queue = queue.Queue(maxsize=queue_size)

    def run(self) -> None:
        with ProcessPoolExecutor(self.num_jobs) as ex:
            for args in zip(*self.iterables):
                future = ex.submit(self.fn, *args)
                self.queue.put(future, block=True)


def foo(*iz):
    ret = "item"
    for i in iz:
        ret = ret + f'_{i}'
    return ret


def it():
    from time import sleep

    for i in range(100):
        yield i
        sleep(0.01)


if __name__ == "__main__":
    for x in parallel_map(foo, it(), it(), num_jobs=3):
        print(x)
