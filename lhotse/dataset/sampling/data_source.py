import random
from collections import deque
from typing import Generator, Iterable

from lhotse import CutSet
from lhotse.cut import Cut


class DataSource:
    """
    An iterator wrapper over CutSet that helps with the sampling process:
    it allows for deterministic re-shuffling of elements and "returning"
    sampled elements to be yielded again.
    """

    def __init__(self, items: CutSet):
        self._orig_items = items
        self._shuffled_items = self._orig_items
        self._iter = None
        self._reusable = deque()

    def shuffle(self, seed: int) -> "DataSource":
        """
        Shuffles the elements using the provided random seed value.
        When the input CutSet is lazy, we use a streaming variant of
        shuffle, that may be less random.
        """
        self.reset()
        r = random.Random(seed)
        if self._orig_items.is_lazy:
            self._shuffled_items = streaming_shuffle(iter(self._orig_items), rng=r)
        else:
            self._shuffled_items = self._orig_items.shuffle(rng=r)
        return self

    def sort_like(self, other: "DataSource") -> "DataSource":
        """
        Sorts the underlying CutSet to provide Cuts in the same order of cut_ids
        as the other DataSource.
        """
        self.reset()
        self._shuffled_items = self._orig_items.sort_like(other._shuffled_items)
        return self

    def take_back(self, cut: Cut) -> None:
        """Push the cut in front of other cuts to be sampled again."""
        self._reusable.append(cut)

    def reset(self) -> None:
        """Reset the iterable state of DataSource."""
        self._iter = None
        self._reusable.clear()

    def __iter__(self) -> "DataSource":
        self.reset()
        self._iter = iter(self._shuffled_items)
        return self

    def __next__(self) -> Cut:
        if self._reusable:
            return self._reusable.popleft()
        return next(self._iter)

    def __len__(self) -> int:
        return len(self._shuffled_items)


def streaming_shuffle(
        data: Iterable[Cut],
        bufsize: int = 10000,
        rng: random.Random = random,
) -> Generator[Cut, None, None]:
    """
    Shuffle the data in the stream.

    This uses a buffer of size ``bufsize``. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.

    This code is mostly borrowed from WebDataset; note that we use much larger default
    buffer size because Cuts are very lightweight and fast to read.
    https://github.com/webdataset/webdataset/blob/master/webdataset/iterators.py#L145

    .. warning: The order of the elements is expected to be much less random than
        if the whole sequence was shuffled before-hand with standard methods like
        ``random.shuffle``.

    :param data: iterator
    :param bufsize: buffer size for shuffling
    :param rng: either random module or random.Random instance
    :return: a generator of cuts, shuffled on-the-fly.
    """
    buf = []
    startup = True
    for sample in data:
        if len(buf) < bufsize:
            try:
                buf.append(next(data))
            except StopIteration:
                pass
        k = rng.randint(0, len(buf) - 1)
        sample, buf[k] = buf[k], sample
        if startup and len(buf) < bufsize:
            buf.append(sample)
            continue
        startup = False
        yield sample
    for sample in buf:
        yield sample
