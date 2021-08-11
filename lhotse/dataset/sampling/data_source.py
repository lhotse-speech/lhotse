import random
from collections import deque

from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling import streaming_shuffle


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

    def shuffle(self, seed: int) -> 'DataSource':
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

    def sort_like(self, other: 'DataSource') -> 'DataSource':
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

    def __iter__(self) -> 'DataSource':
        self.reset()
        self._iter = iter(self._shuffled_items)
        return self

    def __next__(self) -> Cut:
        if self._reusable:
            return self._reusable.popleft()
        return next(self._iter)

    def __len__(self) -> int:
        return len(self._shuffled_items)