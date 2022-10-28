from typing import List

from lhotse import CutSet
from lhotse.cut import Cut


class CutShardWriter:
    """
    CutShardWriter writes Cuts into multiple JSONL file shards.
    The JSONL can be compressed with gzip if the file extension ends with ``.gz``.

    Example::

        >>> with CutShardWriter("some_dir/cuts.%06d.jsonl.gz", shard_size=100) as w:
        ...     for cut in ...:
        ...         w.write(cut)

    It would create files such as ``some_dir/cuts.000000.jsonl.gz``, ``some_dir/cuts.000001.jsonl.gz``, etc.

    See also: :class:`~lhotse.shar.writers.tar.TarWriter`
    """

    def __init__(self, pattern: str, shard_size: int = 1000):
        self.pattern = pattern
        self.shard_size = shard_size
        self.reset()

    def reset(self):
        self.fname = None
        self.stream = None
        self.num_shards = 0
        self.num_items = 0
        self.num_items_total = 0

    def __enter__(self):
        self.reset()

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        if self.stream is not None:
            self.stream.close()

    def _next_stream(self):
        self.close()

        self.fname = self.pattern % self.num_shards
        self.stream = CutSet.open_writer(self.fname)

        self.num_shards += 1
        self.num_items = 0

    @property
    def output_paths(self) -> List[str]:
        return [self.pattern % i for i in range(self.num_shards)]

    def write(self, cut: Cut, flush: bool = False) -> None:
        if (
            # the first item written
            self.num_items_total == 0
            or (
                # desired shard size achieved
                self.num_items > 0
                and self.num_items % self.shard_size == 0
            )
        ):
            self._next_stream()

        self.stream.write(cut, flush=flush)
        self.num_items += 1
        self.num_items_total += 1
