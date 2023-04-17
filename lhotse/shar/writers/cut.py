import logging
from typing import List, Optional, Union

from lhotse.cut import Cut
from lhotse.serialization import SequentialJsonlWriter


class JsonlShardWriter:
    """
    JsonlShardWriter writes Cuts or dicts into multiple JSONL file shards.
    The JSONL can be compressed with gzip if the file extension ends with ``.gz``.

    Example::

        >>> with JsonlShardWriter("some_dir/cuts.%06d.jsonl.gz", shard_size=100) as w:
        ...     for cut in ...:
        ...         w.write(cut)

    It would create files such as ``some_dir/cuts.000000.jsonl.gz``, ``some_dir/cuts.000001.jsonl.gz``, etc.

    See also: :class:`~lhotse.shar.writers.tar.TarWriter`
    """

    def __init__(self, pattern: str, shard_size: Optional[int] = 1000):
        self.pattern = pattern
        if not self.sharding_enabled and shard_size is not None:
            logging.warning(
                "Sharding is disabled because `pattern` doesn't contain a formatting marker (e.g., '%06d'), "
                "but shard_size is not None - ignoring shard_size."
            )
        self.shard_size = shard_size
        self.reset()

    @property
    def sharding_enabled(self) -> bool:
        return "%" in self.pattern

    def reset(self):
        self.fname = None
        self.stream = None
        self.num_shards = 0
        self.num_items = 0
        self.num_items_total = 0

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        if self.stream is not None:
            self.stream.close()

    def _next_stream(self):
        self.close()

        if self.sharding_enabled:
            self.fname = self.pattern % self.num_shards
            self.num_shards += 1
        else:
            self.fname = self.pattern

        self.stream = SequentialJsonlWriter(self.fname)

        self.num_items = 0

    @property
    def output_paths(self) -> List[str]:
        if self.sharding_enabled:
            return [self.pattern % i for i in range(self.num_shards)]
        return [self.pattern]

    def write(self, data: Union[Cut, dict], flush: bool = False) -> None:
        if (
            # the first item written
            self.num_items_total == 0
            or (
                # desired shard size achieved
                self.sharding_enabled
                and self.num_items > 0
                and self.num_items % self.shard_size == 0
            )
        ):
            self._next_stream()

        self.stream.write(data, flush=flush)
        self.num_items += 1
        self.num_items_total += 1

    def write_placeholder(self, cut_id: str, flush: bool = False) -> None:
        self.write({"cut_id": cut_id}, flush=flush)
