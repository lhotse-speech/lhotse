from lhotse import CutSet
from lhotse.cut import Cut


class CutShardWriter:
    def __init__(self, pattern: str, shard_size: int = 1000):
        self.pattern = pattern
        self.shard_size = shard_size
        self.fname = None
        self.stream = None
        self.num_shards = 0
        self.num_items = 0
        self.num_items_total = 0

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        if self.stream is not None:
            self.stream.close()

    def _next_stream(self):
        self.close()

        # TODO: support gopen-like capabilities
        self.fname = self.pattern % self.num_shards
        self.stream = CutSet.open_writer(self.fname)

        self.num_shards += 1
        self.num_items = 0

    def write(self, cut: Cut, flush: bool = False) -> bool:
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

    def open_manifest(self):
        raise NotImplemented
