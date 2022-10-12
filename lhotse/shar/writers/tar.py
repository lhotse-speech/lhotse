import tarfile
from io import BytesIO


class TarWriter:
    def __init__(self, pattern: str, shard_size: int):
        self.pattern = pattern
        assert "%" in self.pattern
        self.shard_size = shard_size
        self.gzip = pattern.endswith(".gz")
        self.fname = None
        self.stream = None
        self.tarstream = None
        self.num_shards = 0
        self.num_items = 0
        self.num_items_total = 0

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        if self.tarstream is not None:
            self.tarstream.close()
        if self.stream is not None:
            self.stream.close()

    def _next_stream(self):
        self.close()

        self.fname = self.pattern % self.num_shards

        # TODO: support gopen-like capabilities
        self.stream = open(self.fname, "wb")

        self.tarstream = tarfile.open(
            fileobj=self.stream, mode="w|gz" if self.gzip else "w"
        )

        self.num_shards += 1
        self.num_items = 0

    def write(self, key: str, data: BytesIO):
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

        ti = tarfile.TarInfo(key)
        data.seek(0)
        ti.size = len(data.getvalue())
        self.tarstream.addfile(ti, data)
        self.num_items += 1
        self.num_items_total += 1
