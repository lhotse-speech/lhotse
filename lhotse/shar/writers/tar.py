import tarfile
from io import BytesIO

from lhotse.serialization import open_best


class TarWriter:
    """
    TarWriter is a convenience wrapper over :class:`tarfile.TarFile` that
    allows writing binary data into tar files that are automatically segmented.
    Each segment is a separate tar file called a "shard."

    Shards are useful in training of deep learning models that require a substantial
    amount of data. Each shard can be read sequentially, which allows faster reads
    from magnetic disks, NFS, or otherwise slow storage.

    Example::

        >>> with TarWriter("some_dir/data.%06d.tar", shard_size=100) as w:
        ...     w.write("blob1", binary_blob1)
        ...     w.write("blob2", binary_blob2)  # etc.

    It would create files such as ``some_dir/data.000000.tar``, ``some_dir/data.000001.tar``, etc.

    This class is heavily inspired by the WebDataset library:
    https://github.com/webdataset/webdataset
    """

    def __init__(self, pattern: str, shard_size: int):
        self.pattern = pattern
        assert "%" in self.pattern
        self.shard_size = shard_size
        self.gzip = pattern.endswith(".gz")
        self.reset()

    def reset(self):
        self.fname = None
        self.stream = None
        self.tarstream = None
        self.num_shards = 0
        self.num_items = 0
        self.num_items_total = 0

    def __enter__(self):
        self.reset()
        return self

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
        self.stream = open_best(self.fname, "wb")
        self.tarstream = tarfile.open(
            fileobj=self.stream, mode="w|gz" if self.gzip else "w|"
        )

        self.num_shards += 1
        self.num_items = 0

    def write(self, key: str, data: BytesIO, count: bool = True):
        if count and (
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
        if count:
            self.num_items += 1
            self.num_items_total += 1
