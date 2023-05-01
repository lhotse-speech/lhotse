import logging
import tarfile
from io import BytesIO
from typing import List, Optional

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

    It's also possible to use ``TarWriter`` with automatic sharding disabled::

        >>> with TarWriter("some_dir/data.tar", shard_size=None) as w:
        ...     w.write("blob1", binary_blob1)
        ...     w.write("blob2", binary_blob2)  # etc.

    This class is heavily inspired by the WebDataset library:
    https://github.com/webdataset/webdataset
    """

    def __init__(self, pattern: str, shard_size: Optional[int] = 1000):
        self.pattern = str(pattern)
        if self.sharding_enabled and shard_size is None:
            raise RuntimeError(
                "shard_size must be specified when sharding is enabled via a formatting marker such as '%06d'"
            )
        if not self.sharding_enabled and shard_size is not None:
            logging.warning(
                "Sharding is disabled because `pattern` doesn't contain a formatting marker (e.g., '%06d'), "
                "but shard_size is not None - ignoring shard_size."
            )
        self.shard_size = shard_size
        self.gzip = self.pattern.endswith(".gz")
        self.reset()

    @property
    def sharding_enabled(self) -> bool:
        return "%" in self.pattern

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

        if self.sharding_enabled:
            self.fname = self.pattern % self.num_shards
            self.num_shards += 1
        else:
            self.fname = self.pattern

        self.stream = open_best(self.fname, "wb")
        self.tarstream = tarfile.open(
            fileobj=self.stream, mode="w|gz" if self.gzip else "w|"
        )

        self.num_items = 0

    @property
    def output_paths(self) -> List[str]:
        if self.sharding_enabled:
            return [self.pattern % i for i in range(self.num_shards)]
        return [self.pattern]

    def write(self, key: str, data: BytesIO, count: bool = True):
        if count and (
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

        ti = tarfile.TarInfo(key)
        data.seek(0)
        ti.size = len(data.getvalue())
        self.tarstream.addfile(ti, data)
        if count:
            self.num_items += 1
            self.num_items_total += 1
