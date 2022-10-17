import random
import tarfile
from pathlib import Path

from lhotse.lazy import (
    ImitatesDict,
    LazyIteratorChain,
    LazyJsonlIterator,
    count_newlines_fast,
)
from lhotse.serialization import extension_contains, open_best
from lhotse.shar.readers.utils import fill_shar_placeholder
from lhotse.utils import Pathlike


class LazySharIterator(ImitatesDict):
    """
    LazySharIterator reads cuts and their corresponding data from multiple shards,
    also recognized as the Lhotse Shar format.
    Each shard is numbered and represented as a collection of one text manifest and
    one or more binary tarfiles.
    Each tarfile contains a single type of data, e.g., recordings, features, or custom fields.

    Given an example directory named ``some_dir`, its expected layout is
    ``some_dir/cuts.000000.jsonl.gz``, ``some_dir/recording.000000.tar``,
    ``some_dir/features.000000.tar``, and then the same names but numbered with ``000001``, etc.
    There may also be other files if the cuts have custom data attached to them.

    The main idea behind Lhotse Shar format is to optimize dataloading with sequential reads,
    while keeping the data composition more flexible than e.g. WebDataset tar archives do.
    To achieve this, Lhotse Shar keeps each data type in a separate archive, along a single
    CutSet JSONL manifest.
    This way, the metadata can be investigated without iterating through the binary data.
    The format also allows iteration over a subset of fields, or extension of existing data
    with new fields.

    As you iterate over cuts from ``LazySharIterator``, it keeps a file handle open for the
    JSONL manifest and all of the tar files that correspond to the current shard.
    The tar files are read item by item together, and their binary data is attached to
    the cuts.
    It can be normally accessed using methods such as ``cut.load_audio()``.

    Example::

    >>> cuts = LazySharIterator("some_dir")
    ... for cut in cuts:
    ...     print("Cut", cut.id, "has duration of", cut.duration)
    ...     audio = cut.load_audio()
    ...     fbank = cut.load_features()

    See also: :class:`~lhotse.shar.writers.shar.SharWriter`
    """

    def __init__(
        self, in_dir: Pathlike, shuffle_shards: bool = False, seed: int = 42
    ) -> None:
        self.in_dir = Path(in_dir)
        self._len = None

        # TODO: make it work with cloud storage
        all_paths = list(self.in_dir.glob("*"))
        self.fields = set(p.stem.split(".")[0] for p in all_paths)
        assert "cuts" in self.fields
        self.fields.remove("cuts")

        self.streams = {
            "cuts": sorted(
                p
                for p in all_paths
                if p.name.split(".")[0] == "cuts" and extension_contains(".jsonl", p)
            )
        }
        num_shards = len(self.streams["cuts"])
        for field in self.fields:
            self.streams[field] = sorted(
                p for p in all_paths if p.name.split(".")[0] == field
            )
            assert (
                len(self.streams[field]) == num_shards
            ), f"Expected {num_shards} shards available for field '{field}' but found {len(self.streams[field])}"

        self.shards = [
            {field: self.streams[field][shard_idx] for field in self.streams}
            for shard_idx in range(num_shards)
        ]
        if shuffle_shards:
            random.Random(seed).shuffle(self.shards)

    def __iter__(self):
        for shard in self.shards:
            # TODO: more careful open/close using some ctxmanager and with statement
            cuts = LazyJsonlIterator(shard["cuts"])
            tarpaths = {field: path for field, path in shard.items() if field != "cuts"}
            tars = {
                field: tarfile.open(fileobj=open_best(path, mode="rb"), mode="r|*")
                for field, path in tarpaths.items()
            }
            for cut, *tarinfos in zip(cuts, *tars.values()):
                for field, tar_f, tarinfo in zip(tars.keys(), tars.values(), tarinfos):
                    assert (
                        Path(tarinfo.path).stem == cut.id
                    ), f"Mismatched IDs: cut ID is '{cut.id}' but found data with name '{tarinfo.path}'"
                    fill_shar_placeholder(
                        cut,
                        field=field,
                        data=tar_f.extractfile(tarinfo).read(),
                        tarpath=tarinfo.path,
                    )
                yield cut
            for tar_f in tars.values():
                tar_f.close()

    def __len__(self) -> int:
        if self._len is None:
            self._len = sum(count_newlines_fast(p) for p in self.streams["cuts"])
        return self._len

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)
