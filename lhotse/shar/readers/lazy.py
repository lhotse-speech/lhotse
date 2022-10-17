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
    TODO: describe LazySharIterator
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
