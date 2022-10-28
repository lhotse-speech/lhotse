import random
import tarfile
from pathlib import Path
from typing import Dict, Generator, Optional, Sequence, Tuple

from lhotse.lazy import (
    ImitatesDict,
    LazyIteratorChain,
    LazyJsonlIterator,
    count_newlines_fast,
)
from lhotse.serialization import (
    decode_json_line,
    deserialize_item,
    extension_contains,
    load_jsonl,
    open_best,
)
from lhotse.shar.readers.utils import fill_shar_placeholder
from lhotse.utils import Pathlike, exactly_one_not_null, pairwise


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

    We can simply load a directory created by :class:`~lhotse.shar.writers.shar.SharWriter`.
    Example::

    >>> cuts = LazySharIterator(in_dir="some_dir")
    ... for cut in cuts:
    ...     print("Cut", cut.id, "has duration of", cut.duration)
    ...     audio = cut.load_audio()
    ...     fbank = cut.load_features()

    :class:`.LazySharIterator` can also be initialized from a dict, where the keys
    indicate fields to be read, and the values point to actual shard locations.
    This is useful when only a subset of data is needed, or it is stored in different
    directories. Example::

    >>> cuts = LazySharIterator({
    ...     "cuts": ["some_dir/cuts.000000.jsonl.gz"],
    ...     "recording": ["another_dir/recording.000000.tar"],
    ...     "features": ["yet_another_dir/features.000000.tar"],
    ... })
    ... for cut in cuts:
    ...     print("Cut", cut.id, "has duration of", cut.duration)
    ...     audio = cut.load_audio()
    ...     fbank = cut.load_features()

    We also support providing shell commands as shard sources, inspired by WebDataset.
    Example::

    >>> cuts = LazySharIterator({
    ...     "cuts": ["pipe:curl https://my.page/cuts.000000.jsonl.gz"],
    ...     "recording": ["pipe:curl https://my.page/recording.000000.tar"],
    ... })
    ... for cut in cuts:
    ...     print("Cut", cut.id, "has duration of", cut.duration)
    ...     audio = cut.load_audio()

    :param fields: a dict whose keys specify which fields to load,
        and values are lists of shards (either paths or shell commands).
        The field "cuts" pointing to CutSet shards always has to be present.
    :param in_dir: path to a directory created with ``SharWriter`` with
        all the shards in a single place. Can be used instead of ``fields``.
    :param split_for_dataloading: bool, by default ``False`` which does nothing.
        Setting it to ``True`` is intended for PyTorch training with multiple
        dataloader workers and possibly multiple DDP nodes.
        It results in each node+worker combination receiving a unique subset
        of shards from which to read data to avoid data duplication.
    :param shuffle_shards: bool, by default ``False``. When ``True``, the shards
        are shuffled (in case of multi-node training, the shuffling is the same
        on each node given the same seed).
    :param seed: When ``shuffle_shards`` is ``True``, we use this number to
        seed the RNG.

    See also: :class:`~lhotse.shar.writers.shar.SharWriter`
    """

    def __init__(
        self,
        fields: Optional[Dict[str, Sequence[Pathlike]]] = None,
        in_dir: Optional[Pathlike] = None,
        split_for_dataloading: bool = False,
        shuffle_shards: bool = False,
        seed: int = 42,
    ) -> None:
        assert exactly_one_not_null(
            fields, in_dir
        ), "To read Lhotse Shar format, provide either 'in_dir' or 'fields' argument."

        self.split_for_dataloading = split_for_dataloading
        self._len = None
        if in_dir is not None:
            self._init_from_dir(in_dir)
        else:
            self._init_from_inputs(fields)

        self.num_shards = len(self.streams["cuts"])
        for field in self.fields:
            assert (
                len(self.streams[field]) == self.num_shards
            ), f"Expected {self.num_shards} shards available for field '{field}' but found {len(self.streams[field])}"

        self.shards = [
            {field: self.streams[field][shard_idx] for field in self.streams}
            for shard_idx in range(self.num_shards)
        ]

        if shuffle_shards:
            random.Random(seed).shuffle(self.shards)

    def _init_from_inputs(self, fields: Optional[Dict[str, Sequence[str]]] = None):
        assert (
            "cuts" in fields
        ), "To initialize Shar reader, please provide the value for key 'cuts' in 'fields'."
        self.fields = set(fields.keys())
        self.fields.remove("cuts")
        self.streams = fields

    def _init_from_dir(self, in_dir: Pathlike):
        self.in_dir = Path(in_dir)

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
        for field in self.fields:
            self.streams[field] = sorted(
                p for p in all_paths if p.name.split(".")[0] == field
            )

    @property
    def shards_for_dataloading(self):
        from .utils import split_by_node, split_by_worker

        return split_by_worker(split_by_node(self.shards))

    def __iter__(self):
        shards = (
            self.shards_for_dataloading if self.split_for_dataloading else self.shards
        )
        for shard in shards:
            # TODO: more careful open/close using some ctxmanager and with statement

            # Iterate over cuts for the current shard
            cuts = LazyJsonlIterator(shard["cuts"])

            # Iterate over tarfiles containing data for specific fields of each cut
            tarpaths = {field: path for field, path in shard.items() if field != "cuts"}

            # Open every tarfile so it's ready for streaming
            tars = {
                field: tarfile.open(fileobj=open_best(path, mode="rb"), mode="r|*")
                for field, path in tarpaths.items()
            }

            # *tardata contains all fields for a single cut (recording, features, array, etc.)
            for cut, *tardata in zip(
                cuts, *map(iterate_tarfile_pairwise, tars.values())
            ):
                for (
                    field,
                    tar_f,
                    (
                        (data_bytes, data_path),
                        (manifest_bytes, manifest_path),
                    ),
                ) in zip(
                    tars.keys(),
                    tars.values(),
                    tardata,
                ):
                    assert (
                        data_path.stem == cut.id
                    ), f"Mismatched IDs: cut ID is '{cut.id}' but found data with name '{data_path}'"
                    manifest = deserialize_item(
                        decode_json_line(manifest_bytes.decode("utf-8"))
                    )
                    setattr(cut, field, manifest)
                    fill_shar_placeholder(
                        cut,
                        field=field,
                        data=data_bytes,
                        tarpath=data_path,
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


def iterate_tarfile_pairwise(
    tar_file: tarfile.TarFile,
) -> Generator[Tuple[bytes, Path], None, None]:
    result = []
    for tarinfo in tar_file:
        if len(result) == 2:
            yield tuple(result)
            result = []
        result.append(parse_tarinfo(tarinfo, tar_file))

    if len(result) == 2:
        yield tuple(result)

    if len(result) == 1:
        raise RuntimeError(
            "Uneven number of files in the tarfile (expected to iterate pairs of binary data + JSON metadata."
        )


def parse_tarinfo(
    tarinfo: tarfile.TarInfo, tar_file: tarfile.TarFile
) -> Tuple[bytes, Path]:
    """
    Parse a tarinfo object and return the data it points to as well as the internal path.
    """
    data = tar_file.extractfile(tarinfo).read()
    path = Path(tarinfo.path)
    return data, path
