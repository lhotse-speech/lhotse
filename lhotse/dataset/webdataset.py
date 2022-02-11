import pickle
import sys
from pathlib import Path
from typing import Dict

from tqdm.auto import tqdm

from lhotse import CutSet
from lhotse.utils import Pathlike, is_module_available


def export_to_webdataset(cuts: CutSet, output_path: Pathlike, verbose: bool = True) -> None:
    if not is_module_available("webdataset"):
        raise ImportError("Please 'pip install webdataset' first.")
    import webdataset as wds

    output_path = Path(output_path).with_suffix(".tar")

    with wds.TarWriter(str(output_path)) as sink:
        for idx, cut in tqdm(enumerate(cuts), desc="Creating WDS tarball", disable=not verbose):
            cut = cut.move_to_memory()
            data = pickle.dumps(cut.to_dict())
            sink.write({"__key__": cut.id, "data": data})


class LazyWebdatasetIterator:
    """
    LazyWebdatasetIterator provides the ability to read Lhotse objects from a
    WebDataset tarball on-the-fly, without reading its full contents into memory.

    This class is designed to be a partial "drop-in" replacement for ordinary dicts
    to support lazy loading of RecordingSet, SupervisionSet and CutSet.
    Since it does not support random access reads, some methods of these classes
    might not work properly.
    """

    def __init__(self, path: Pathlike) -> None:
        from lhotse.serialization import extension_contains

        self.path = path
        assert extension_contains(".tar", self.path)

    def _reset(self) -> None:
        import webdataset as wds

        self._ds = wds.WebDataset(self.path)
        self._ds_iter = iter(self._ds)

    def __getstate__(self):
        """
        Store the state for pickling -- we'll only store the path, and re-initialize
        this iterator when unpickled. This is necessary to transfer this object across processes
        for PyTorch's DataLoader workers.
        """
        state = {"path": self.path}
        return state

    def __setstate__(self, state: Dict):
        """Restore the state when unpickled -- open the jsonl file again."""
        self.__dict__.update(state)

    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        from lhotse.serialization import deserialize_item

        data_dict = next(self._ds_iter)
        data = pickle.loads(data_dict["data"])
        item = deserialize_item(data)
        return item

    def values(self):
        yield from self

    def keys(self):
        return (item.id for item in self)

    def items(self):
        return ((item.id, item) for item in self)

    def __add__(self, other) -> "LazyIteratorChain":
        from lhotse.serialization import LazyIteratorChain

        return LazyIteratorChain(self, other)

