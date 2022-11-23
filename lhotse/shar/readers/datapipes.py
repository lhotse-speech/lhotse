from pathlib import Path
from typing import Generator, Iterable, Tuple

import lilcom
import numpy as np
import torch
import torchaudio

from lhotse.cut import Cut
from lhotse.lazy import LazyManifestIterator
from lhotse.shar.utils import fill_shar_placeholder
from lhotse.utils import Pathlike, is_module_available
from lhotse.workarounds import AltGzipFile

# This is to avoid a hard dependency on torchadata which is only available
# in later PyTorch versions.
if is_module_available("torchdata"):
    from torchdata.datapipes.iter import IterDataPipe, TarArchiveLoader, Zipper
else:
    IterDataPipe = object
    TarArchiveLoader = object
    Zipper = object


class AudioReader(IterDataPipe):
    def __init__(self, tarstream_iter: TarArchiveLoader) -> None:
        self.dp = tarstream_iter

    def __iter__(self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        for tarpath, tarstream in self.dp:
            blob = tarstream.read()
            load_fn = torchaudio.load
            if tarpath.endswith(".flac"):
                load_fn = torchaudio.backend.soundfile_backend.load
            _, audio = load_fn(blob)
            yield tarpath, audio


class ArrayReader(IterDataPipe):
    def __init__(self, tarstream_iter: TarArchiveLoader) -> None:
        self.dp = tarstream_iter

    def __iter__(self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        for tarpath, tarstream in self.dp:
            if tarpath.endswith(".llc"):
                arr = lilcom.decompress(tarstream.read())
            elif tarpath.endswith(".npy"):
                stream = AltGzipFile(fileobj=tarstream, mode="rb")
                arr = np.load(stream)
            else:
                raise RuntimeError(f"Unknown array/tensor format: {tarpath}")
            yield tarpath, torch.from_numpy(arr)


class CutsReader(IterDataPipe):
    def __init__(self, paths_iter: Iterable[Pathlike]) -> None:
        self.dp = paths_iter

    def __iter__(self) -> Generator[Cut, None, None]:
        for path in self.dp:
            self.cuts = LazyManifestIterator(path)
            yield from self.cuts


class SharReader(IterDataPipe):
    def __init__(self, *component_datapipes: IterDataPipe) -> None:
        raise NotImplementedError("We don't support this yet.")

        self.dps = sorted(
            component_datapipes, key=lambda dp: isinstance(dp, CutsReader), reverse=True
        )

        num_cut_readers_found = sum(int(isinstance(dp, CutsReader)) for dp in self.dps)
        assert (
            num_cut_readers_found == 1
        ), f"Expected exactly one CutsReader instance in the inputs to SharReader (found {num_cut_readers_found})"

        for dp in self.dps[1:]:
            assert isinstance(dp, TarArchiveLoader), (
                f"SharReader expects its inputs to be a single CutsReader followed by an arbitrary number "
                f"of TarArchiveLoader datapipes; other types are not supported (found: '{type(dp).__name__}')"
            )

        self.zipped = Zipper(*self.dps)

    def __iter__(self):
        for cut, *items in self.zipped:
            # TODO: add support for changes in Shar that require pairwise iteration over items in tarfiles
            for tarpath, tarstream in items:
                tarpath = Path(tarpath)
                item_id = tarpath.stem
                assert (
                    item_id == cut.id
                ), f"Mismatched elements: cut.id='{cut.id}' != item_id='{item_id}'"
                field = tarpath.parent.stem.split(".")[0]  # TODO: explain the format
                fill_shar_placeholder(
                    cut=cut, field=field, data=tarstream.read(), tarpath=tarpath
                )
            yield cut


def tar_datapipe(in_dir: str, pattern: str) -> TarArchiveLoader:
    import torchdata as td

    dp = td.datapipes.iter.FileLister(in_dir, pattern)
    dp = td.datapipes.iter.FileOpener(dp, mode="b")
    dp = td.datapipes.iter.TarArchiveLoader(dp, mode="r|*")
    return dp


def cut_datapipe(in_dir: str) -> CutsReader:
    import torchdata as td

    dp = td.datapipes.iter.FileLister(in_dir, "cuts.*.jsonl.gz")
    dp = CutsReader(dp)
    return dp


def load_shar_datapipe(in_dir: Pathlike) -> SharReader:
    """
    ``load_shar_datapipe`` reads cuts and their corresponding data from multiple shards,
    also recognized as the Lhotse Shar format.
    Each shard is numbered and represented as a collection of one text manifest and
    one or more binary tarfiles.
    Each tarfile contains a single type of data, e.g., recordings, features, or custom fields.

    .. note:: This function is experimental and uses the ``torchdata`` library
        to return a datapipe over cuts with attached data.

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

    Example::

        >>> cuts = load_shar_datapipe("some_dir")
        ... for cut in cuts:
        ...     print("Cut", cut.id, "has duration of", cut.duration)
        ...     audio = cut.load_audio()
        ...     fbank = cut.load_features()

    You can also use all of the ``torchdata`` datapipe methods, e.g.::

        >>> cuts = load_shar_datapipe("some_dir").shuffle().batch(10)

    See also: :class:`~lhotse.shar.writers.shar.SharWriter`, :class:`~lhotse.shar.readers.lazy.LazySharIterator`.
    """
    assert is_module_available("torchdata"), (
        "To use datapipe-based Shar reading API, you need to have torchdata installed "
        "(and a recent enough version of PyTorch, e.g. 1.12)."
    )

    # TODO: figure out how to make it work with cloud storage
    fields = set(p.stem.split(".")[0] for p in Path(in_dir).glob("*"))
    assert "cuts" in fields
    fields.remove("cuts")

    in_dir = str(in_dir)
    cutsdp = cut_datapipe(in_dir)
    tardps = [tar_datapipe(in_dir, f"{field}.*.tar") for field in fields]

    return SharReader(cutsdp, *tardps)
