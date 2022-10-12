from pathlib import Path
from typing import Generator, Iterable, Tuple

import lilcom
import numpy as np
import torch
import torchaudio

from lhotse import CutSet, Recording
from lhotse.array import Array, TemporalArray
from lhotse.cut import Cut
from lhotse.features import Features
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
            self.cuts = CutSet.from_jsonl_lazy(path)
            yield from self.cuts


class SharReader(IterDataPipe):
    def __init__(self, *component_datapipes: IterDataPipe) -> None:
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
            for tarpath, tarstream in items:
                tarpath = Path(tarpath)
                item_id = tarpath.stem
                assert (
                    item_id == cut.id
                ), f"Mismatched elements: cut.id='{cut.id}' != item_id='{item_id}'"
                field = tarpath.parent.stem.split(".")[0]  # TODO: explain the format
                fill_placeholder(
                    cut=cut, field=field, data=tarstream.read(), tarpath=tarpath
                )
            yield cut


def fill_placeholder(cut: Cut, field: str, data: bytes, tarpath: Path) -> None:
    manifest = getattr(cut, field)
    if isinstance(manifest, Recording):
        assert (
            len(manifest.sources) == 1
        ), "Multiple AudioSources are not supported yet."
        manifest.sources[0].type = "memory"
        manifest.sources[0].source = data
    elif isinstance(manifest, (Features, Array)):
        manifest.storage_key = data
        if tarpath.suffix == ".llc":
            manifest.storage_type = "memory_lilcom"
        elif tarpath.suffix == ".npy":
            manifest.storage_type = "memory_npy"
        else:
            raise RuntimeError(f"Unknown array/tensor format: {tarpath}")
    elif isinstance(manifest, TemporalArray):
        manifest.array.storage_key = data
        if tarpath.suffix == ".llc":
            manifest.array.storage_type = "memory_lilcom"
        elif tarpath.suffix == ".npy":
            manifest.array.storage_type = "memory_npy"
        else:
            raise RuntimeError(f"Unknown array/tensor format: {tarpath}")
    else:
        raise RuntimeError(f"Unknown manifest type: {type(manifest).__name__}")


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


def load_shar(in_dir: Pathlike) -> SharReader:
    # TODO: figure out how to make it work with cloud storage
    fields = set(p.stem.split(".")[0] for p in Path(in_dir).glob("*"))
    assert "cuts" in fields
    fields.remove("cuts")

    in_dir = str(in_dir)
    cutsdp = cut_datapipe(in_dir)
    tardps = [tar_datapipe(in_dir, f"{field}.*.tar") for field in fields]

    return SharReader(cutsdp, *tardps)
