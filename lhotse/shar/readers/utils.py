from pathlib import Path

from lhotse import Features, Recording
from lhotse.array import Array, TemporalArray
from lhotse.cut import Cut


def fill_shar_placeholder(cut: Cut, field: str, data: bytes, tarpath: Path) -> None:
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
