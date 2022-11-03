from pathlib import Path
from typing import Optional, TypeVar, Union

from lhotse import AudioSource, Features, Recording, fastcopy
from lhotse.array import Array, TemporalArray
from lhotse.cut import Cut
from lhotse.utils import Pathlike

Manifest = TypeVar("Manifest", Recording, Features, Array, TemporalArray)


def to_shar_placeholder(manifest: Manifest) -> Manifest:
    if isinstance(manifest, Recording):
        assert (
            len(manifest.sources) == 1
        ), "Multiple AudioSources are not supported yet."
        # TODO: modify Recording's start/duration/num_samples if needed to match the Cut (in case we read subset of audio)
        return fastcopy(
            manifest,
            sources=[
                AudioSource(type="shar", channels=src.channels, source="")
                for src in manifest.sources
            ],
        )
    # TODO: modify Features/TemporalArray's start/duration/num_frames if needed to match the Cut (in case we read subset of array)
    elif isinstance(manifest, (Array, Features)):
        return fastcopy(manifest, storage_type="shar", storage_path="", storage_key="")
    elif isinstance(manifest, TemporalArray):
        return fastcopy(
            manifest,
            array=fastcopy(
                manifest.array, storage_type="shar", storage_path="", storage_key=""
            ),
        )


def fill_shar_placeholder(
    manifest: Union[Cut, Recording, Features, Array, TemporalArray],
    data: bytes,
    tarpath: Pathlike,
    field: Optional[str] = None,
) -> None:
    if isinstance(manifest, Cut):
        assert (
            field is not None
        ), "'field' argument must be provided when filling a Shar placeholder in a Cut."
        manifest = getattr(manifest, field)
        fill_shar_placeholder(
            manifest=manifest, field=field, data=data, tarpath=tarpath
        )

    tarpath = Path(tarpath)

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
