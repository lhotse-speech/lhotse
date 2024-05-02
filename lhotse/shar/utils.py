from pathlib import Path
from typing import Optional, TypeVar, Union

from lhotse import AudioSource, Features, Recording, compute_num_samples, fastcopy
from lhotse.array import Array, TemporalArray
from lhotse.cut import Cut
from lhotse.utils import Pathlike

Manifest = TypeVar("Manifest", Recording, Features, Array, TemporalArray)


def to_shar_placeholder(manifest: Manifest, cut: Optional[Cut] = None) -> Manifest:
    if isinstance(manifest, Recording):
        return fastcopy(
            manifest,
            # Creates a single AudioSource out of multiple ones.
            sources=[
                AudioSource(type="shar", channels=manifest.channel_ids, source="")
            ],
            # Removes the transform metadata because they were already executed.
            transforms=None,
            duration=cut.duration if cut is not None else manifest.duration,
            num_samples=compute_num_samples(cut.duration, manifest.sampling_rate)
            if cut is not None
            else manifest.num_samples,
        )
    elif isinstance(manifest, Array):
        return fastcopy(manifest, storage_type="shar", storage_path="", storage_key="")
    elif isinstance(manifest, Features):
        return fastcopy(
            manifest,
            start=0,
            duration=cut.duration if cut is not None else manifest.duration,
            storage_type="shar",
            storage_path="",
            storage_key="",
        )
    elif isinstance(manifest, TemporalArray):
        return fastcopy(
            manifest,
            start=0,
            array=fastcopy(
                manifest.array,
                storage_type="shar",
                storage_path="",
                storage_key="",
            ),
        )
    else:
        raise RuntimeError(f"Unexpected manifest type: {type(manifest)}")


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
        ), "We expected a single (possibly multi-channel) AudioSource in Shar format."
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
