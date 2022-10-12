import warnings
from functools import partial
from typing import Any, Callable, Dict, Type, TypeVar, Union

from typing_extensions import Literal

from lhotse import AudioSource, Features, fastcopy
from lhotse.array import Array, TemporalArray
from lhotse.audio import Recording
from lhotse.cut import Cut
from lhotse.shar.writers.array import ArrayTarWriter
from lhotse.shar.writers.audio import AudioTarWriter
from lhotse.shar.writers.cut import CutShardWriter
from lhotse.utils import Pathlike

WriterName = Literal["wav", "flac", "mp3", "lilcom", "numpy"]
FieldWriterInstance = Union[AudioTarWriter, ArrayTarWriter]
FieldWriter = Type[FieldWriterInstance]


class SharWriter:
    def __init__(
        self,
        output_dir: Pathlike,
        fields: Dict[
            str,
            Union[str, FieldWriter, Callable[[Any], FieldWriterInstance]],
        ],
        shard_size: int = 1000,
        warn_unused_fields: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.fields = fields
        self.warn_unused_fields = warn_unused_fields

        self.writers = {
            "cut": CutShardWriter(
                pattern=f"{self.output_dir}/cuts.%06d.jsonl.gz", shard_size=shard_size
            ),
        }
        for field, writer_type in self.fields.items():
            writer_type = resolve_writer(writer_type)
            self.writers[field] = writer_type(
                pattern=f"{self.output_dir}/{field}.%06d.tar", shard_size=shard_size
            )

    def __enter__(self):
        for w in self.writers.values():
            w.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for w in self.writers.values():
            w.__exit__(exc_type, exc_val, exc_tb)

    def write(self, cut: Cut) -> None:

        # handle audio
        if cut.has_recording and "recording" in self.fields:
            # TODO: with/without transcoding
            data = cut.load_audio()
            self.writers["recording"].write(cut.id, data, cut.sampling_rate)
            cut.recording = to_placeholder(cut.recording)
        elif cut.has_recording and self.warn_unused_fields:
            warnings.warn(
                "Found cut with 'recording' field that is not specified for Shar writing."
            )

        # handle features
        if cut.has_features and "features" in self.fields:
            data = cut.load_features()
            self.writers["features"].write(cut.id, data)
            cut.features = to_placeholder(cut.features)
        elif cut.has_features and self.warn_unused_fields:
            warnings.warn(
                "Found cut with 'features' field that is not specified for Shar writing."
            )

        # handle custom
        if hasattr(cut, "custom"):
            for key, val in cut.custom.items():
                if not isinstance(val, (Array, TemporalArray, Recording)):
                    continue

                if key not in self.fields:
                    if self.warn_unused_fields:
                        warnings.warn(
                            f"Found cut with '{key}' field that is not specified for Shar writing."
                        )
                    continue

                data = cut.load_custom(key)
                kwargs = {}
                if isinstance(val, Recording):
                    kwargs["sampling_rate"] = val.sampling_rate
                self.writers[key].write(cut.id, data, **kwargs)
                setattr(cut, key, to_placeholder(getattr(cut, key)))

        self.writers["cut"].write(cut)


Manifest = TypeVar("Manifest", Recording, Features, Array, TemporalArray)


def to_placeholder(manifest: Manifest) -> Manifest:
    if isinstance(manifest, Recording):
        return fastcopy(
            manifest,
            sources=[
                AudioSource(type="shar", channels=src.channels, source="")
                for src in manifest.sources
            ],
        )
    elif isinstance(manifest, (Array, Features)):
        return fastcopy(manifest, storage_type="shar", storage_path="", storage_key="")
    elif isinstance(manifest, TemporalArray):
        return fastcopy(
            manifest,
            array=fastcopy(
                manifest.array, storage_type="shar", storage_path="", storage_key=""
            ),
        )


def resolve_writer(
    name_or_callable: Union[str, FieldWriter, Callable[[Any], FieldWriterInstance]]
) -> FieldWriter:
    if not isinstance(name_or_callable, str):
        return name_or_callable

    opts = {
        "wav": partial(AudioTarWriter, format="wav"),
        "flac": partial(AudioTarWriter, format="flac"),
        "mp3": partial(AudioTarWriter, format="mp3"),
        "lilcom": partial(ArrayTarWriter, compression="lilcom"),
        "numpy": partial(ArrayTarWriter, compression="numpy"),
    }
    assert (
        name_or_callable in opts
    ), f"Unknown field type (got: '{name_or_callable}', we support only: {', '.join(opts)}"
    return opts[name_or_callable]
