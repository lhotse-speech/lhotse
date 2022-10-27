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
    """
    SharWriter writes cuts and their corresponding data into multiple shards,
    also recognized as the Lhotse Shar format.
    Each shard is numbered and represented as a collection of one text manifest and
    one or more binary tarfiles.
    Each tarfile contains a single type of data, e.g., recordings, features, or custom fields.

    The main idea behind Lhotse Shar format is to optimize dataloading with sequential reads,
    while keeping the data composition more flexible than e.g. WebDataset tar archives do.
    To achieve this, Lhotse Shar keeps each data type in a separate archive, along a single
    CutSet JSONL manifest.
    This way, the metadata can be investigated without iterating through the binary data.
    The format also allows iteration over a subset of fields, or extension of existing data
    with new fields.

    The user has to specify which fields should be saved, and what compression to use for each of them.
    Currently we support ``wav``, ``flac``, and ``mp3`` compression for ``recording`` and custom audio fields,
    and ``lilcom`` or ``numpy`` for ``features`` and custom array fields.

    Example::

        >>> cuts = CutSet(...)  # cuts have 'recording' and 'features'
        >>> with SharWriter("some_dir", shard_size=100, fields={"recording": "mp3", "features": "lilcom"}) as w:
        ...     for cut in cuts:
        ...         w.write(cut)

    It would create a directory ``some_dir`` with files such as ``some_dir/cuts.000000.jsonl.gz``,
    ``some_dir/recording.000000.tar``, ``some_dir/features.000000.tar``,
    and then the same names but numbered with ``000001``, etc.

    See also: :class:`~lhotse.shar.writers.tar.TarWriter`, :class:`~lhotse.shar.writers.audio.AudioTarWriter`,
        :class:`~lhotse.shar.writers.array.ArrayTarWriter`.
    """

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
        self.close()

    def close(self):
        for w in self.writers.values():
            w.close()

    def write(self, cut: Cut) -> None:

        # handle audio
        if cut.has_recording and "recording" in self.fields:
            data = cut.load_audio()
            recording = to_placeholder(cut.recording)
            self.writers["recording"].write(
                cut.id, data, cut.sampling_rate, manifest=recording
            )
            cut = fastcopy(cut, recording=recording)
        elif cut.has_recording and self.warn_unused_fields:
            warnings.warn(
                "Found cut with 'recording' field that is not specified for Shar writing."
            )

        # handle features
        if cut.has_features and "features" in self.fields:
            data = cut.load_features()
            features = to_placeholder(cut.features)
            self.writers["features"].write(cut.id, data, manifest=features)
            cut = fastcopy(cut, features=features)
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
                placeholder_obj = to_placeholder(getattr(cut, key))
                kwargs = {}
                if isinstance(val, Recording):
                    kwargs["sampling_rate"] = val.sampling_rate
                self.writers[key].write(
                    cut.id, data, manifest=placeholder_obj, **kwargs
                )
                cut = fastcopy(cut, custom=cut.custom.copy())
                setattr(cut, key, placeholder_obj)

        self.writers["cut"].write(cut)


Manifest = TypeVar("Manifest", Recording, Features, Array, TemporalArray)


def to_placeholder(manifest: Manifest) -> Manifest:
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
