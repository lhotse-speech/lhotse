import warnings
from functools import partial
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

from typing_extensions import Literal

from lhotse import AudioSource, Features, fastcopy
from lhotse.array import Array, TemporalArray
from lhotse.audio import Recording
from lhotse.cut import Cut
from lhotse.shar.utils import to_shar_placeholder
from lhotse.shar.writers.array import ArrayTarWriter
from lhotse.shar.writers.audio import AudioTarWriter
from lhotse.shar.writers.cut import JsonlShardWriter
from lhotse.utils import Pathlike, ifnone

WriterName = Literal["wav", "flac", "mp3", "lilcom", "numpy", "jsonl"]
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

    When ``shard_size`` is set to ``None``, we will disable automatic sharding and the
    shard number suffix will be omitted from the file names.

    The option ``warn_unused_fields`` will emit a warning when cuts have some data attached to them
    (e.g., recording, features, or custom arrays) but saving it was not specified via ``fields``.

    The option ``include_cuts`` controls whether we store the cuts alongside ``fields`` (true by default).
    Turning it off is useful when extending existing dataset with new fields/feature types,
    but the original cuts do not require any modification.

    See also: :class:`~lhotse.shar.writers.tar.TarWriter`, :class:`~lhotse.shar.writers.audio.AudioTarWriter`,
        :class:`~lhotse.shar.writers.array.ArrayTarWriter`.
    """

    def __init__(
        self,
        output_dir: Pathlike,
        fields: Dict[str, str],
        shard_size: Optional[int] = 1000,
        warn_unused_fields: bool = True,
        include_cuts: bool = True,
        shard_suffix: Optional[str] = None,
    ) -> None:
        self.output_dir = str(output_dir)
        self.shard_size = shard_size
        self.fields = fields
        self.warn_unused_fields = warn_unused_fields
        self.include_cuts = include_cuts
        if self.sharding_enabled:
            assert (
                shard_suffix is None
            ), f"shard_suffix must be None when shard_size is specified (got: '{shard_suffix}')."
            self.shard_suffix = ".%06d"
        else:
            self.shard_suffix = ifnone(shard_suffix, "")

        self.writers = {}
        if include_cuts:
            self.writers["cuts"] = JsonlShardWriter(
                pattern=_create_cuts_output_url(self.output_dir, self.shard_suffix),
                shard_size=self.shard_size,
            )
        for field, writer_type in self.fields.items():
            make_writer_fn, ext = resolve_writer(writer_type)
            self.writers[field] = make_writer_fn(
                pattern=f"{self.output_dir}/{field}{self.shard_suffix}{ext}",
                shard_size=self.shard_size,
            )

    @property
    def sharding_enabled(self) -> bool:
        return self.shard_size is not None and self.shard_size > 0

    @property
    def output_paths(self) -> Dict[str, List[str]]:
        return {k: w.output_paths for k, w in self.writers.items()}

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

        # Handle audio.
        if "recording" in self.fields:
            if cut.has_recording:
                data = cut.load_audio()
                recording = to_shar_placeholder(cut.recording, cut)
                self.writers["recording"].write(
                    cut.id, data, cut.sampling_rate, manifest=recording
                )
                cut = fastcopy(cut, recording=recording)
            else:
                self.writers["recording"].write_placeholder(cut.id)
        elif cut.has_recording and self.warn_unused_fields:
            warnings.warn(
                "Found cut with 'recording' field that is not specified for Shar writing."
            )

        # Handle features.
        if "features" in self.fields:
            if cut.has_features:
                data = cut.load_features()
                features = to_shar_placeholder(cut.features, cut)
                self.writers["features"].write(cut.id, data, manifest=features)
                cut = fastcopy(cut, features=features)
            else:
                self.writers["features"].write_placeholder(cut.id)
        elif cut.has_features and self.warn_unused_fields:
            warnings.warn(
                "Found cut with 'features' field that is not specified for Shar writing."
            )

        # Handle custom data and non-data attributes.
        for key in self.fields:

            # Skip fields already taken care of.
            if key in ["recording", "features"]:
                continue

            # Check if the custom attribute is available: if yes
            if cut.has_custom(key):
                val = getattr(cut, key)
                if not isinstance(val, (Array, TemporalArray, Recording)):
                    assert isinstance(
                        self.writers[key], JsonlShardWriter
                    ), f"Expected writer type 'jsonl' (got '{self.fields[key]}') for non-data field '{key}'."
                    self.writers[key].write({"cut_id": cut.id, key: val})
                else:
                    data = cut.load_custom(key)
                    placeholder_obj = to_shar_placeholder(val, cut)
                    kwargs = {}
                    if isinstance(val, Recording):
                        kwargs["sampling_rate"] = val.sampling_rate
                    self.writers[key].write(
                        cut.id, data, manifest=placeholder_obj, **kwargs
                    )
                    cut = fastcopy(cut, custom=cut.custom.copy())
                    setattr(cut, key, placeholder_obj)
            else:
                self.writers[key].write_placeholder(cut.id)

        # Warn in case there is some data that wasn't requested to be saved.
        for key, val in ifnone(cut.custom, {}).items():
            if not isinstance(val, (Array, TemporalArray, Recording)):
                continue
            if key not in self.fields:
                if self.warn_unused_fields:
                    warnings.warn(
                        f"Found cut with '{key}' field that is not specified for Shar writing."
                    )
                continue

        # We will write only the relevant subset of the binary data alongside cut,
        # so we need to update the offset (start).
        cut = fastcopy(cut, start=0)

        if "cuts" in self.writers:
            self.writers["cuts"].write(cut)


def resolve_writer(name: str) -> Tuple[FieldWriter, str]:
    opts = {
        "wav": (partial(AudioTarWriter, format="wav"), ".tar"),
        "flac": (partial(AudioTarWriter, format="flac"), ".tar"),
        "mp3": (partial(AudioTarWriter, format="mp3"), ".tar"),
        "lilcom": (partial(ArrayTarWriter, compression="lilcom"), ".tar"),
        "numpy": (partial(ArrayTarWriter, compression="numpy"), ".tar"),
        "jsonl": (JsonlShardWriter, ".jsonl.gz"),
    }
    assert (
        name in opts
    ), f"Unknown field type (got: '{name}', we support only: {', '.join(opts)}"
    return opts[name]


def _create_cuts_output_url(base_output_url: str, shard_suffix: str) -> str:

    # special case where we want to ensure the CutSet actually gets gzipped
    if base_output_url.startswith("pipe:"):
        base_output_url = base_output_url.replace("pipe:", "pipe:gzip -c | ")

    return f"{base_output_url}/cuts{shard_suffix}.jsonl.gz"
