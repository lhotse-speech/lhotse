import itertools
import logging
import pickle
import random
import secrets
import warnings
from collections import defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from functools import partial, reduce
from itertools import chain, islice
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch
from intervaltree import IntervalTree
from tqdm.auto import tqdm

from lhotse.audio import RecordingSet, null_result_on_audio_loading_error
from lhotse.augmentation import AugmentFn
from lhotse.cut.base import Cut
from lhotse.cut.data import DataCut
from lhotse.cut.mixed import MixedCut, MixTrack
from lhotse.cut.mono import MonoCut
from lhotse.cut.multi import MultiCut
from lhotse.cut.padding import PaddingCut
from lhotse.features import FeatureExtractor, Features, FeatureSet
from lhotse.features.base import StatsAccumulator, compute_global_stats
from lhotse.features.io import FeaturesWriter, LilcomChunkyWriter
from lhotse.lazy import (
    AlgorithmMixin,
    Dillable,
    LazyFlattener,
    LazyIteratorChain,
    LazyManifestIterator,
    LazyMapper,
    LazySlicer,
    T,
)
from lhotse.serialization import Serializable
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    DEFAULT_PADDING_VALUE,
    LOG_EPSILON,
    Decibels,
    Pathlike,
    Seconds,
    compute_num_frames,
    compute_num_samples,
    exactly_one_not_null,
    fastcopy,
    ifnone,
    split_manifest_lazy,
    split_sequence,
    uuid4,
)

FW = TypeVar("FW", bound=FeaturesWriter)


def is_cut(example) -> bool:
    return isinstance(example, Cut)


class CutSet(Serializable, AlgorithmMixin):
    """
    :class:`~lhotse.cut.CutSet` represents a collection of cuts.
    CutSet ties together all types of data -- audio, features and supervisions, and is suitable to represent
    training/dev/test sets.

    CutSet can be either "lazy" (acts as an iterable) which is best for representing full datasets,
    or "eager" (acts as a list), which is best for representing individual mini-batches (and sometimes test/dev datasets).
    Almost all operations are available for both modes, but some of them are more efficient depending on the mode
    (e.g. indexing an "eager" manifest is O(1)).

    .. note::
        :class:`~lhotse.cut.CutSet` is the basic building block of PyTorch-style Datasets for speech/audio processing tasks.

    When coming from Kaldi, there is really no good equivalent -- the closest concept may be Kaldi's "egs" for training
    neural networks, which are chunks of feature matrices and corresponding alignments used respectively as inputs and
    supervisions. :class:`~lhotse.cut.CutSet` is different because it provides you with all kinds of metadata,
    and you can select just the interesting bits to feed them to your models.

    :class:`~lhotse.cut.CutSet` can be created from any combination of :class:`~lhotse.audio.RecordingSet`,
    :class:`~lhotse.supervision.SupervisionSet`, and :class:`~lhotse.features.base.FeatureSet`
    with :meth:`lhotse.cut.CutSet.from_manifests`::

        >>> from lhotse import CutSet
        >>> cuts = CutSet.from_manifests(recordings=my_recording_set)
        >>> cuts2 = CutSet.from_manifests(features=my_feature_set)
        >>> cuts3 = CutSet.from_manifests(
        ...     recordings=my_recording_set,
        ...     features=my_feature_set,
        ...     supervisions=my_supervision_set,
        ... )

    When creating a :class:`.CutSet` with :meth:`.CutSet.from_manifests`, the resulting cuts will have the same duration
    as the input recordings or features. For long recordings, it is not viable for training.
    We provide several methods to transform the cuts into shorter ones.

    Consider the following scenario::

                          Recording
        |-------------------------------------------|
        "Hey, Matt!"     "Yes?"        "Oh, nothing"
        |----------|     |----|        |-----------|

        .......... CutSet.from_manifests() ..........
                            Cut1
        |-------------------------------------------|

        ............. Example CutSet A ..............
            Cut1          Cut2              Cut3
        |----------|     |----|        |-----------|

        ............. Example CutSet B ..............
                  Cut1                  Cut2
        |---------------------||--------------------|

        ............. Example CutSet C ..............
                     Cut1        Cut2
                    |---|      |------|

    The CutSet's A, B and C can be created like::

        >>> cuts_A = cuts.trim_to_supervisions()
        >>> cuts_B = cuts.cut_into_windows(duration=5.0)
        >>> cuts_C = cuts.trim_to_unsupervised_segments()

    .. note::
        Some operations support parallel execution via an optional ``num_jobs`` parameter.
        By default, all processing is single-threaded.

    .. caution::
        Operations on cut sets are not mutating -- they return modified copies of :class:`.CutSet` objects,
        leaving the original object unmodified (and all of its cuts are also unmodified).

    :class:`~lhotse.cut.CutSet` can be stored and read from JSON, JSONL, etc. and supports optional gzip compression::

        >>> cuts.to_file('cuts.jsonl.gz')
        >>> cuts4 = CutSet.from_file('cuts.jsonl.gz')

    It behaves similarly to a ``dict``::

            >>> 'rec1-1-0' in cuts
            True
            >>> cut = cuts['rec1-1-0']
            >>> for cut in cuts:
            >>>    pass
            >>> len(cuts)
            127

    :class:`~lhotse.cut.CutSet` has some convenience properties and methods to gather information about the dataset::

        >>> ids = list(cuts.ids)
        >>> speaker_id_set = cuts.speakers
        >>> # The following prints a message:
        >>> cuts.describe()
        Cuts count: 547
        Total duration (hours): 326.4
        Speech duration (hours): 79.6 (24.4%)
        ***
        Duration statistics (seconds):
        mean    2148.0
        std      870.9
        min      477.0
        25%     1523.0
        50%     2157.0
        75%     2423.0
        max     5415.0
        dtype: float64


    Manipulation examples::

        >>> longer_than_5s = cuts.filter(lambda c: c.duration > 5)
        >>> first_100 = cuts.subset(first=100)
        >>> split_into_4 = cuts.split(num_splits=4)
        >>> shuffled = cuts.shuffle()
        >>> random_sample = cuts.sample(n_cuts=10)
        >>> new_ids = cuts.modify_ids(lambda c: c.id + '-newid')

    These operations can be composed to implement more complex operations, e.g.
    bucketing by duration:

        >>> buckets = cuts.sort_by_duration().split(num_splits=30)

    Cuts in a :class:`.CutSet` can be detached from parts of their metadata::

        >>> cuts_no_feat = cuts.drop_features()
        >>> cuts_no_rec = cuts.drop_recordings()
        >>> cuts_no_sup = cuts.drop_supervisions()

    Sometimes specific sorting patterns are useful when a small CutSet represents a mini-batch::

        >>> cuts = cuts.sort_by_duration(ascending=False)
        >>> cuts = cuts.sort_like(other_cuts)

    :class:`~lhotse.cut.CutSet` offers some batch processing operations::

        >>> cuts = cuts.pad(num_frames=300)  # or duration=30.0
        >>> cuts = cuts.truncate(max_duration=30.0, offset_type='start')  # truncate from start to 30.0s
        >>> cuts = cuts.mix(other_cuts, snr=[10, 30], mix_prob=0.5)

    :class:`~lhotse.cut.CutSet` supports lazy data augmentation/transformation methods which require adjusting some information
    in the manifest (e.g., ``num_samples`` or ``duration``).
    Note that in the following examples, the audio is untouched -- the operations are stored in the manifest,
    and executed upon reading the audio::

        >>> cuts_sp = cuts.perturb_speed(factor=1.1)
        >>> cuts_vp = cuts.perturb_volume(factor=2.)
        >>> cuts_24k = cuts.resample(24000)
        >>> cuts_rvb = cuts.reverb_rir(rir_recordings)

    .. caution::
        If the :class:`.CutSet` contained :class:`~lhotse.features.base.Features` manifests, they will be
        detached after performing audio augmentations such as :meth:`.CutSet.perturb_speed`,
        :meth:`.CutSet.resample`, :meth:`.CutSet.perturb_volume`, or :meth:`.CutSet.reverb_rir`.

    :class:`~lhotse.cut.CutSet` offers parallel feature extraction capabilities
    (see `meth`:.CutSet.compute_and_store_features: for details),
    and can be used to estimate global mean and variance::

        >>> from lhotse import Fbank
        >>> cuts = CutSet()
        >>> cuts = cuts.compute_and_store_features(
        ...     extractor=Fbank(),
        ...     storage_path='/data/feats',
        ...     num_jobs=4
        ... )
        >>> mvn_stats = cuts.compute_global_feature_stats('/data/features/mvn_stats.pkl', max_cuts=10000)

    See also:

        - :class:`~lhotse.cut.Cut`
    """

    def __init__(self, cuts: Optional[Iterable[Cut]] = None) -> None:
        self.cuts = ifnone(cuts, [])

    def __eq__(self, other: "CutSet") -> bool:
        return self.cuts == other.cuts

    @property
    def data(self) -> Iterable[Cut]:
        """Alias property for ``self.cuts``"""
        return self.cuts

    @property
    def mixed_cuts(self) -> "CutSet":
        return CutSet.from_cuts(cut for cut in self.cuts if isinstance(cut, MixedCut))

    @property
    def simple_cuts(self) -> "CutSet":
        return CutSet.from_cuts(cut for cut in self.cuts if isinstance(cut, MonoCut))

    @property
    def multi_cuts(self) -> "CutSet":
        return CutSet.from_cuts(cut for cut in self.cuts if isinstance(cut, MultiCut))

    @property
    def ids(self) -> Iterable[str]:
        return (c.id for c in self.cuts)

    @property
    def speakers(self) -> FrozenSet[str]:
        return frozenset(
            supervision.speaker for cut in self for supervision in cut.supervisions
        )

    @staticmethod
    def from_files(
        paths: List[Pathlike], shuffle_iters: bool = True, seed: Optional[int] = None
    ) -> "CutSet":
        """
        Constructor that creates a single CutSet out of many manifest files.
        We will iterate sequentially over each of the files, and by default we
        will randomize the file order every time CutSet is iterated.

        This is intended primarily for large datasets which are split into many small manifests,
        to ensure that the order in which data is seen during training can be properly randomized.

        :param paths: a list of paths to cut manifests.
        :param shuffle_iters: bool, should we shuffle `paths` each time we iterate the returned
            CutSet (enabled by default).
        :param seed: int, random seed controlling the shuffling RNG.
            By default, we'll use Python's global RNG so the order
            will be different on each script execution.
        :return: a lazy CutSet instance.
        """
        return CutSet(
            LazyIteratorChain(
                *(LazyManifestIterator(p) for p in paths),
                shuffle_iters=shuffle_iters,
                seed=seed,
            )
        )

    @staticmethod
    def from_cuts(cuts: Iterable[Cut]) -> "CutSet":
        """Left for backward compatibility, where it implicitly created an "eager" CutSet."""
        return CutSet(list(cuts))

    from_items = from_cuts

    @staticmethod
    def from_manifests(
        recordings: Optional[RecordingSet] = None,
        supervisions: Optional[SupervisionSet] = None,
        features: Optional[FeatureSet] = None,
        output_path: Optional[Pathlike] = None,
        random_ids: bool = False,
        tolerance: Seconds = 0.001,
        lazy: bool = False,
    ) -> "CutSet":
        """
        Create a CutSet from any combination of supervision, feature and recording manifests.
        At least one of ``recordings`` or ``features`` is required.

        The created cuts will be of type :class:`.MonoCut`, even when the recordings have multiple channels.
        The :class:`.MonoCut` boundaries correspond to those found in the ``features``, when available,
        otherwise to those found in the ``recordings``.

        When ``supervisions`` are provided, we'll be searching them for matching recording IDs
        and attaching to created cuts, assuming they are fully within the cut's time span.

        :param recordings: an optional :class:`~lhotse.audio.RecordingSet` manifest.
        :param supervisions: an optional :class:`~lhotse.supervision.SupervisionSet` manifest.
        :param features: an optional :class:`~lhotse.features.base.FeatureSet` manifest.
        :param output_path: an optional path where the :class:`.CutSet` is stored.
        :param random_ids: boolean, should the cut IDs be randomized. By default, use the recording ID
            with a loop index and a channel idx, i.e. "{recording_id}-{idx}-{channel}")
        :param tolerance: float, tolerance for supervision and feature segment boundary comparison.
            By default, it's 1ms. Increasing this value can be helpful when importing Kaldi data
            directories with precomputed features (typically 0.02 - 0.1 should be sufficient).
        :param lazy: boolean, when ``True``, output_path must be provided
        :return: a new :class:`.CutSet` instance.
        """
        if lazy:
            return create_cut_set_lazy(
                recordings=recordings,
                supervisions=supervisions,
                features=features,
                output_path=output_path,
                random_ids=random_ids,
                tolerance=tolerance,
            )
        else:
            return create_cut_set_eager(
                recordings=recordings,
                supervisions=supervisions,
                features=features,
                output_path=output_path,
                random_ids=random_ids,
                tolerance=tolerance,
            )

    @staticmethod
    def from_dicts(data: Iterable[dict]) -> "CutSet":
        return CutSet.from_cuts(deserialize_cut(cut) for cut in data)

    @staticmethod
    def from_webdataset(
        path: Union[Pathlike, Sequence[Pathlike]], **wds_kwargs
    ) -> "CutSet":
        """
        Provides the ability to read Lhotse objects from a WebDataset tarball (or a
        collection of them, i.e., shards) sequentially, without reading the full contents
        into memory. It also supports passing a list of paths, or WebDataset-style pipes.

        CutSets stored in this format are potentially much faster to read from due to
        sequential I/O (we observed speedups of 50-100x vs random-read mechanisms).

        Since this mode does not support random access reads, some methods of CutSet
        might not work properly (e.g. ``len()``).

        The behaviour of the underlying ``WebDataset`` instance can be customized by
        providing its kwargs directly to the constructor of this class. For details,
        see :func:`lhotse.dataset.webdataset.mini_webdataset` documentation.

        **Examples**

        Read manifests and data from a single tarball::

            >>> cuts = CutSet.from_webdataset("data/cuts-train.tar")

        Read manifests and data from a multiple tarball shards::

            >>> cuts = CutSet.from_webdataset("data/shard-{000000..004126}.tar")
            >>> # alternatively
            >>> cuts = CutSet.from_webdataset(["data/shard-000000.tar", "data/shard-000001.tar", ...])

        Read manifests and data from shards in cloud storage (here AWS S3 via AWS CLI)::

            >>> cuts = CutSet.from_webdataset("pipe:aws s3 cp data/shard-{000000..004126}.tar -")

        Read manifests and data from shards which are split between PyTorch DistributeDataParallel
        nodes and dataloading workers, with shard-level shuffling enabled::

            >>> cuts = CutSet.from_webdataset(
            ...     "data/shard-{000000..004126}.tar",
            ...     split_by_worker=True,
            ...     split_by_node=True,
            ...     shuffle_shards=True,
            ... )

        """
        from lhotse.dataset.webdataset import LazyWebdatasetIterator

        return CutSet(cuts=LazyWebdatasetIterator(path, **wds_kwargs))

    @staticmethod
    def from_shar(
        fields: Optional[Dict[str, Sequence[Pathlike]]] = None,
        in_dir: Optional[Pathlike] = None,
        split_for_dataloading: bool = False,
        shuffle_shards: bool = False,
        stateful_shuffle: bool = True,
        seed: Union[int, Literal["randomized"]] = 42,
        cut_map_fns: Optional[Sequence[Callable[[Cut], Cut]]] = None,
    ) -> "CutSet":
        """
        Reads cuts and their corresponding data from multiple shards,
        also recognized as the Lhotse Shar format.
        Each shard is numbered and represented as a collection of one text manifest and
        one or more binary tarfiles.
        Each tarfile contains a single type of data, e.g., recordings, features, or custom fields.

        Given an example directory named ``some_dir``, its expected layout is
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
        locations. Example::

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
        The "cuts" field expects a .jsonl stream, while the other fields expect a .tar stream.
        Example::

            >>> cuts = LazySharIterator({
            ...     "cuts": ["pipe:curl https://my.page/cuts.000000.jsonl"]
            ...     "recording": ["pipe:curl https://my.page/recording.000000.tar"],
            ... })
            ... for cut in cuts:
            ...     print("Cut", cut.id, "has duration of", cut.duration)
            ...     audio = cut.load_audio()

        The shell command can also contain pipes, which can be used to e.g. decompressing.
        Example::

            >>> cuts = LazySharIterator({
            ...     "cuts": ["pipe:curl https://my.page/cuts.000000.jsonl.gz | gunzip -c -"],
                    (...)
            ... })

        Finally, we allow specifying URLs or cloud storage URIs for the shard sources.
        We defer to ``smart_open`` library to handle those.
        Example::

            >>> cuts = LazySharIterator({
            ...     "cuts": ["s3://my-bucket/cuts.000000.jsonl.gz"],
            ...     "recording": ["s3://my-bucket/recording.000000.tar"],
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
            This is mutually exclusive with ``seed='randomized'``.
        :param shuffle_shards: bool, by default ``False``. When ``True``, the shards
            are shuffled (in case of multi-node training, the shuffling is the same
            on each node given the same seed).
        :param seed: When ``shuffle_shards`` is ``True``, we use this number to
            seed the RNG.
            Seed can be set to ``'randomized'`` in which case we expect that the user provided
            :func:`lhotse.dataset.dataloading.worker_init_fn` as DataLoader's ``worker_init_fn``
            argument. It will cause the iterator to shuffle shards differently on each node
            and dataloading worker in PyTorch training. This is mutually exclusive with
            ``split_for_dataloading=True``.
            Seed can be set to ``'trng'`` which, like ``'randomized'``, shuffles the shards
            differently on each iteration, but is not possible to control (and is not reproducible).
            ``trng`` mode is mostly useful when the user has limited control over the training loop
            and may not be able to guarantee internal Shar epoch is being incremented, but needs
            randomness on each iteration (e.g. useful with PyTorch Lightning).
        :param stateful_shuffle: bool, by default ``False``. When ``True``, every
            time this object is fully iterated, it increments an internal epoch counter
            and triggers shard reshuffling with RNG seeded by ``seed`` + ``epoch``.
            Doesn't have any effect when ``shuffle_shards`` is ``False``.
        :param cut_map_fns: optional sequence of callables that accept cuts and return cuts.
            It's expected to have the same length as the number of shards, so each function
            corresponds to a specific shard.
            It can be used to attach shard-specific custom attributes to cuts.

        See also: :class:`~lhotse.shar.readers.lazy.LazySharIterator`,
            :meth:`~lhotse.cut.set.CutSet.to_shar`.
        """
        from lhotse.shar import LazySharIterator

        return CutSet(
            cuts=LazySharIterator(
                fields=fields,
                in_dir=in_dir,
                split_for_dataloading=split_for_dataloading,
                shuffle_shards=shuffle_shards,
                stateful_shuffle=stateful_shuffle,
                seed=seed,
                cut_map_fns=cut_map_fns,
            )
        )

    def to_shar(
        self,
        output_dir: Pathlike,
        fields: Dict[str, str],
        shard_size: Optional[int] = 1000,
        shard_offset: int = 0,
        warn_unused_fields: bool = True,
        include_cuts: bool = True,
        num_jobs: int = 1,
        fault_tolerant: bool = False,
        verbose: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Writes cuts and their corresponding data into multiple shards,
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
            >>> output_paths = cuts.to_shar(
            ...     "some_dir", shard_size=100, fields={"recording": "mp3", "features": "lilcom"}
            ... )

        It would create a directory ``some_dir`` with files such as ``some_dir/cuts.000000.jsonl.gz``,
        ``some_dir/recording.000000.tar``, ``some_dir/features.000000.tar``,
        and then the same names but numbered with ``000001``, etc.
        The starting shard offset can be set using ``shard_offset`` parameter. The writer starts from 0 by default.
        The function returns a dict that maps field names to lists of saved shard paths.

        When ``shard_size`` is set to ``None``, we will disable automatic sharding and the
        shard number suffix will be omitted from the file names.

        The option ``warn_unused_fields`` will emit a warning when cuts have some data attached to them
        (e.g., recording, features, or custom arrays) but saving it was not specified via ``fields``.

        The option ``include_cuts`` controls whether we store the cuts alongside ``fields`` (true by default).
        Turning it off is useful when extending existing dataset with new fields/feature types,
        but the original cuts do not require any modification.

        When ``num_jobs`` is greater than 1, we will first split the CutSet into shard CutSets,
        and then export the ``fields`` in parallel using multiple subprocesses. Enabling ``verbose``
        will display a progress bar.

        .. note:: It is recommended not to set ``num_jobs`` too high on systems with slow disks,
            as the export will likely be bottlenecked by I/O speed in these cases.
            Try experimenting with 4-8 jobs first.

        The option ``fault_tolerant`` will skip over audio files that failed to load with a warning.
        By default it is disabled.

        See also: :class:`~lhotse.shar.writers.shar.SharWriter`,
            :meth:`~lhotse.cut.set.CutSet.to_shar`.
        """
        assert num_jobs > 0 and isinstance(
            num_jobs, int
        ), f"The number of jobs must be an integer greater than 0 (got {num_jobs})."

        if num_jobs == 1:
            return _export_to_shar_single(
                cuts=self,
                output_dir=output_dir,
                shard_size=shard_size,
                shard_offset=shard_offset,
                fields=fields,
                warn_unused_fields=warn_unused_fields,
                include_cuts=include_cuts,
                shard_suffix=None,
                fault_tolerant=fault_tolerant,
                verbose=verbose,
            )

        progbar = partial(tqdm, desc="Shard progress") if verbose else lambda x: x
        shards = self.split_lazy(
            output_dir=output_dir,
            chunk_size=shard_size,
            prefix="cuts",
            num_digits=6,
            start_idx=shard_offset,
        )
        with ProcessPoolExecutor(num_jobs) as ex:
            futures = []
            output_paths = defaultdict(list)
            for idx, shard in enumerate(shards):
                futures.append(
                    ex.submit(
                        _export_to_shar_single,
                        cuts=shard,
                        output_dir=output_dir,
                        shard_size=None,  # already sharded
                        shard_offset=shard_offset,
                        fields=fields,
                        warn_unused_fields=warn_unused_fields,
                        include_cuts=True,
                        shard_suffix=f".{idx:06d}",
                        fault_tolerant=fault_tolerant,
                        verbose=False,
                        preload=True,
                    )
                )
            for f in progbar(as_completed(futures)):
                partial_paths = f.result()
                for k, v in partial_paths.items():
                    output_paths[k].extend(v)
        for k in output_paths:
            output_paths[k] = sorted(output_paths[k])
        return dict(output_paths)

    def to_dicts(self) -> Iterable[dict]:
        return (cut.to_dict() for cut in self)

    def decompose(
        self, output_dir: Optional[Pathlike] = None, verbose: bool = False
    ) -> Tuple[Optional[RecordingSet], Optional[SupervisionSet], Optional[FeatureSet]]:
        """
        Return a 3-tuple of unique (recordings, supervisions, features) found in
        this :class:`CutSet`. Some manifest sets may also be ``None``, e.g.,
        if features were not extracted.

        .. note:: :class:`.MixedCut` is iterated over its track cuts.

        :param output_dir: directory where the manifests will be saved.
            The following files will be created: 'recordings.jsonl.gz',
            'supervisions.jsonl.gz', 'features.jsonl.gz'.
        :param verbose: when ``True``, shows a progress bar.
        """
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        stored_rids = set()
        stored_sids = set()

        with RecordingSet.open_writer(
            output_dir / "recordings.jsonl.gz" if output_dir is not None else None
        ) as rw, SupervisionSet.open_writer(
            output_dir / "supervisions.jsonl.gz" if output_dir is not None else None
        ) as sw, FeatureSet.open_writer(
            output_dir / "features.jsonl.gz" if output_dir is not None else None
        ) as fw:

            def save(cut: DataCut):
                if cut.has_recording and cut.recording_id not in stored_rids:
                    rw.write(cut.recording)
                    stored_rids.add(cut.recording_id)
                if cut.has_features:
                    # Note: we have no way of saying if features are unique,
                    #       so we will always write them.
                    fw.write(cut.features)
                for sup in cut.supervisions:
                    if sup.id not in stored_sids:
                        # Supervisions inside cuts are relative to cuts start,
                        # so we correct the offset.
                        sw.write(sup.with_offset(cut.start))
                        stored_sids.add(sup.id)

            for cut in tqdm(self, desc="Decomposing cuts") if verbose else self:
                if isinstance(cut, DataCut):
                    save(cut)
                elif isinstance(cut, MixedCut):
                    for track in cut.tracks:
                        if isinstance(track.cut, DataCut):
                            save(track.cut)

        return rw.open_manifest(), sw.open_manifest(), fw.open_manifest()

    def describe(self, full: bool = False) -> None:
        """
        Print a message describing details about the ``CutSet`` - the number of cuts and the
        duration statistics, including the total duration and the percentage of speech segments.

        :param full: when ``True``, prints the full duration statistics, including % of speech
            by speaker count.

        Example output (for AMI train set):

        >>> cs.describe(full=True)

            Cut statistics:
            ╒═══════════════════════════╤══════════╕
            │ Cuts count:               │ 133      │
            ├───────────────────────────┼──────────┤
            │ Total duration (hh:mm:ss) │ 79:23:03 │
            ├───────────────────────────┼──────────┤
            │ mean                      │ 2148.7   │
            ├───────────────────────────┼──────────┤
            │ std                       │ 867.4    │
            ├───────────────────────────┼──────────┤
            │ min                       │ 477.9    │
            ├───────────────────────────┼──────────┤
            │ 25%                       │ 1509.8   │
            ├───────────────────────────┼──────────┤
            │ 50%                       │ 2181.7   │
            ├───────────────────────────┼──────────┤
            │ 75%                       │ 2439.9   │
            ├───────────────────────────┼──────────┤
            │ 99%                       │ 5300.7   │
            ├───────────────────────────┼──────────┤
            │ 99.5%                     │ 5355.3   │
            ├───────────────────────────┼──────────┤
            │ 99.9%                     │ 5403.2   │
            ├───────────────────────────┼──────────┤
            │ max                       │ 5415.2   │
            ├───────────────────────────┼──────────┤
            │ Recordings available:     │ 133      │
            ├───────────────────────────┼──────────┤
            │ Features available:       │ 0        │
            ├───────────────────────────┼──────────┤
            │ Supervisions available:   │ 102222   │
            ╘═══════════════════════════╧══════════╛
            Speech duration statistics:
            ╒══════════════════════════════╤══════════╤═══════════════════════════╕
            │ Total speech duration        │ 64:59:51 │ 81.88% of recording       │
            ├──────────────────────────────┼──────────┼───────────────────────────┤
            │ Total speaking time duration │ 74:33:09 │ 93.91% of recording       │
            ├──────────────────────────────┼──────────┼───────────────────────────┤
            │ Total silence duration       │ 14:23:12 │ 18.12% of recording       │
            ├──────────────────────────────┼──────────┼───────────────────────────┤
            │ Single-speaker duration      │ 56:18:24 │ 70.93% (86.63% of speech) │
            ├──────────────────────────────┼──────────┼───────────────────────────┤
            │ Overlapped speech duration   │ 08:41:28 │ 10.95% (13.37% of speech) │
            ╘══════════════════════════════╧══════════╧═══════════════════════════╛
            Speech duration statistics by number of speakers:
            ╒══════════════════════╤═══════════════════════╤════════════════════════════╤═══════════════╤══════════════════════╕
            │ Number of speakers   │ Duration (hh:mm:ss)   │ Speaking time (hh:mm:ss)   │ % of speech   │ % of speaking time   │
            ╞══════════════════════╪═══════════════════════╪════════════════════════════╪═══════════════╪══════════════════════╡
            │ 1                    │ 56:18:24              │ 56:18:24                   │ 86.63%        │ 75.53%               │
            ├──────────────────────┼───────────────────────┼────────────────────────────┼───────────────┼──────────────────────┤
            │ 2                    │ 07:51:44              │ 15:43:28                   │ 12.10%        │ 21.09%               │
            ├──────────────────────┼───────────────────────┼────────────────────────────┼───────────────┼──────────────────────┤
            │ 3                    │ 00:47:36              │ 02:22:47                   │ 1.22%         │ 3.19%                │
            ├──────────────────────┼───────────────────────┼────────────────────────────┼───────────────┼──────────────────────┤
            │ 4                    │ 00:02:08              │ 00:08:31                   │ 0.05%         │ 0.19%                │
            ├──────────────────────┼───────────────────────┼────────────────────────────┼───────────────┼──────────────────────┤
            │ Total                │ 64:59:51              │ 74:33:09                   │ 100.00%       │ 100.00%              │
            ╘══════════════════════╧═══════════════════════╧════════════════════════════╧═══════════════╧══════════════════════╛
        """
        from lhotse.cut.describe import CutSetStatistics

        stats = CutSetStatistics(full=full)
        stats.accumulate(self).describe()

    def split(
        self,
        num_splits: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> List["CutSet"]:
        """
        Split the :class:`~lhotse.CutSet` into ``num_splits`` pieces of equal size.

        :param num_splits: Requested number of splits.
        :param shuffle: Optionally shuffle the recordings order first.
        :param drop_last: determines how to handle splitting when ``len(seq)`` is not divisible
            by ``num_splits``. When ``False`` (default), the splits might have unequal lengths.
            When ``True``, it may discard the last element in some splits to ensure they are
            equally long.
        :return: A list of :class:`~lhotse.CutSet` pieces.
        """
        return [
            CutSet(subset)
            for subset in split_sequence(
                self,
                num_splits=num_splits,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        ]

    def split_lazy(
        self,
        output_dir: Pathlike,
        chunk_size: int,
        prefix: str = "",
        num_digits: int = 8,
        start_idx: int = 0,
    ) -> List["CutSet"]:
        """
        Splits a manifest (either lazily or eagerly opened) into chunks, each
        with ``chunk_size`` items (except for the last one, typically).

        In order to be memory efficient, this implementation saves each chunk
        to disk in a ``.jsonl.gz`` format as the input manifest is sampled.

        .. note:: For lowest memory usage, use ``load_manifest_lazy`` to open the
            input manifest for this method.

        :param it: any iterable of Lhotse manifests.
        :param output_dir: directory where the split manifests are saved.
            Each manifest is saved at: ``{output_dir}/{prefix}.{split_idx}.jsonl.gz``
        :param chunk_size: the number of items in each chunk.
        :param prefix: the prefix of each manifest.
        :param num_digits: the width of ``split_idx``, which will be left padded with zeros to achieve it.
        :param start_idx: The split index to start counting from (default is ``0``).
        :return: a list of lazily opened chunk manifests.
        """
        return split_manifest_lazy(
            self,
            output_dir=output_dir,
            chunk_size=chunk_size,
            prefix=prefix,
            num_digits=num_digits,
            start_idx=start_idx,
        )

    def subset(
        self,
        *,  # only keyword arguments allowed
        supervision_ids: Optional[Iterable[str]] = None,
        cut_ids: Optional[Iterable[str]] = None,
        first: Optional[int] = None,
        last: Optional[int] = None,
    ) -> "CutSet":
        """
        Return a new ``CutSet`` according to the selected subset criterion.
        Only a single argument to ``subset`` is supported at this time.

        Example:
            >>> cuts = CutSet.from_yaml('path/to/cuts')
            >>> train_set = cuts.subset(supervision_ids=train_ids)
            >>> test_set = cuts.subset(supervision_ids=test_ids)

        :param supervision_ids: List of supervision IDs to keep.
        :param cut_ids: List of cut IDs to keep.
            The returned :class:`.CutSet` preserves the order of `cut_ids`.
        :param first: int, the number of first cuts to keep.
        :param last: int, the number of last cuts to keep.
        :return: a new ``CutSet`` with the subset results.
        """
        assert exactly_one_not_null(
            supervision_ids, cut_ids, first, last
        ), "subset() can handle only one non-None arg."

        if first is not None:
            assert first > 0
            out = CutSet.from_cuts(islice(self, first))
            return out

        if last is not None:
            assert last > 0
            N = len(self)
            if last > N:
                return self
            return CutSet.from_cuts(islice(self, N - last, N))

        if supervision_ids is not None:
            # Remove cuts without supervisions
            supervision_ids = set(supervision_ids)
            return CutSet.from_cuts(
                cut.filter_supervisions(lambda s: s.id in supervision_ids)
                for cut in self
                if any(s.id in supervision_ids for s in cut.supervisions)
            )

        if cut_ids is not None:
            cut_ids = list(cut_ids)  # Remember the original order
            id_set = frozenset(cut_ids)  # Make a set for quick lookup
            # Iteration makes it possible to subset lazy manifests
            cuts = CutSet([cut for cut in self if cut.id in id_set])
            if len(cuts) < len(cut_ids):
                logging.warning(
                    f"In CutSet.subset(cut_ids=...): expected {len(cut_ids)} cuts but got {len(cuts)} "
                    f"instead ({len(cut_ids) - len(cuts)} cut IDs were not in the CutSet)."
                )
            # Restore the requested cut_ids order.
            return cuts.sort_like(cut_ids)

    def map(
        self,
        transform_fn: Callable[[T], T],
        apply_fn: Optional[Callable[[T], bool]] = is_cut,
    ) -> "CutSet":
        ans = CutSet(LazyMapper(self.data, fn=transform_fn, apply_fn=apply_fn))
        if self.is_lazy:
            return ans
        return ans.to_eager()

    def filter_supervisions(
        self, predicate: Callable[[SupervisionSegment], bool]
    ) -> "CutSet":
        """
        Return a new CutSet with Cuts containing only `SupervisionSegments` satisfying `predicate`

        Cuts without supervisions are preserved

        Example:
            >>> cuts = CutSet.from_yaml('path/to/cuts')
            >>> at_least_five_second_supervisions = cuts.filter_supervisions(lambda s: s.duration >= 5)

        :param predicate: A callable that accepts `SupervisionSegment` and returns bool
        :return: a CutSet with filtered supervisions
        """
        return self.map(partial(_filter_supervisions, predicate=predicate))

    def merge_supervisions(
        self,
        merge_policy: str = "delimiter",
        custom_merge_fn: Optional[Callable[[str, Iterable[Any]], Any]] = None,
    ) -> "CutSet":
        """
        Return a copy of the cut that has all of its supervisions merged into
        a single segment.

        The new start is the start of the earliest superivion, and the new duration
        is a minimum spanning duration for all the supervisions. The text fields of
        all segments are concatenated with a whitespace.

        :param merge_policy: one of "keep_first" or "delimiter". If "keep_first", we
            keep only the first segment's field value, otherwise all string fields
            (including IDs) are prefixed with "cat#" and concatenated with a hash symbol "#".
            This is also applied to ``custom`` fields. Fields with a ``None`` value are omitted.
        :param custom_merge_fn: a function that will be called to merge custom fields values.
            We expect ``custom_merge_fn`` to handle all possible custom keys.
            When not provided, we will treat all custom values as strings.
            It will be called roughly like:
            ``custom_merge_fn(custom_key, [s.custom[custom_key] for s in sups])``
        """
        return self.map(
            partial(
                _merge_supervisions,
                merge_policy=merge_policy,
                custom_merge_fn=custom_merge_fn,
            )
        )

    def trim_to_supervisions(
        self,
        keep_overlapping: bool = True,
        min_duration: Optional[Seconds] = None,
        context_direction: Literal["center", "left", "right", "random"] = "center",
        keep_all_channels: bool = False,
        num_jobs: int = 1,
    ) -> "CutSet":
        """
        Return a new CutSet with Cuts that have identical spans as their supervisions.

        For example, the following cut::

                    Cut
            |-----------------|
             Sup1
            |----|  Sup2
               |-----------|

        is transformed into two cuts::

             Cut1
            |----|
             Sup1
            |----|
               Sup2
               |-|
                    Cut2
               |-----------|
               Sup1
               |-|
                    Sup2
               |-----------|

        For the case of a multi-channel cut with multiple supervisions, we can either trim
        while respecting the supervision channels (in which case output cut has the same channels
        as the supervision) or ignore the channels (in which case output cut has the same channels
        as the input cut).

        :param keep_overlapping: when ``False``, it will discard parts of other supervisions that overlap with the
            main supervision. In the illustration above, it would discard ``Sup2`` in ``Cut1`` and ``Sup1`` in ``Cut2``.
            In this mode, we guarantee that there will always be exactly one supervision per cut.
        :param min_duration: An optional duration in seconds; specifying this argument will extend the cuts
            that would have been shorter than ``min_duration`` with actual acoustic context in the recording/features.
            If there are supervisions present in the context, they are kept when ``keep_overlapping`` is true.
            If there is not enough context, the returned cut will be shorter than ``min_duration``.
            If the supervision segment is longer than ``min_duration``, the return cut will be longer.
        :param context_direction: Which direction should the cut be expanded towards to include context.
            The value of "center" implies equal expansion to left and right;
            random uniformly samples a value between "left" and "right".
        :param keep_all_channels: If ``True``, the output cut will have the same channels as the input cut. By default,
            the trimmed cut will have the same channels as the supervision.
        :param num_jobs: Number of parallel workers to process the cuts.
        :return: a ``CutSet``.
        """

        if num_jobs == 1:
            return CutSet(
                LazyFlattener(
                    LazyMapper(
                        self,
                        partial(
                            _trim_to_supervisions_single,
                            keep_overlapping=keep_overlapping,
                            min_duration=min_duration,
                            context_direction=context_direction,
                            keep_all_channels=keep_all_channels,
                        ),
                    )
                )
            )

        from lhotse.manipulation import split_parallelize_combine

        result = split_parallelize_combine(
            num_jobs,
            self,
            _trim_to_supervisions_single,
            keep_overlapping=keep_overlapping,
            min_duration=min_duration,
            context_direction=context_direction,
            keep_all_channels=keep_all_channels,
        )
        return result

    def trim_to_alignments(
        self,
        type: str,
        max_pause: Seconds = 0.0,
        max_segment_duration: Optional[Seconds] = None,
        delimiter: str = " ",
        keep_all_channels: bool = False,
        num_jobs: int = 1,
    ) -> "CutSet":
        """
        Return a new CutSet with Cuts that have identical spans as the alignments of
        type `type`. An additional `max_pause` is allowed between the alignments to
        merge contiguous alignment items.

        For the case of a multi-channel cut with multiple alignments, we can either trim
        while respecting the supervision channels (in which case output cut has the same channels
        as the supervision) or ignore the channels (in which case output cut has the same channels
        as the input cut).

        :param type: The type of the alignment to trim to (e.g. "word").
        :param max_pause: The maximum pause allowed between the alignments to merge them.
        :param delimiter: The delimiter to use when concatenating the alignment items.
        :param keep_all_channels: If ``True``, the output cut will have the same channels as the input cut. By default,
            the trimmed cut will have the same channels as the supervision.
        :param num_jobs: Number of parallel workers to process the cuts.
        :return: a ``CutSet``.
        """

        if num_jobs == 1:
            return CutSet(
                LazyFlattener(
                    LazyMapper(
                        self,
                        partial(
                            _trim_to_alignments_single,
                            type=type,
                            max_pause=max_pause,
                            max_segment_duration=max_segment_duration,
                            delimiter=delimiter,
                            keep_all_channels=keep_all_channels,
                        ),
                    )
                )
            )

        from lhotse.manipulation import split_parallelize_combine

        result = split_parallelize_combine(
            num_jobs,
            self,
            _trim_to_alignments_single,
            type=type,
            max_pause=max_pause,
            max_segment_duration=max_segment_duration,
            delimiter=delimiter,
            keep_all_channels=keep_all_channels,
        )
        return result

    def trim_to_unsupervised_segments(self) -> "CutSet":
        """
        Return a new CutSet with Cuts created from segments that have no supervisions (likely
        silence or noise).

        :return: a ``CutSet``.
        """
        from lhotse.cut.describe import find_segments_with_speaker_count

        cuts = []
        for cut in self:
            segments = find_segments_with_speaker_count(
                cut, min_speakers=0, max_speakers=0
            )
            for span in segments:
                cuts.append(cut.truncate(offset=span.start, duration=span.duration))
        return CutSet(cuts)

    def trim_to_supervision_groups(
        self,
        max_pause: Optional[Seconds] = None,
        num_jobs: int = 1,
    ) -> "CutSet":
        """
        Return a new CutSet with Cuts based on supervision groups. A supervision group is
        a set of supervisions with no gaps between them (or gaps shorter than ``max_pause``).
        This is similar to the concept of an `utterance group` as described in this paper:
        https://arxiv.org/abs/2211.00482

        For example, the following cut::

                                                Cut
        ╔═════════════════════════════════════════════════════════════════════════════════╗
        ║┌──────────────────────┐                              ┌────────┐                 ║
        ║│ Hello this is John.  │                              │   Hi   │                 ║
        ║└──────────────────────┘                              └────────┘                 ║
        ║            ┌──────────────────────────────────┐            ┌───────────────────┐║
        ║            │     Hey, John. How are you?      │            │  What do you do?  │║
        ║            └──────────────────────────────────┘            └───────────────────┘║
        ╚═════════════════════════════════════════════════════════════════════════════════╝

        is transformed into two cuts::

                            Cut 1                                       Cut 2
        ╔════════════════════════════════════════════════╗    ╔═══════════════════════════╗
        ║┌──────────────────────┐                        ║    ║┌────────┐                 ║
        ║│ Hello this is John.  │                        ║    ║│   Hi   │                 ║
        ║└──────────────────────┘                        ║    ║└────────┘                 ║
        ║            ┌──────────────────────────────────┐║    ║      ┌───────────────────┐║
        ║            │     Hey, John. How are you?      │║    ║      │  What do you do?  │║
        ║            └──────────────────────────────────┘║    ║      └───────────────────┘║
        ╚════════════════════════════════════════════════╝    ╚═══════════════════════════╝

        For the case of a multi-channel cut with multiple supervisions, we keep all the channels
        in the recording.

        :param max_pause: An optional duration in seconds; if the gap between two supervisions
            is longer than this, they will be treated as separate groups.
        :param num_jobs: Number of parallel workers to process the cuts.
        :return: a ``CutSet``.
        """

        if num_jobs == 1:
            return CutSet(
                LazyFlattener(
                    LazyMapper(
                        self,
                        partial(
                            _trim_to_supervision_groups_single,
                            max_pause=max_pause,
                        ),
                    )
                )
            )

        from lhotse.manipulation import split_parallelize_combine

        result = split_parallelize_combine(
            num_jobs,
            self,
            _trim_to_supervision_groups_single,
            max_pause=max_pause,
        )
        return result

    def combine_same_recording_channels(self) -> "CutSet":
        """
        Find cuts that come from the same recording and have matching start and end times, but
        represent different channels. Then, combine them together to form MultiCut's and return
        a new ``CutSet`` containing these MultiCut's. This is useful for processing microphone array
        recordings.

        It is intended to be used as the first operation after creating a new ``CutSet`` (but
        might also work in other circumstances, e.g. if it was cut to windows first).

        Example:
            >>> ami = prepare_ami('path/to/ami')
            >>> cut_set = CutSet.from_manifests(recordings=ami['train']['recordings'])
            >>> multi_channel_cut_set = cut_set.combine_same_recording_channels()

        In the AMI example, the ``multi_channel_cut_set`` will yield MultiCuts that hold all single-channel
        Cuts together.
        """
        if self.mixed_cuts or self.multi_cuts:
            raise ValueError(
                "This operation is not applicable to CutSet's containing MixedCut's or MultiCut's."
            )
        from cytoolz.itertoolz import groupby

        groups = groupby(lambda cut: (cut.recording.id, cut.start, cut.end), self)
        return CutSet.from_cuts(MultiCut.from_mono(*cuts) for cuts in groups.values())

    def sort_by_recording_id(self, ascending: bool = True) -> "CutSet":
        """
        Sort the CutSet alphabetically according to 'recording_id'. Ascending by default.

        This is advantageous before caling `save_audios()` on a `trim_to_supervision()`
        processed `CutSet`, also make sure that `set_caching_enabled(True)` was called.
        """
        return CutSet(
            sorted(self, key=(lambda cut: cut.recording.id), reverse=not ascending)
        )

    def sort_by_duration(self, ascending: bool = False) -> "CutSet":
        """
        Sort the CutSet according to cuts duration and return the result. Descending by default.
        """
        return CutSet(
            sorted(self, key=(lambda cut: cut.duration), reverse=not ascending)
        )

    def sort_like(self, other: Union["CutSet", Sequence[str]]) -> "CutSet":
        """
        Sort the CutSet according to the order of cut IDs in ``other`` and return the result.
        """
        other_ids = list(other.ids if isinstance(other, CutSet) else other)
        assert set(self.ids) == set(
            other_ids
        ), "sort_like() expects both CutSet's to have identical cut IDs."
        index_map: Dict[str, int] = {v: index for index, v in enumerate(other_ids)}
        ans: List[Cut] = [None] * len(other_ids)
        for cut in self:
            ans[index_map[cut.id]] = cut
        return CutSet(ans)

    def index_supervisions(
        self, index_mixed_tracks: bool = False, keep_ids: Optional[Set[str]] = None
    ) -> Dict[str, IntervalTree]:
        """
        Create a two-level index of supervision segments. It is a mapping from a Cut's ID to an
        interval tree that contains the supervisions of that Cut.

        The interval tree can be efficiently queried for overlapping and/or enveloping segments.
        It helps speed up some operations on Cuts of very long recordings (1h+) that contain many
        supervisions.

        :param index_mixed_tracks: Should the tracks of MixedCut's be indexed as additional, separate entries.
        :param keep_ids: If specified, we will only index the supervisions with the specified IDs.
        :return: a mapping from Cut ID to an interval tree of SupervisionSegments.
        """
        indexed = {}
        for cut in self:
            indexed.update(
                cut.index_supervisions(
                    index_mixed_tracks=index_mixed_tracks, keep_ids=keep_ids
                )
            )
        return indexed

    def pad(
        self,
        duration: Seconds = None,
        num_frames: int = None,
        num_samples: int = None,
        pad_feat_value: float = LOG_EPSILON,
        direction: str = "right",
        preserve_id: bool = False,
        pad_value_dict: Optional[Dict[str, Union[int, float]]] = None,
    ) -> "CutSet":
        """
        Return a new CutSet with Cuts padded to ``duration``, ``num_frames`` or ``num_samples``.
        Cuts longer than the specified argument will not be affected.
        By default, cuts will be padded to the right (i.e. after the signal).

        When none of ``duration``, ``num_frames``, or ``num_samples`` is specified,
        we'll try to determine the best way to pad to the longest cut based on
        whether features or recordings are available.

        :param duration: The cuts minimal duration after padding.
            When not specified, we'll choose the duration of the longest cut in the CutSet.
        :param num_frames: The cut's total number of frames after padding.
        :param num_samples: The cut's total number of samples after padding.
        :param pad_feat_value: A float value that's used for padding the features.
            By default we assume a log-energy floor of approx. -23 (1e-10 after exp).
        :param direction: string, 'left', 'right' or 'both'. Determines whether the padding is added
            before or after the cut.
        :param preserve_id: When ``True``, preserves the cut ID from before padding.
            Otherwise, generates a new random ID (default).
        :param pad_value_dict: Optional dict that specifies what value should be used
            for padding arrays in custom attributes.
        :return: A padded CutSet.
        """
        # When the user does not specify explicit padding duration/num_frames/num_samples,
        # we'll try to pad using frames if there are features,
        # otherwise using samples if there are recordings,
        # otherwise duration which is always there.
        if all(arg is None for arg in (duration, num_frames, num_samples)):
            if all(c.has_features for c in self):
                num_frames = max(c.num_frames for c in self)
            elif all(c.has_recording for c in self):
                num_samples = max(c.num_samples for c in self)
            else:
                duration = max(cut.duration for cut in self)

        return self.map(
            partial(
                _pad,
                duration=duration,
                num_frames=num_frames,
                num_samples=num_samples,
                pad_feat_value=pad_feat_value,
                direction=direction,
                preserve_id=preserve_id,
                pad_value_dict=pad_value_dict,
            )
        )

    def truncate(
        self,
        max_duration: Seconds,
        offset_type: str,
        keep_excessive_supervisions: bool = True,
        preserve_id: bool = False,
        rng: Optional[random.Random] = None,
    ) -> "CutSet":
        """
        Return a new CutSet with the Cuts truncated so that their durations are at most `max_duration`.
        Cuts shorter than `max_duration` will not be changed.
        :param max_duration: float, the maximum duration in seconds of a cut in the resulting manifest.
        :param offset_type: str, can be:
        - 'start' => cuts are truncated from their start;
        - 'end' => cuts are truncated from their end minus max_duration;
        - 'random' => cuts are truncated randomly between their start and their end minus max_duration
        :param keep_excessive_supervisions: bool. When a cut is truncated in the middle of a supervision segment,
        should the supervision be kept.
        :param preserve_id: bool. Should the truncated cut keep the same ID or get a new, random one.
        :param rng: optional random number generator to be used with a 'random' ``offset_type``.
        :return: a new CutSet instance with truncated cuts.
        """
        assert offset_type in (
            "start",
            "end",
            "random",
        ), f"Unknown offset type: '{offset_type}'"
        return self.map(
            partial(
                _truncate_single,
                max_duration=max_duration,
                offset_type=offset_type,
                keep_excessive_supervisions=keep_excessive_supervisions,
                preserve_id=preserve_id,
                rng=rng,
            )
        )

    def extend_by(
        self,
        duration: Seconds,
        direction: str = "both",
        preserve_id: bool = False,
        pad_silence: bool = True,
    ) -> "CutSet":
        """
        Returns a new CutSet with cuts extended by `duration` amount.

        :param duration: float (seconds), specifies the duration by which the CutSet is extended.
        :param direction: string, 'left', 'right' or 'both'. Determines whether to extend on the left,
            right, or both sides. If 'both', extend on both sides by the same duration (equal to `duration`).
        :param preserve_id: bool. Should the extended cut keep the same ID or get a new, random one.
        :param pad_silence: bool. If True, the extended part of the cut will be padded with silence if required
            to match the specified duration.
        :return: a new CutSet instance.
        """
        return self.map(
            partial(
                _extend_by,
                duration=duration,
                direction=direction,
                preserve_id=preserve_id,
                pad_silence=pad_silence,
            )
        )

    def cut_into_windows(
        self,
        duration: Seconds,
        hop: Optional[Seconds] = None,
        keep_excessive_supervisions: bool = True,
        num_jobs: int = 1,
    ) -> "CutSet":
        """
        Return a new ``CutSet``, made by traversing each ``DataCut`` in windows of ``duration`` seconds by ``hop`` seconds and
        creating new ``DataCut`` out of them.

        The last window might have a shorter duration if there was not enough audio, so you might want to
        use either ``.filter()`` or ``.pad()`` afterwards to obtain a uniform duration ``CutSet``.

        :param duration: Desired duration of the new cuts in seconds.
        :param hop: Shift between the windows in the new cuts in seconds.
        :param keep_excessive_supervisions: bool. When a cut is truncated in the middle of a supervision segment,
            should the supervision be kept.
        :param num_jobs: The number of parallel workers.
        :return: a new CutSet with cuts made from shorter duration windows.
        """
        if not hop:
            hop = duration
        if num_jobs == 1:
            return CutSet(
                LazyFlattener(
                    LazyMapper(
                        self,
                        partial(
                            _cut_into_windows_single,
                            duration=duration,
                            hop=hop,
                            keep_excessive_supervisions=keep_excessive_supervisions,
                        ),
                    )
                )
            )

        from lhotse.manipulation import split_parallelize_combine

        result = split_parallelize_combine(
            num_jobs,
            self,
            _cut_into_windows_single,
            duration=duration,
            hop=hop,
            keep_excessive_supervisions=keep_excessive_supervisions,
        )
        return result

    def load_audio(
        self,
        collate: bool = False,
        limit: int = 1024,
    ) -> Union[List[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Reads the audio of all cuts in this :class:`.CutSet` into memory.
        Useful when this object represents a mini-batch.

        :param collate: Should we collate the read audio into a single array.
            Shorter cuts will be padded. False by default.
        :param limit: Maximum number of read audio examples.
            By default it's 1024 which covers most frequently encountered mini-batch sizes.
            If you are working with larger batch sizes, increase this limit.
        :return: A list of numpy arrays, or a single array with batch size as the first dim.
        """
        assert not self.is_lazy, "Cannot load audio of cuts in a lazy CutSet."
        assert len(self) < limit, (
            f"Cannot load audio of a CutSet with len={len(self)} because limit was set to {limit}. "
            f"This is a safe-guard against accidental CPU memory blow-ups. "
            f"If you know what you're doing, set the limit higher."
        )
        if collate:
            from lhotse.dataset.collation import collate_audio

            audios, audio_lens = collate_audio(self)
            return audios.numpy(), audio_lens.numpy()

        return [cut.load_audio() for cut in self]

    def sample(self, n_cuts: int = 1) -> Union[Cut, "CutSet"]:
        """
        Randomly sample this ``CutSet`` and return ``n_cuts`` cuts.
        When ``n_cuts`` is 1, will return a single cut instance; otherwise will return a ``CutSet``.
        """
        assert n_cuts > 0
        cut_indices = random.sample(range(len(self)), min(n_cuts, len(self)))
        cuts = [self[idx] for idx in cut_indices]
        if n_cuts == 1:
            return cuts[0]
        return CutSet(cuts)

    def resample(self, sampling_rate: int, affix_id: bool = False) -> "CutSet":
        """
        Return a new :class:`~lhotse.cut.CutSet` that contains cuts resampled to the new
        ``sampling_rate``. All cuts in the manifest must contain recording information.
        If the feature manifests are attached, they are dropped.

        :param sampling_rate: The new sampling rate.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the ``CutSet``.
        """
        return self.map(
            partial(_resample, sampling_rate=sampling_rate, affix_id=affix_id)
        )

    def perturb_speed(self, factor: float, affix_id: bool = True) -> "CutSet":
        """
        Return a new :class:`~lhotse.cut.CutSet` that contains speed perturbed cuts
        with a factor of ``factor``. It requires the recording manifests to be present.
        If the feature manifests are attached, they are dropped.
        The supervision manifests are modified to reflect the speed perturbed
        start times and durations.

        :param factor: The resulting playback speed is ``factor`` times the original one.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the ``CutSet``.
        """
        return self.map(partial(_perturb_speed, factor=factor, affix_id=affix_id))

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> "CutSet":
        """
        Return a new :class:`~lhotse.cut.CutSet` that contains tempo perturbed cuts
        with a factor of ``factor``.

        Compared to speed perturbation, tempo preserves pitch.
        It requires the recording manifests to be present.
        If the feature manifests are attached, they are dropped.
        The supervision manifests are modified to reflect the tempo perturbed
        start times and durations.

        :param factor: The resulting playback tempo is ``factor`` times the original one.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the ``CutSet``.
        """
        return self.map(partial(_perturb_tempo, factor=factor, affix_id=affix_id))

    def perturb_volume(self, factor: float, affix_id: bool = True) -> "CutSet":
        """
        Return a new :class:`~lhotse.cut.CutSet` that contains volume perturbed cuts
        with a factor of ``factor``. It requires the recording manifests to be present.
        If the feature manifests are attached, they are dropped.
        The supervision manifests are remaining the same.

        :param factor: The resulting playback volume is ``factor`` times the original one.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the ``CutSet``.
        """
        return self.map(partial(_perturb_volume, factor=factor, affix_id=affix_id))

    def narrowband(
        self, codec: str, restore_orig_sr: bool = True, affix_id: bool = True
    ) -> "CutSet":
        """
        Return a new :class:`~lhotse.cut.CutSet` that contains narrowband effect cuts.
        It requires the recording manifests to be present.
        If the feature manifests are attached, they are dropped.
        The supervision manifests are remaining the same.

        :param codec: Codec name.
        :param restore_orig_sr: Restore original sampling rate.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the ``CutSet``.
        """
        return self.map(
            lambda cut: cut.narrowband(
                codec=codec, restore_orig_sr=restore_orig_sr, affix_id=affix_id
            )
        )

    def normalize_loudness(
        self, target: float, mix_first: bool = True, affix_id: bool = True
    ) -> "CutSet":
        """
        Return a new :class:`~lhotse.cut.CutSet` that will lazily apply loudness normalization
        to the desired ``target`` loudness (in dBFS).

        :param target: The target loudness in dBFS.
        :param affix_id: When true, we will modify the ``Cut.id`` field
            by affixing it with "_ln{target}".
        :return: a modified copy of the current ``CutSet``.
        """
        return self.map(
            partial(
                _normalize_loudness,
                target=target,
                mix_first=mix_first,
                affix_id=affix_id,
            )
        )

    def dereverb_wpe(self, affix_id: bool = True) -> "CutSet":
        """
        Return a new :class:`~lhotse.cut.CutSet` that will lazily apply WPE dereverberation.

        :param affix_id: When true, we will modify the ``Cut.id`` field
            by affixing it with "_wpe".
        :return: a modified copy of the current ``CutSet``.
        """
        return self.map(partial(_dereverb_wpe, affix_id=affix_id))

    def reverb_rir(
        self,
        rir_recordings: Optional["RecordingSet"] = None,
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: List[int] = [0],
    ) -> "CutSet":
        """
        Return a new :class:`~lhotse.cut.CutSet` that contains original cuts convolved with
        randomly chosen impulse responses from `rir_recordings`. It requires the recording manifests to be present.
        If the feature manifests are attached, they are dropped.
        The supervision manifests remain the same.

        If no ``rir_recordings`` are provided, we will generate a set of impulse responses using a fast random
        generator (https://arxiv.org/abs/2208.04101).

        :param rir_recordings: RecordingSet containing the room impulse responses.
        :param normalize_output: When true, output will be normalized to have energy as input.
        :param early_only: When true, only the early reflections (first 50 ms) will be used.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :param rir_channels: The channels of the impulse response to use. By default, first channel will be used.
            If it is a multi-channel RIR, applying RIR will produce MixedCut. If no RIR is
            provided, we will generate one with as many channels as this argument specifies.
        :return: a modified copy of the ``CutSet``.
        """
        rir_recordings = list(rir_recordings) if rir_recordings else None
        return self.map(
            partial(
                _reverb_rir,
                rir_recording=random.choice(rir_recordings) if rir_recordings else None,
                normalize_output=normalize_output,
                early_only=early_only,
                affix_id=affix_id,
                rir_channels=rir_channels,
            )
        )

    def mix(
        self,
        cuts: "CutSet",
        duration: Optional[Seconds] = None,
        allow_padding: bool = False,
        snr: Optional[Union[Decibels, Sequence[Decibels]]] = 20,
        preserve_id: Optional[str] = None,
        mix_prob: float = 1.0,
        seed: Union[int, Literal["trng", "randomized"], random.Random] = 42,
        random_mix_offset: bool = False,
    ) -> "CutSet":
        """
        Mix cuts in this ``CutSet`` with randomly sampled cuts from another ``CutSet``.
        A typical application would be data augmentation with noise, music, babble, etc.

        :param cuts: a ``CutSet`` containing cuts to be mixed into this ``CutSet``.
        :param duration: an optional float in seconds.
            When ``None``, we will preserve the duration of the cuts in ``self``
            (i.e. we'll truncate the mix if it exceeded the original duration).
            Otherwise, we will keep sampling cuts to mix in until we reach the specified
            ``duration`` (and truncate to that value, should it be exceeded).
        :param allow_padding: an optional bool.
            When it is ``True``, we will allow the offset to be larger than the reference
            cut by padding the reference cut.
        :param snr: an optional float, or pair (range) of floats, in decibels.
            When it's a single float, we will mix all cuts with this SNR level
            (where cuts in ``self`` are treated as signals, and cuts in ``cuts`` are treated as noise).
            When it's a pair of floats, we will uniformly sample SNR values from that range.
            When ``None``, we will mix the cuts without any level adjustment
            (could be too noisy for data augmentation).
        :param preserve_id: optional string ("left", "right"). when specified, append will preserve the cut id
            of the left- or right-hand side argument. otherwise, a new random id is generated.
        :param mix_prob: an optional float in range [0, 1].
            Specifies the probability of performing a mix.
            Values lower than 1.0 mean that some cuts in the output will be unchanged.
        :param seed: an optional int or "trng". Random seed for choosing the cuts to mix and the SNR.
            If "trng" is provided, we'll use the ``secrets`` module for non-deterministic results
            on each iteration. You can also directly pass a ``random.Random`` instance here.
        :param random_mix_offset: an optional bool.
            When ``True`` and the duration of the to be mixed in cut in longer than the original cut,
             select a random sub-region from the to be mixed in cut.
        :return: a new ``CutSet`` with mixed cuts.
        """
        return CutSet(
            LazyCutMixer(
                cuts=self,
                mix_in_cuts=cuts,
                duration=duration,
                allow_padding=allow_padding,
                snr=snr,
                preserve_id=preserve_id,
                mix_prob=mix_prob,
                seed=seed,
                random_mix_offset=random_mix_offset,
            )
        )

    def drop_features(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its extracted features.
        """
        return self.map(_drop_features)

    def drop_recordings(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its recordings.
        """
        return self.map(_drop_recordings)

    def drop_supervisions(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its supervisions.
        """
        return self.map(_drop_supervisions)

    def drop_alignments(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from the alignments present in its supervisions.
        """
        return self.map(_drop_alignments)

    def drop_in_memory_data(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from any in-memory data it held.
        The manifests for in-memory data are converted into placeholders that can still be looked up for
        metadata, but will fail on attempts to load the data.
        """
        return self.map(_drop_in_memory_data)

    def compute_and_store_features(
        self,
        extractor: FeatureExtractor,
        storage_path: Pathlike,
        num_jobs: Optional[int] = None,
        augment_fn: Optional[AugmentFn] = None,
        storage_type: Type[FW] = LilcomChunkyWriter,
        executor: Optional[Executor] = None,
        mix_eagerly: bool = True,
        progress_bar: bool = True,
    ) -> "CutSet":
        """
        Extract features for all cuts, possibly in parallel,
        and store them using the specified storage object.

        Examples:

            Extract fbank features on one machine using 8 processes,
            store arrays partitioned in 8 archive files with lilcom compression:

            >>> cuts = CutSet(...)
            ... cuts.compute_and_store_features(
            ...     extractor=Fbank(),
            ...     storage_path='feats',
            ...     num_jobs=8,
            ... )

            Extract fbank features on one machine using 8 processes,
            store each array in a separate file with lilcom compression:

            >>> cuts = CutSet(...)
            ... cuts.compute_and_store_features(
            ...     extractor=Fbank(),
            ...     storage_path='feats',
            ...     num_jobs=8,
            ...     storage_type=LilcomFilesWriter
            ... )

            Extract fbank features on multiple machines using a Dask cluster
            with 80 jobs,
            store arrays partitioned in 80 archive files with lilcom compression:

            >>> from distributed import Client
            ... cuts = CutSet(...)
            ... cuts.compute_and_store_features(
            ...     extractor=Fbank(),
            ...     storage_path='feats',
            ...     num_jobs=80,
            ...     executor=Client(...)
            ... )

            Extract fbank features on one machine using 8 processes,
            store each array in an S3 bucket (requires ``smart_open``):

            >>> cuts = CutSet(...)
            ... cuts.compute_and_store_features(
            ...     extractor=Fbank(),
            ...     storage_path='s3://my-feature-bucket/my-corpus-features',
            ...     num_jobs=8,
            ...     storage_type=LilcomURLWriter
            ... )

        :param extractor: A ``FeatureExtractor`` instance
            (either Lhotse's built-in or a custom implementation).
        :param storage_path: The path to location where we will store the features.
            The exact type and layout of stored files will be dictated by the
            ``storage_type`` argument.
        :param num_jobs: The number of parallel processes used to extract the features.
            We will internally split the CutSet into this many chunks
            and process each chunk in parallel.
        :param augment_fn: an optional callable used for audio augmentation.
            Be careful with the types of augmentations used: if they modify
            the start/end/duration times of the cut and its supervisions,
            you will end up with incorrect supervision information when using this API.
            E.g. for speed perturbation, use ``CutSet.perturb_speed()`` instead.
        :param storage_type: a ``FeaturesWriter`` subclass type.
            It determines how the features are stored to disk,
            e.g. separate file per array, HDF5 files with multiple arrays, etc.
        :param executor: when provided, will be used to parallelize the feature extraction process.
            By default, we will instantiate a ProcessPoolExecutor.
            Learn more about the ``Executor`` API at
            https://lhotse.readthedocs.io/en/latest/parallelism.html
        :param mix_eagerly: Related to how the features are extracted for ``MixedCut``
            instances, if any are present.
            When False, extract and store the features for each track separately,
            and mix them dynamically when loading the features.
            When True, mix the audio first and store the mixed features,
            returning a new ``DataCut`` instance with the same ID.
            The returned ``DataCut`` will not have a ``Recording`` attached.
        :param progress_bar: Should a progress bar be displayed (automatically turned off
            for parallel computation).
        :return: Returns a new ``CutSet`` with ``Features`` manifests attached to the cuts.
        """
        from lhotse.manipulation import combine

        # Pre-conditions and args setup
        progress = (
            lambda x: x
        )  # does nothing, unless we overwrite it with an actual prog bar
        if num_jobs is None:
            num_jobs = 1
        if num_jobs == 1 and executor is not None:
            logging.warning(
                "Executor argument was passed but num_jobs set to 1: "
                "we will ignore the executor and use non-parallel execution."
            )
            executor = None

        if num_jobs > 1 and torch.get_num_threads() > 1:
            logging.warning(
                "num_jobs is > 1 and torch's number of threads is > 1 as well: "
                "For certain configs this can result in a never ending computation. "
                "If this happens, use torch.set_num_threads(1) to circumvent this."
            )

        # Non-parallel execution
        if executor is None and num_jobs == 1:
            if progress_bar:
                progress = partial(
                    tqdm, desc="Extracting and storing features", total=len(self)
                )

            with storage_type(storage_path) as storage:
                return CutSet.from_cuts(
                    maybe_cut
                    for maybe_cut in progress(
                        null_result_on_audio_loading_error(
                            cut.compute_and_store_features
                        )(
                            extractor=extractor,
                            storage=storage,
                            augment_fn=augment_fn,
                            mix_eagerly=mix_eagerly,
                        )
                        for cut in self
                    )
                    if maybe_cut is not None
                )

        # HACK: support URL storage for writing
        if "://" in str(storage_path):
            # "storage_path" is actually an URL
            def sub_storage_path(idx: int) -> str:
                return f"{storage_path}/feats-{idx}"

        else:
            # We are now sure that "storage_path" will be the root for
            # multiple feature storages - we can create it as a directory
            storage_path = Path(storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)

            def sub_storage_path(idx: int) -> str:
                return storage_path / f"feats-{idx}"

        # Parallel execution: prepare the CutSet splits
        # We use LazySlicer to pick every k element out of n
        # (e.g. with 2 jobs, job 1 picks every 0th elem, job 2 picks every 1st elem)
        cut_sets = [CutSet(LazySlicer(self, k=i, n=num_jobs)) for i in range(num_jobs)]

        # Initialize the default executor if None was given
        if executor is None:
            import multiprocessing

            executor = ProcessPoolExecutor(
                num_jobs, mp_context=multiprocessing.get_context("spawn")
            )

        # Submit the chunked tasks to parallel workers.
        # Each worker runs the non-parallel version of this function inside.
        futures = [
            executor.submit(
                CutSet.compute_and_store_features,
                cs,
                extractor=extractor,
                storage_path=sub_storage_path(i),
                augment_fn=augment_fn,
                storage_type=storage_type,
                mix_eagerly=mix_eagerly,
                # Disable individual workers progress bars for readability
                progress_bar=False,
            )
            for i, cs in enumerate(cut_sets)
        ]

        if progress_bar:
            progress = partial(
                tqdm,
                desc="Extracting and storing features (chunks progress)",
                total=len(futures),
            )

        cuts_with_feats = combine(progress(f.result() for f in futures))
        return cuts_with_feats

    def compute_and_store_features_batch(
        self,
        extractor: FeatureExtractor,
        storage_path: Pathlike,
        manifest_path: Optional[Pathlike] = None,
        batch_duration: Seconds = 600.0,
        num_workers: int = 4,
        collate: bool = False,
        augment_fn: Optional[AugmentFn] = None,
        storage_type: Type[FW] = LilcomChunkyWriter,
        overwrite: bool = False,
    ) -> "CutSet":
        """
        Extract features for all cuts in batches.
        This method is intended for use with compatible feature extractors that
        implement an accelerated :meth:`~lhotse.FeatureExtractor.extract_batch` method.
        For example, ``kaldifeat`` extractors can be used this way (see, e.g.,
        :class:`~lhotse.KaldifeatFbank` or :class:`~lhotse.KaldifeatMfcc`).

        When a CUDA GPU is available and enabled for the feature extractor, this can
        be much faster than :meth:`.CutSet.compute_and_store_features`.
        Otherwise, the speed will be comparable to single-threaded extraction.

        Example: extract fbank features on one GPU, using 4 dataloading workers
        for reading audio, and store the arrays in an archive file with
        lilcom compression::

            >>> from lhotse import KaldifeatFbank, KaldifeatFbankConfig
            >>> extractor = KaldifeatFbank(KaldifeatFbankConfig(device='cuda'))
            >>> cuts = CutSet(...)
            ... cuts = cuts.compute_and_store_features_batch(
            ...     extractor=extractor,
            ...     storage_path='feats',
            ...     batch_duration=500,
            ...     num_workers=4,
            ... )

        :param extractor: A :class:`~lhotse.features.base.FeatureExtractor` instance,
            which should implement an accelerated ``extract_batch`` method.
        :param storage_path: The path to location where we will store the features.
            The exact type and layout of stored files will be dictated by the
            ``storage_type`` argument.
        :param manifest_path: Optional path where to write the CutSet manifest
            with attached feature manifests. If not specified, we will be keeping
            all manifests in memory.
        :param batch_duration: The maximum number of audio seconds in a batch.
            Determines batch size dynamically.
        :param num_workers: How many background dataloading workers should be used
            for reading the audio.
        :param collate: If ``True``, the waveforms will be collated into a single
            padded tensor before being passed to the feature extractor. Some extractors
            can be faster this way (for e.g., see ``lhotse.features.kaldi.extractors``).
            If you are using ``kaldifeat`` extractors, you should set this to ``False``.
        :param augment_fn: an optional callable used for audio augmentation.
            Be careful with the types of augmentations used: if they modify
            the start/end/duration times of the cut and its supervisions,
            you will end up with incorrect supervision information when using this API.
            E.g. for speed perturbation, use ``CutSet.perturb_speed()`` instead.
        :param storage_type: a ``FeaturesWriter`` subclass type.
            It determines how the features are stored to disk,
            e.g. separate file per array, HDF5 files with multiple arrays, etc.
        :param overwrite: should we overwrite the manifest, HDF5 files, etc.
            By default, this method will append to these files if they exist.
        :return: Returns a new ``CutSet`` with ``Features`` manifests attached to the cuts.
        """
        from concurrent.futures import ThreadPoolExecutor

        from torch.utils.data import DataLoader

        from lhotse.dataset import SimpleCutSampler, UnsupervisedWaveformDataset
        from lhotse.qa import validate_features

        frame_shift = extractor.frame_shift

        # We're opening a sequential cuts writer that can resume previously interrupted
        # operation. It scans the input JSONL file for cut IDs that should be ignored.
        # Note: this only works when ``manifest_path`` argument was specified, otherwise
        # we hold everything in memory and start from scratch.
        cuts_writer = CutSet.open_writer(manifest_path, overwrite=overwrite)

        # We tell the sampler to ignore cuts that were already processed.
        # It will avoid I/O for reading them in the DataLoader.
        sampler = SimpleCutSampler(self, max_duration=batch_duration)
        sampler.filter(lambda cut: cut.id not in cuts_writer.ignore_ids)
        dataset = UnsupervisedWaveformDataset(collate=collate)
        dloader = DataLoader(
            dataset, batch_size=None, sampler=sampler, num_workers=num_workers
        )

        # Background worker to save features to disk.
        def _save_worker(cuts: List[Cut], features: List[np.ndarray]) -> None:
            for cut, feat_mat in zip(cuts, features):
                if isinstance(cut, PaddingCut):
                    # For padding cuts, just fill out the fields in the manifest
                    # and don't store anything.
                    cuts_writer.write(
                        fastcopy(
                            cut,
                            num_frames=feat_mat.shape[0],
                            num_features=feat_mat.shape[1],
                            frame_shift=frame_shift,
                        )
                    )
                    continue
                # Store the computed features and describe them in a manifest.
                if isinstance(feat_mat, torch.Tensor):
                    feat_mat = feat_mat.cpu().numpy()
                storage_key = feats_writer.write(cut.id, feat_mat)
                feat_manifest = Features(
                    start=cut.start,
                    duration=cut.duration,
                    type=extractor.name,
                    num_frames=feat_mat.shape[0],
                    num_features=feat_mat.shape[1],
                    frame_shift=frame_shift,
                    sampling_rate=cut.sampling_rate,
                    channels=cut.channel,
                    storage_type=feats_writer.name,
                    storage_path=str(feats_writer.storage_path),
                    storage_key=storage_key,
                )
                validate_features(feat_manifest, feats_data=feat_mat)

                # Update the cut manifest.
                if isinstance(cut, DataCut):
                    feat_manifest.recording_id = cut.recording_id
                    cut = fastcopy(cut, features=feat_manifest)
                if isinstance(cut, MixedCut):
                    # If this was a mixed cut, we will just discard its
                    # recordings and create a new mono cut that has just
                    # the features attached. We will also set its `channel`
                    # to 0, since we are creating a mono cut.
                    feat_manifest.recording_id = cut.id
                    cut = MonoCut(
                        id=cut.id,
                        start=0,
                        duration=cut.duration,
                        channel=0,
                        # Update supervisions recording_id for consistency
                        supervisions=[
                            fastcopy(s, recording_id=cut.id, channel=0)
                            for s in cut.supervisions
                        ],
                        features=feat_manifest,
                        recording=None,
                    )
                cuts_writer.write(cut, flush=True)

        futures = []
        with cuts_writer, storage_type(
            storage_path, mode="w" if overwrite else "a"
        ) as feats_writer, tqdm(
            desc="Computing features in batches", total=sampler.num_cuts
        ) as progress, ThreadPoolExecutor(
            max_workers=1  # We only want one background worker so that serialization is deterministic.
        ) as executor:
            # Display progress bar correctly.
            progress.update(len(cuts_writer.ignore_ids))
            for batch in dloader:
                cuts = batch["cuts"]
                waves = batch["audio"]
                wave_lens = batch["audio_lens"] if collate else None

                if len(cuts) == 0:
                    # Fault-tolerant audio loading filtered out everything.
                    continue

                assert all(c.sampling_rate == cuts[0].sampling_rate for c in cuts)

                # Optionally apply the augment_fn
                if augment_fn is not None:
                    waves = [
                        augment_fn(w, c.sampling_rate) for c, w in zip(cuts, waves)
                    ]

                # The actual extraction is here.
                with torch.no_grad():
                    # Note: chunk_size option limits the memory consumption
                    # for very long cuts.
                    features = extractor.extract_batch(
                        waves, sampling_rate=cuts[0].sampling_rate, lengths=wave_lens
                    )

                futures.append(executor.submit(_save_worker, cuts, features))
                progress.update(len(cuts))

        # If ``manifest_path`` was provided, this is a lazy manifest;
        # otherwise everything is in memory.
        return cuts_writer.open_manifest()

    def save_audios(
        self,
        storage_path: Pathlike,
        format: str = "wav",
        encoding: Optional[str] = None,
        num_jobs: Optional[int] = None,
        executor: Optional[Executor] = None,
        augment_fn: Optional[AugmentFn] = None,
        progress_bar: bool = True,
        shuffle_on_split: bool = True,
        **kwargs,
    ) -> "CutSet":
        """
        Store waveforms of all cuts as audio recordings to disk.

        :param storage_path: The path to location where we will store the audio recordings.
            For each cut, a sub-directory will be created that starts with the first 3
            characters of the cut's ID. The audio recording is then stored in the sub-directory
            using filename ``{cut.id}.{format}``
        :param format: Audio format argument supported by ``torchaudio.save`` or ``soundfile.write``.
            Tested values are: ``wav``, ``flac``, and ``opus``.
        :param encoding: Audio encoding argument supported by ``torchaudio.save`` or ``soundfile.write``.
            Please refer to the documentation of the relevant library used in your audio backend.
        :param num_jobs: The number of parallel processes used to store the audio recordings.
            We will internally split the CutSet into this many chunks
            and process each chunk in parallel.
        :param augment_fn: an optional callable used for audio augmentation.
            Be careful with the types of augmentations used: if they modify
            the start/end/duration times of the cut and its supervisions,
            you will end up with incorrect supervision information when using this API.
            E.g. for speed perturbation, use ``CutSet.perturb_speed()`` instead.
        :param executor: when provided, will be used to parallelize the process.
            By default, we will instantiate a ProcessPoolExecutor.
            Learn more about the ``Executor`` API at
            https://lhotse.readthedocs.io/en/latest/parallelism.html
        :param progress_bar: Should a progress bar be displayed (automatically turned off
            for parallel computation).
        :param shuffle_on_split: Shuffle the ``CutSet`` before splitting it for the parallel workers.
            It is active only when `num_jobs > 1`. The default is True.
        :param kwargs: Deprecated arguments go here and are ignored.
        :return: Returns a new ``CutSet``.
        """
        from cytoolz import identity

        from lhotse.manipulation import combine

        # Pre-conditions and args setup
        progress = (
            identity  # does nothing, unless we overwrite it with an actual prog bar
        )
        if num_jobs is None:
            num_jobs = 1
        if num_jobs == 1 and executor is not None:
            logging.warning(
                "Executor argument was passed but num_jobs set to 1: "
                "we will ignore the executor and use non-parallel execution."
            )
            executor = None

        def file_storage_path(cut: Cut, storage_path: Pathlike) -> Path:
            # Introduce a sub-directory that starts with the first 3 characters of the cut's ID.
            # This allows to avoid filesystem performance problems related to storing
            # too many files in a single directory.
            subdir = Path(storage_path) / cut.id[:3]
            subdir.mkdir(exist_ok=True, parents=True)
            return subdir / (cut.id + "." + format)

        # Non-parallel execution
        if executor is None and num_jobs == 1:
            if progress_bar:
                progress = partial(
                    tqdm, desc="Storing audio recordings", total=len(self)
                )
            return CutSet(
                progress(
                    cut.save_audio(
                        storage_path=file_storage_path(cut, storage_path),
                        format=format,
                        encoding=encoding,
                        augment_fn=augment_fn,
                    )
                    for cut in self
                )
            ).to_eager()

        # Parallel execution: prepare the CutSet splits
        cut_sets = self.split(num_jobs, shuffle=shuffle_on_split)

        # Initialize the default executor if None was given
        if executor is None:
            import multiprocessing

            # The `is_caching_enabled()` state gets transfered to
            # the spawned sub-processes implictly (checked).
            executor = ProcessPoolExecutor(
                max_workers=num_jobs,
                mp_context=multiprocessing.get_context("spawn"),
            )

        # Submit the chunked tasks to parallel workers.
        # Each worker runs the non-parallel version of this function inside.
        futures = [
            executor.submit(
                CutSet.save_audios,
                cs,
                storage_path=storage_path,
                format=format,
                encoding=encoding,
                augment_fn=augment_fn,
                # Disable individual workers progress bars for readability
                progress_bar=False,
            )
            for i, cs in enumerate(cut_sets)
        ]

        if progress_bar:
            progress = partial(
                tqdm,
                desc="Storing audio recordings (chunks progress)",
                total=len(futures),
            )

        cuts = combine(progress(f.result() for f in futures))
        return cuts

    def compute_global_feature_stats(
        self,
        storage_path: Optional[Pathlike] = None,
        max_cuts: Optional[int] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute the global means and standard deviations for each feature bin in the manifest.
        It follows the implementation in scikit-learn:
        https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/utils/extmath.py#L715
        which follows the paper:
        "Algorithms for computing the sample variance: analysis and recommendations", by Chan, Golub, and LeVeque.

        :param storage_path: an optional path to a file where the stats will be stored with pickle.
        :param max_cuts: optionally, limit the number of cuts used for stats estimation. The cuts will be
            selected randomly in that case.
        :param extractor: optional FeatureExtractor, when provided, we ignore any pre-computed features.

        :return a dict of ``{'norm_means': np.ndarray, 'norm_stds': np.ndarray}`` with the
            shape of the arrays equal to the number of feature bins in this manifest.
        """
        if extractor is not None:
            cuts = self
            if max_cuts is not None:
                cuts = islice(cuts, max_cuts)
            cuts = iter(cuts)
            first = next(cuts)
            stats = StatsAccumulator(
                feature_dim=extractor.feature_dim(first.sampling_rate)
            )
            for cut in chain([first], cuts):
                arr = cut.compute_features(extractor)
                stats.update(arr)
            mvn = stats.get()
            if storage_path is not None:
                with open(storage_path, "wb") as f:
                    pickle.dump(mvn, f)
            return mvn

        have_features = [cut.has_features for cut in self]
        if not any(have_features):
            raise ValueError(
                "Could not find any features in this CutSet; did you forget to extract them?"
            )
        if not all(have_features):
            logging.warning(
                f"Computing global stats: only {sum(have_features)}/{len(have_features)} cuts have features."
            )
        return compute_global_stats(
            # islice(X, 50) is like X[:50] except it works with lazy iterables
            feature_manifests=islice(
                (cut.features for cut in self if cut.has_features),
                max_cuts if max_cuts is not None else len(self),
            ),
            storage_path=storage_path,
        )

    def with_features_path_prefix(self, path: Pathlike) -> "CutSet":
        return self.map(partial(_add_features_path_prefix_single, path=path))

    def with_recording_path_prefix(self, path: Pathlike) -> "CutSet":
        return self.map(partial(_add_recording_path_prefix_single, path=path))

    def copy_data(self, output_dir: Pathlike, verbose: bool = True) -> "CutSet":
        """
        Copies every data item referenced by this CutSet into a new directory.
        The structure is as follows:

        - output_dir
        ├── audio
        |   ├── rec1.flac
        |   └── ...
        ├── custom
        |   ├── field1
        |   |   ├── arr1-1.npy
        |   |   └── ...
        |   └── field2
        |       ├── arr2-1.npy
        |       └── ...
        ├── features.lca
        └── cuts.jsonl.gz

        :param output_dir: The root directory where we'll store the copied data.
        :param verbose: Show progress bar, enabled by default.
        :return: CutSet manifest pointing to the new data.
        """
        from lhotse.array import Array, TemporalArray
        from lhotse.features.io import NumpyHdf5Writer

        output_dir = Path(output_dir)
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(exist_ok=True, parents=True)
        feature_file = output_dir / "features.lca"
        custom_dir = output_dir / "custom"
        custom_dir.mkdir(exist_ok=True, parents=True)

        custom_writers = {}

        progbar = partial(tqdm, desc="Copying CutSet data") if verbose else lambda x: x

        with CutSet.open_writer(
            output_dir / "cuts.jsonl.gz"
        ) as manifest_writer, LilcomChunkyWriter(feature_file) as feature_writer:

            def _copy_single(cut):
                cut = fastcopy(cut)
                if cut.has_features:
                    cut.features = cut.features.copy_feats(writer=feature_writer)
                if cut.has_recording:
                    cut = cut.save_audio(
                        (audio_dir / cut.recording_id).with_suffix(".flac")
                    )
                if cut.custom is not None:
                    for k, v in cut.custom.items():
                        if isinstance(v, (Array, TemporalArray)):
                            if k not in custom_writers:
                                p = custom_dir / k
                                p.mkdir(exist_ok=True, parents=True)
                                custom_writers[k] = NumpyHdf5Writer(p)
                            cust_writer = custom_writers[k]
                            cust_writer.write(cut.id, v.load())
                return cut

            for item in progbar(self):
                if isinstance(item, PaddingCut):
                    manifest_writer.write(item)
                    continue

                if isinstance(item, MixedCut):
                    cpy = fastcopy(item)
                    for t in cpy.tracks:
                        if isinstance(t.cut, DataCut):
                            _copy_single(t.cut)
                    manifest_writer.write(cpy)

                elif isinstance(item, DataCut):
                    cpy = _copy_single(item)
                    manifest_writer.write(cpy)

                else:
                    raise RuntimeError(f"Unexpected manifest type: {type(item)}")

        for w in custom_writers.values():
            w.close()

        return manifest_writer.open_manifest()

    def copy_feats(
        self, writer: FeaturesWriter, output_path: Optional[Pathlike] = None
    ) -> "CutSet":
        """
        Save a copy of every feature matrix found in this CutSet using ``writer``
        and return a new manifest with cuts referring to the new feature locations.

        :param writer: a :class:`lhotse.features.io.FeaturesWriter` instance.
        :param output_path: optional path where the new manifest should be stored.
            It's used to write the manifest incrementally and return a lazy manifest,
            otherwise the copy is stored in memory.
        :return: a copy of the manifest.
        """
        with CutSet.open_writer(output_path) as manifest_writer:
            for item in self:
                if not item.has_features or isinstance(item, PaddingCut):
                    manifest_writer.write(item)
                    continue

                if isinstance(item, MixedCut):
                    cpy = fastcopy(item)
                    for t in cpy.tracks:
                        if isinstance(t.cut, DataCut):
                            t.cut.features = t.cut.features.copy_feats(writer=writer)
                    manifest_writer.write(cpy)

                elif isinstance(item, DataCut):
                    cpy = fastcopy(item)
                    cpy.features = cpy.features.copy_feats(writer=writer)
                    manifest_writer.write(cpy)

                else:
                    manifest_writer.write(item)

        return manifest_writer.open_manifest()

    def modify_ids(self, transform_fn: Callable[[str], str]) -> "CutSet":
        """
        Modify the IDs of cuts in this ``CutSet``.
        Useful when combining multiple ``CutSet``s that were created from a single source,
        but contain features with different data augmentations techniques.

        :param transform_fn: A callable (function) that accepts a string (cut ID) and returns
        a new string (new cut ID).
        :return: a new ``CutSet`` with cuts with modified IDs.
        """
        return self.map(partial(_with_id, transform_fn=transform_fn))

    def fill_supervisions(
        self, add_empty: bool = True, shrink_ok: bool = False
    ) -> "CutSet":
        """
        Fills the whole duration of each cut in a :class:`.CutSet` with a supervision segment.

        If the cut has one supervision, its start is set to 0 and duration is set to ``cut.duration``.
        Note: this may either expand a supervision that was shorter than a cut, or shrink a supervision
        that exceeds the cut.

        If there are no supervisions, we will add an empty one when ``add_empty==True``, otherwise
        we won't change anything.

        If there are two or more supervisions, we will raise an exception.

        :param add_empty: should we add an empty supervision with identical time bounds as the cut.
        :param shrink_ok: should we raise an error if a supervision would be shrank as a result
            of calling this method.
        """
        return self.map(
            partial(_fill_supervision, add_empty=add_empty, shrink_ok=shrink_ok)
        )

    def map_supervisions(
        self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]
    ) -> "CutSet":
        """
        Modify the SupervisionSegments by `transform_fn` in this CutSet.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a new, modified CutSet.
        """
        return self.map(partial(_map_supervisions, transform_fn=transform_fn))

    def transform_text(self, transform_fn: Callable[[str], str]) -> "CutSet":
        """
        Return a copy of this ``CutSet`` with all ``SupervisionSegments`` text transformed with ``transform_fn``.
        Useful for text normalization, phonetic transcription, etc.

        :param transform_fn: a function that accepts a string and returns a string.
        :return: a new, modified CutSet.
        """
        return self.map_supervisions(
            partial(_transform_text, transform_fn=transform_fn)
        )

    def prefetch(self, buffer_size: int = 10) -> "CutSet":
        """
        Pre-fetches the CutSet elements in a background process.
        Useful for enabling concurrent reading/processing/writing in ETL-style tasks.

        .. caution:: This method internally uses a PyTorch DataLoader with a single worker.
            It is not suitable for use in typical PyTorch training scripts.

        .. caution:: If you run into pickling issues when using this method, you're also likely
            using .filter/.map methods with a lambda function.
            Please set ``lhotse.set_dill_enabled(True)`` to resolve these issues, or convert lambdas
            to regular functions + ``functools.partial``

        """
        from torch.utils.data import DataLoader

        from lhotse.dataset import DynamicCutSampler, IterableDatasetWrapper

        return CutSet(
            DataLoader(
                dataset=IterableDatasetWrapper(
                    _BackgroundCutFetcher(),
                    DynamicCutSampler(self, max_cuts=1, rank=0, world_size=1),
                ),
                batch_size=None,
                num_workers=1,
                prefetch_factor=buffer_size,
            )
        )

    def to_huggingface_dataset(self):
        """
        Converts a CutSet to a HuggingFace Dataset. Currently, only MonoCut with one recording source is supported.
        Other cut types will be supported in the future.

        Currently, two formats are supported:
            1. If each cut has one supervision (e.g. LibriSpeech), each cut is represented as a single row (entry)
               in the HuggingFace dataset with all the supervision information stored along the cut information.
               The final HuggingFace dataset format is:
                   ╔═══════════════════╦═══════════════════════════════╗
                   ║      Feature      ║            Type               ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║        id         ║ Value(dtype='string')         ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║      audio        ║ Audio()                       ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║     duration      ║ Value(dtype='float32')        ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║   num_channels    ║ Value(dtype='uint16')         ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║       text        ║ Value(dtype='string')         ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║     speaker       ║ Value(dtype='string')         ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║     language      ║ Value(dtype='string')         ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║   {x}_alignment   ║ Sequence(Alignment)           ║
                   ╚═══════════════════╩═══════════════════════════════╝
               where x stands for the alignment type (commonly used: "word", "phoneme").

               Alignment is represented as:
                   ╔═══════════════════╦═══════════════════════════════╗
                   ║      Feature      ║            Type               ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║      symbol       ║ Value(dtype='string')         ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║       start       ║ Value(dtype='float32')        ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║        end        ║ Value(dtype='float32')        ║
                   ╚═══════════════════╩═══════════════════════════════╝


            2. If each cut has multiple supervisions (e.g. AMI), each cut is represented as a single row (entry)
               while all the supervisions are stored in a separate list of dictionaries under the 'segments' key.
               The final HuggingFace dataset format is:
                   ╔══════════════╦════════════════════════════════════╗
                   ║   Feature    ║                 Type               ║
                   ╠══════════════╬════════════════════════════════════╣
                   ║      id      ║ Value(dtype='string')              ║
                   ╠══════════════╬════════════════════════════════════╣
                   ║    audio     ║ Audio()                            ║
                   ╠══════════════╬════════════════════════════════════╣
                   ║   duration   ║ Value(dtype='float32')             ║
                   ╠══════════════╬════════════════════════════════════╣
                   ║ num_channels ║ Value(dtype='uint16')              ║
                   ╠══════════════╬════════════════════════════════════╣
                   ║   segments   ║ Sequence(Segment)                  ║
                   ╚══════════════╩════════════════════════════════════╝
               where one Segment is represented as:
                   ╔═══════════════════╦═══════════════════════════════╗
                   ║      Feature      ║            Type               ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║        text       ║ Value(dtype='string')         ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║       start       ║ Value(dtype='float32')        ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║        end        ║ Value(dtype='float32')        ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║      channel      ║ Value(dtype='string')         ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║      speaker      ║ Value(dtype='string')         ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║      language     ║ Value(dtype='string')         ║
                   ╠═══════════════════╬═══════════════════════════════╣
                   ║   {x}_alignment   ║ Sequence(Alignment)           ║
                   ╚═══════════════════╩═══════════════════════════════╝
        :return: A HuggingFace Dataset.
        """
        from lhotse.hf import export_cuts_to_hf

        return export_cuts_to_hf(self)

    @staticmethod
    def from_huggingface_dataset(
        *dataset_args,
        audio_key: str = "audio",
        text_key: str = "sentence",
        lang_key: str = "language",
        gender_key: str = "gender",
        **dataset_kwargs,
    ):
        """
        Initializes a Lhotse CutSet from an existing HF dataset,
        or args/kwargs passed on to ``datasets.load_dataset()``.

        Use ``audio_key``, ``text_key``, ``lang_key`` and ``gender_key`` options to indicate which keys in dict examples
        returned from HF Dataset should be looked up for audio, transcript, language, and gender respectively.
        The remaining keys in HF dataset examples will be stored inside ``cut.custom`` dictionary.

        Example with existing HF dataset::

            >>> import datasets
            ... dataset = datasets.load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
            ... dataset = dataset.map(some_transform)
            ... cuts = CutSet.from_huggingface_dataset(dataset)
            ... for cut in cuts:
            ...     pass

        Example providing HF dataset init args/kwargs::

            >>> import datasets
            ... cuts = CutSet.from_huggingface_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test")
            ... for cut in cuts:
            ...     pass

        """
        from lhotse.hf import LazyHFDatasetIterator

        return CutSet(
            LazyHFDatasetIterator(
                *dataset_args,
                audio_key=audio_key,
                text_key=text_key,
                lang_key=lang_key,
                gender_key=gender_key,
                **dataset_kwargs,
            )
        )

    def __repr__(self) -> str:
        try:
            len_val = len(self)
        except:
            len_val = "<unknown>"
        return f"CutSet(len={len_val}) [underlying data type: {type(self.data)}]"

    def __contains__(self, other: Union[str, Cut]) -> bool:
        if isinstance(other, str):
            return any(other == item.id for item in self)
        else:
            return any(other.id == item.id for item in self)

    def __getitem__(self, index_or_id: Union[int, str]) -> Cut:
        try:
            return self.cuts[index_or_id]  # int passed, eager manifest, fast
        except TypeError:
            # either lazy manifest or str passed, both are slow
            if self.is_lazy:
                return next(item for idx, item in enumerate(self) if idx == index_or_id)
            else:
                # string id passed, support just for backward compatibility, not recommended
                return next(item for item in self if item.id == index_or_id)

    def __len__(self) -> int:
        return len(self.cuts)

    def __iter__(self) -> Iterable[Cut]:
        yield from self.cuts


class _BackgroundCutFetcher(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet):
        assert len(cuts) == 1
        return cuts[0]


def mix(
    reference_cut: Cut,
    mixed_in_cut: Cut,
    offset: Seconds = 0,
    allow_padding: bool = False,
    snr: Optional[Decibels] = None,
    preserve_id: Optional[str] = None,
) -> MixedCut:
    """
    Overlay, or mix, two cuts. Optionally the ``mixed_in_cut`` may be shifted by ``offset`` seconds
    and scaled down (positive SNR) or scaled up (negative SNR).
    Returns a MixedCut, which contains both cuts and the mix information.
    The actual feature mixing is performed during the call to :meth:`~MixedCut.load_features`.

    :param reference_cut: The reference cut for the mix - offset and snr are specified w.r.t this cut.
    :param mixed_in_cut: The mixed-in cut - it will be offset and rescaled to match the offset and snr parameters.
    :param offset: How many seconds to shift the ``mixed_in_cut`` w.r.t. the ``reference_cut``.
    :param allow_padding: If the offset is larger than the cut duration, allow the cut to be padded.
    :param snr: Desired SNR of the ``right_cut`` w.r.t. the ``left_cut`` in the mix.
    :param preserve_id: optional string ("left", "right"). when specified, append will preserve the cut id
        of the left- or right-hand side argument. otherwise, a new random id is generated.
    :return: A :class:`~MixedCut` instance.
    """

    # Start with a series of sanity checks
    if (
        any(isinstance(cut, PaddingCut) for cut in (reference_cut, mixed_in_cut))
        and snr is not None
    ):
        warnings.warn(
            "You are mixing cuts to a padding cut with a specified SNR - "
            "the resulting energies would be extremely low or high. "
            "We are setting snr to None, so that the original signal energies will be retained instead."
        )
        snr = None

    if reference_cut.num_features is not None and mixed_in_cut.num_features is not None:
        assert (
            reference_cut.num_features == mixed_in_cut.num_features
        ), "Cannot mix cuts with different feature dimensions."
    assert offset <= reference_cut.duration or allow_padding, (
        f"Cannot mix cut '{mixed_in_cut.id}' with offset {offset},"
        f" which is greater than cuts {reference_cut.id} duration"
        f" of {reference_cut.duration}. Set `allow_padding=True` to allow padding."
    )
    assert reference_cut.sampling_rate == mixed_in_cut.sampling_rate, (
        f"Cannot mix cuts with different sampling rates "
        f"({reference_cut.sampling_rate} vs. "
        f"{mixed_in_cut.sampling_rate}). "
        f"Please resample the recordings first."
    )

    # If either of the cuts is a MultiCut, we need to further check a few things.
    if isinstance(reference_cut, MultiCut) or isinstance(mixed_in_cut, MultiCut):
        # If both are MultiCuts, we need to check that they point to the same channels
        if isinstance(reference_cut, MultiCut) and isinstance(mixed_in_cut, MultiCut):
            assert (
                reference_cut.channel == mixed_in_cut.channel
            ), "Cannot mix MultiCuts with different channel ids."
        # If only one of them is a MultiCut and the other is a MixedCut, we need to check
        # all the tracks of the MixedCut to make sure they point to the same channels.
        if isinstance(reference_cut, MixedCut) or isinstance(mixed_in_cut, MixedCut):
            if isinstance(reference_cut, MixedCut):
                mixed_cut = reference_cut
                multi_cut = mixed_in_cut
            else:
                mixed_cut = mixed_in_cut
                multi_cut = reference_cut
            assert all(
                track.type != "MultiCut" or track.cut.channel == multi_cut.channel
                for track in mixed_cut.tracks
            ), "Cannot mix a MultiCut with a MixedCut that contains MultiCuts with different channel ids."

    # Determine the ID of the result.
    if preserve_id is None:
        mixed_cut_id = str(uuid4())
    elif preserve_id == "left":
        mixed_cut_id = reference_cut.id
    elif preserve_id == "right":
        mixed_cut_id = mixed_in_cut.id
    else:
        raise ValueError(
            "Unexpected value for 'preserve_id' argument: "
            f"got '{preserve_id}', expected one of (None, 'left', 'right')."
        )

    # If the offset is larger than the left_cut duration, pad it.
    if offset > reference_cut.duration:
        reference_cut = reference_cut.pad(duration=offset)

    # When the left_cut is a MixedCut and it does not have existing transforms,
    # take its existing tracks, otherwise create a new track.
    if (
        isinstance(reference_cut, MixedCut)
        and len(ifnone(reference_cut.transforms, [])) == 0
    ):
        old_tracks = reference_cut.tracks
    elif isinstance(reference_cut, (DataCut, PaddingCut, MixedCut)):
        old_tracks = [MixTrack(cut=reference_cut)]
    else:
        raise ValueError(f"Unsupported type of cut in mix(): {type(reference_cut)}")

    # When the right_cut is a MixedCut, adapt its existing tracks with the new offset and snr,
    # otherwise create a new track.
    if isinstance(mixed_in_cut, MixedCut):
        # Similarly for mixed_in_cut, if it is a MixedCut and it does not have existing transforms,
        # take its existing tracks, otherwise create a new track.
        if len(ifnone(mixed_in_cut.transforms, [])) > 0:
            new_tracks = [MixTrack(cut=mixed_in_cut, offset=offset, snr=snr)]
        else:
            new_tracks = [
                MixTrack(
                    cut=track.cut,
                    offset=round(track.offset + offset, ndigits=8),
                    snr=(
                        # When no new SNR is specified, retain whatever was there in the first place.
                        track.snr
                        if snr is None
                        # When new SNR is specified but none was specified before, assign the new SNR value.
                        else snr
                        if track.snr is None
                        # When both new and previous SNR were specified, assign their sum,
                        # as the SNR for each track is defined with regard to the first track energy.
                        else track.snr + snr
                        if snr is not None and track is not None
                        # When no SNR was specified whatsoever, use none.
                        else None
                    ),
                )
                for track in mixed_in_cut.tracks
            ]
    elif isinstance(mixed_in_cut, (DataCut, PaddingCut)):
        new_tracks = [MixTrack(cut=mixed_in_cut, offset=offset, snr=snr)]
    else:
        raise ValueError(f"Unsupported type of cut in mix(): {type(mixed_in_cut)}")

    return MixedCut(id=mixed_cut_id, tracks=old_tracks + new_tracks)


def pad(
    cut: Cut,
    duration: Seconds = None,
    num_frames: int = None,
    num_samples: int = None,
    pad_feat_value: float = LOG_EPSILON,
    direction: str = "right",
    preserve_id: bool = False,
    pad_value_dict: Optional[Dict[str, Union[int, float]]] = None,
) -> Cut:
    """
    Return a new MixedCut, padded with zeros in the recording, and ``pad_feat_value`` in each feature bin.

    The user can choose to pad either to a specific `duration`; a specific number of frames `num_frames`;
    or a specific number of samples `num_samples`. The three arguments are mutually exclusive.

    :param cut: DataCut to be padded.
    :param duration: The cut's minimal duration after padding.
    :param num_frames: The cut's total number of frames after padding.
    :param num_samples: The cut's total number of samples after padding.
    :param pad_feat_value: A float value that's used for padding the features.
        By default we assume a log-energy floor of approx. -23 (1e-10 after exp).
    :param direction: string, 'left', 'right' or 'both'. Determines whether the padding is added before or after
        the cut.
    :param preserve_id: When ``True``, preserves the cut ID before padding.
        Otherwise, a new random ID is generated for the padded cut (default).
    :param pad_value_dict: Optional dict that specifies what value should be used
        for padding arrays in custom attributes.
    :return: a padded MixedCut if duration is greater than this cut's duration, otherwise ``self``.
    """
    assert exactly_one_not_null(duration, num_frames, num_samples), (
        f"Expected only one of (duration, num_frames, num_samples) to be set: "
        f"got ({duration}, {num_frames}, {num_samples})"
    )
    if hasattr(cut, "custom") and isinstance(cut.custom, dict):
        from lhotse.array import TemporalArray

        arr_keys = [k for k, v in cut.custom.items() if isinstance(v, TemporalArray)]
        if len(arr_keys) > 0:
            padding_values_specified = (
                pad_value_dict is not None
                and all(k in pad_value_dict for k in arr_keys),
            )
            if not padding_values_specified:
                warnings.warn(
                    f"Cut being padded has custom TemporalArray attributes: {arr_keys}. "
                    f"We expected a 'pad_value_dict' argument with padding values for these attributes. "
                    f"We will proceed and use the default padding value (={DEFAULT_PADDING_VALUE})."
                )

    if duration is not None:
        if duration <= cut.duration:
            return cut
        total_num_frames = (
            compute_num_frames(
                duration=duration,
                frame_shift=cut.frame_shift,
                sampling_rate=cut.sampling_rate,
            )
            if cut.has_features
            else None
        )
        total_num_samples = (
            compute_num_samples(duration=duration, sampling_rate=cut.sampling_rate)
            if cut.has_recording
            else None
        )

    if num_frames is not None:
        assert cut.has_features, (
            "Cannot pad a cut using num_frames when it is missing pre-computed features "
            "(did you run cut.compute_and_store_features(...)?)."
        )
        total_num_frames = num_frames
        duration = total_num_frames * cut.frame_shift
        total_num_samples = (
            compute_num_samples(duration=duration, sampling_rate=cut.sampling_rate)
            if cut.has_recording
            else None
        )
        # It is possible that two cuts have the same number of frames,
        # but they differ in the number of samples.
        # In that case, we need to pad them anyway so that they have truly equal durations.
        if (
            total_num_frames <= cut.num_frames
            and duration <= cut.duration
            and (total_num_samples is None or total_num_samples <= cut.num_samples)
        ):
            return cut

    if num_samples is not None:
        assert cut.has_recording, (
            "Cannot pad a cut using num_samples when it is missing a Recording object "
            "(did you attach recording/recording set when creating the cut/cut set?)"
        )
        if num_samples <= cut.num_samples:
            return cut
        total_num_samples = num_samples
        duration = total_num_samples / cut.sampling_rate
        total_num_frames = (
            compute_num_frames(
                duration=duration,
                frame_shift=cut.frame_shift,
                sampling_rate=cut.sampling_rate,
            )
            if cut.has_features
            else None
        )

    padding_duration = round(duration - cut.duration, ndigits=8)

    video = None
    if cut.has_video:
        video = cut.video
        video = video.copy_with(
            num_frames=compute_num_samples(padding_duration, video.fps)
        )

    padding_cut = PaddingCut(
        id=str(uuid4()),
        duration=padding_duration,
        feat_value=pad_feat_value,
        num_features=cut.num_features,
        # The num_frames and sampling_rate fields are tricky, because it is possible to create a MixedCut
        # from Cuts that have different sampling rates and frame shifts. In that case, we are assuming
        # that we should use the values from the reference cut, i.e. the first one in the mix.
        num_frames=(total_num_frames - cut.num_frames if cut.has_features else None),
        num_samples=(
            total_num_samples - cut.num_samples if cut.has_recording else None
        ),
        frame_shift=cut.frame_shift,
        sampling_rate=cut.sampling_rate,
        video=video,
        custom=pad_value_dict,
    )

    if direction == "right":
        padded = cut.append(padding_cut, preserve_id="left" if preserve_id else None)
    elif direction == "left":
        padded = padding_cut.append(cut, preserve_id="right" if preserve_id else None)
    elif direction == "both":
        padded = (
            padding_cut.truncate(duration=padding_cut.duration / 2)
            .append(cut, preserve_id="right" if preserve_id else None)
            .append(
                padding_cut.truncate(duration=padding_cut.duration / 2),
                preserve_id="left" if preserve_id else None,
            )
        )
    else:
        raise ValueError(f"Unknown type of padding: {direction}")

    return padded


def append(
    left_cut: Cut,
    right_cut: Cut,
    snr: Optional[Decibels] = None,
    preserve_id: Optional[str] = None,
) -> MixedCut:
    """Helper method for functional-style appending of Cuts."""
    return left_cut.append(right_cut, snr=snr, preserve_id=preserve_id)


def mix_cuts(cuts: Iterable[Cut]) -> MixedCut:
    """Return a MixedCut that consists of the input Cuts mixed with each other as-is."""
    # The following is a fold (accumulate/aggregate) operation; it starts with cuts[0], and mixes it with cuts[1];
    #  then takes their mix and mixes it with cuts[2]; and so on.
    return reduce(mix, cuts)


def append_cuts(cuts: Iterable[Cut]) -> Cut:
    """Return a MixedCut that consists of the input Cuts appended to each other as-is."""
    # The following is a fold (accumulate/aggregate) operation; it starts with cuts[0], and appends cuts[1] to it;
    #  then takes their it concatenation and appends cuts[2] to it; and so on.
    return reduce(append, cuts)


def compute_supervisions_frame_mask(
    cut: Cut,
    frame_shift: Optional[Seconds] = None,
    use_alignment_if_exists: Optional[str] = None,
):
    """
    Compute a mask that indicates which frames in a cut are covered by supervisions.

    :param cut: a cut object.
    :param frame_shift: optional frame shift in seconds; required when the cut does not have
        pre-computed features, otherwise ignored.
    :param use_alignment_if_exists: optional str (key from alignment dict); use the specified
        alignment type for generating the mask
    :returns a 1D numpy array with value 1 for **frames** covered by at least one supervision,
    and 0 for **frames** not covered by any supervision.
    """
    assert cut.has_features or frame_shift is not None, (
        f"No features available. "
        f"Either pre-compute features or provide frame_shift."
    )
    if cut.has_features:
        frame_shift = cut.frame_shift
        num_frames = cut.num_frames
    else:
        num_frames = compute_num_frames(
            duration=cut.duration,
            frame_shift=frame_shift,
            sampling_rate=cut.sampling_rate,
        )
    mask = np.zeros(num_frames, dtype=np.float32)
    for supervision in cut.supervisions:
        if (
            use_alignment_if_exists
            and supervision.alignment
            and use_alignment_if_exists in supervision.alignment
        ):
            for ali in supervision.alignment[use_alignment_if_exists]:
                st = round(ali.start / frame_shift) if ali.start > 0 else 0
                et = (
                    round(ali.end / frame_shift)
                    if ali.end < cut.duration
                    else num_frames
                )
                mask[st:et] = 1.0
        else:
            st = round(supervision.start / frame_shift) if supervision.start > 0 else 0
            et = (
                round(supervision.end / frame_shift)
                if supervision.end < cut.duration
                else num_frames
            )
            mask[st:et] = 1.0
    return mask


def create_cut_set_eager(
    recordings: Optional[RecordingSet] = None,
    supervisions: Optional[SupervisionSet] = None,
    features: Optional[FeatureSet] = None,
    output_path: Optional[Pathlike] = None,
    random_ids: bool = False,
    tolerance: Seconds = 0.001,
) -> CutSet:
    """
    Create a :class:`.CutSet` from any combination of supervision, feature and recording manifests.
    At least one of ``recordings`` or ``features`` is required.

    The created cuts will be of type :class:`.DataCut` (MonoCut for single-channel and MultiCut for multi-channel).
    The :class:`.DataCut` boundaries correspond to those found in the ``features``, when available,
    otherwise to those found in the ``recordings``.

    When ``supervisions`` are provided, we'll be searching them for matching recording IDs
    and attaching to created cuts, assuming they are fully within the cut's time span.

    :param recordings: an optional :class:`~lhotse.audio.RecordingSet` manifest.
    :param supervisions: an optional :class:`~lhotse.supervision.SupervisionSet` manifest.
    :param features: an optional :class:`~lhotse.features.base.FeatureSet` manifest.
    :param output_path: an optional path where the :class:`.CutSet` is stored.
    :param random_ids: boolean, should the cut IDs be randomized. By default, use the recording ID
        with a loop index and a channel idx, i.e. "{recording_id}-{idx}-{channel}")
    :param tolerance: float, tolerance for supervision and feature segment boundary comparison.
        By default, it's 1ms. Increasing this value can be helpful when importing Kaldi data
        directories with precomputed features.
    :return: a new :class:`.CutSet` instance.
    """
    assert (
        features is not None or recordings is not None
    ), "At least one of 'features' or 'recordings' has to be provided."
    sup_ok, feat_ok, rec_ok = (
        supervisions is not None,
        features is not None,
        recordings is not None,
    )
    if sup_ok:
        supervisions = supervisions.to_eager()  # must be eager to use .find()
    if feat_ok:
        # Case I: Features are provided.
        # Use features to determine the cut boundaries and attach recordings and supervisions as available.
        if rec_ok:
            recordings = recordings.to_eager()  # ensure it can be indexed with cut.id
        cuts = []
        for idx, feats in enumerate(features):
            is_mono = (
                feats.channels is None
                or isinstance(feats.channels, int)
                or len(feats.channels) == 1
            )
            if is_mono:
                cls = MonoCut
                channel = feats.channels if feats.channels is not None else 0
            else:
                cls = MultiCut
                channel = list(feats.channels)
            cuts.append(
                cls(
                    id=str(uuid4()) if random_ids else f"{feats.recording_id}-{idx}",
                    start=feats.start,
                    duration=feats.duration,
                    channel=channel,
                    features=feats,
                    recording=recordings[feats.recording_id] if rec_ok else None,
                    # The supervisions' start times are adjusted if the features object starts at time other than 0s.
                    supervisions=list(
                        supervisions.find(
                            recording_id=feats.recording_id,
                            channel=channel,
                            start_after=feats.start,
                            end_before=feats.end,
                            adjust_offset=True,
                            tolerance=tolerance,
                        )
                    )
                    if sup_ok
                    else [],
                )
            )
    else:
        # Case II: Recordings are provided (and features are not).
        # Use recordings to determine the cut boundaries.
        cuts = []
        for ridx, recording in enumerate(recordings):
            if recording.num_channels == 1:
                cls = MonoCut
                channel = recording.channel_ids[0]
            else:
                cls = MultiCut
                channel = recording.channel_ids
            cuts.append(
                cls(
                    id=str(uuid4()) if random_ids else f"{recording.id}-{ridx}",
                    start=0,
                    duration=recording.duration,
                    channel=channel,
                    recording=recording,
                    supervisions=list(supervisions.find(recording_id=recording.id))
                    if sup_ok
                    else [],
                )
            )
    cuts = CutSet(cuts)
    if output_path is not None:
        cuts.to_file(output_path)
    return cuts


def create_cut_set_lazy(
    output_path: Pathlike,
    recordings: Optional[RecordingSet] = None,
    supervisions: Optional[SupervisionSet] = None,
    features: Optional[FeatureSet] = None,
    random_ids: bool = False,
    tolerance: Seconds = 0.001,
) -> CutSet:
    """
    Create a :class:`.CutSet` from any combination of supervision, feature and recording manifests.
    At least one of ``recordings`` or ``features`` is required.

    This method is the "lazy" variant, which allows to create a :class:`.CutSet` with a minimal memory usage.
    It has some extra requirements:

        - The user must provide an ``output_path``, where we will write the cuts as
            we create them. We'll return a lazily-opened :class:`CutSet` from that file.

        - ``recordings`` and ``features`` (if both provided) have to be of equal length
            and sorted by ``recording_id`` attribute of their elements.

        - ``supervisions`` (if provided) have to be sorted by ``recording_id``;
            note that there may be multiple supervisions with the same ``recording_id``,
            which is allowed.

    In addition, to prepare cuts in a fully memory-efficient way, make sure that:

        - All input manifests are stored in JSONL format and opened lazily
            with ``<manifest_class>.from_jsonl_lazy(path)`` method.

    For more details, see :func:`.create_cut_set_eager`.

    :param output_path: path to which we will write the cuts.
    :param recordings: an optional :class:`~lhotse.audio.RecordingSet` manifest.
    :param supervisions: an optional :class:`~lhotse.supervision.SupervisionSet` manifest.
    :param features: an optional :class:`~lhotse.features.base.FeatureSet` manifest.
    :param random_ids: boolean, should the cut IDs be randomized. By default, use the recording ID
        with a loop index and a channel idx, i.e. "{recording_id}-{idx}-{channel}")
    :param tolerance: float, tolerance for supervision and feature segment boundary comparison.
        By default, it's 1ms. Increasing this value can be helpful when importing Kaldi data
        directories with precomputed features.
    :return: a new :class:`.CutSet` instance.
    """
    assert (
        output_path is not None
    ), "You must provide the 'output_path' argument to create a CutSet lazily."
    assert (
        features is not None or recordings is not None
    ), "At least one of 'features' or 'recordings' has to be provided."
    sup_ok, feat_ok, rec_ok = (
        supervisions is not None,
        features is not None,
        recordings is not None,
    )
    for mtype, m in [
        ("recordings", recordings),
        ("supervisions", supervisions),
        ("features", features),
    ]:
        if m is not None and not m.is_lazy:
            logging.info(
                f"Manifest passed in argument '{mtype}' is not opened lazily; "
                f"open it with {type(m).__name__}.from_jsonl_lazy() to reduce the memory usage of this method."
            )
    if feat_ok:
        # Case I: Features are provided.
        # Use features to determine the cut boundaries and attach recordings and supervisions as available.

        recordings = iter(recordings) if rec_ok else itertools.repeat(None)
        # Find the supervisions that have corresponding recording_id;
        # note that if the supervisions are not sorted, we can't fail here,
        # because there might simply be no supervisions with that ID.
        # It's up to the user to make sure it's sorted properly.
        supervisions = iter(supervisions) if sup_ok else itertools.repeat(None)

        with CutSet.open_writer(output_path) as writer:
            for idx, feats in enumerate(features):
                rec = next(recordings)
                assert rec is None or rec.id == feats.recording_id, (
                    f"Mismatched recording_id: Features.recording_id == {feats.recording_id}, "
                    f"but Recording.id == '{rec.id}'"
                )
                sups, supervisions = _takewhile(
                    supervisions, lambda s: s.recording_id == feats.recording_id
                )
                sups = SupervisionSet.from_segments(sups)

                is_mono = (
                    feats.channels is None
                    or isinstance(feats.channels, int)
                    or len(feats.channels) == 1
                )
                if is_mono:
                    cls = MonoCut
                    channel = feats.channels if feats.channels is not None else 0
                else:
                    cls = MultiCut
                    channel = list(feats.channels)

                cut = cls(
                    id=str(uuid4()) if random_ids else f"{feats.recording_id}-{idx}",
                    start=feats.start,
                    duration=feats.duration,
                    channel=channel,
                    features=feats,
                    recording=rec,
                    # The supervisions' start times are adjusted if the features object starts at time other than 0s.
                    supervisions=list(
                        sups.find(
                            recording_id=feats.recording_id,
                            channel=channel,
                            start_after=feats.start,
                            end_before=feats.end,
                            adjust_offset=True,
                            tolerance=tolerance,
                        )
                    )
                    if sup_ok
                    else [],
                )
                writer.write(cut)
        return CutSet.from_jsonl_lazy(output_path)

    # Case II: Recordings are provided (and features are not).
    # Use recordings to determine the cut boundaries.

    supervisions = iter(supervisions) if sup_ok else itertools.repeat(None)

    with CutSet.open_writer(output_path) as writer:
        for ridx, recording in enumerate(recordings):
            # Find the supervisions that have corresponding recording_id;
            # note that if the supervisions are not sorted, we can't fail here,
            # because there might simply be no supervisions with that ID.
            # It's up to the user to make sure it's sorted properly.
            sups, supervisions = _takewhile(
                supervisions, lambda s: s.recording_id == recording.id
            )
            sups = SupervisionSet.from_segments(sups)

            if recording.num_channels == 1:
                cls = MonoCut
                channel = recording.channel_ids[0]
            else:
                cls = MultiCut
                channel = recording.channel_ids
            cut = cls(
                id=str(uuid4()) if random_ids else f"{recording.id}-{ridx}",
                start=0,
                duration=recording.duration,
                channel=channel,
                recording=recording,
                supervisions=list(sups.find(recording_id=recording.id))
                if sup_ok
                else [],
            )
            writer.write(cut)

    return CutSet.from_jsonl_lazy(output_path)


T = TypeVar("T")


def _takewhile(
    iterable: Iterable[T], predicate: Callable[[T], bool]
) -> Tuple[List[T], Iterable[T]]:
    """
    Collects items from ``iterable`` as long as they satisfy the ``predicate``.
    Returns a tuple of ``(collected_items, iterable)``, where ``iterable`` may
    continue yielding items starting from the first one that did not satisfy
    ``predicate`` (unlike ``itertools.takewhile``).
    """
    collected = []
    try:
        while True:
            item = next(iterable)
            if predicate(item):
                collected.append(item)
            else:
                iterable = chain([item], iterable)
                break

    except StopIteration:
        pass
    return collected, iterable


def deserialize_cut(raw_cut: dict) -> Cut:
    cut_type = raw_cut.pop("type")
    if cut_type == "MonoCut":
        return MonoCut.from_dict(raw_cut)
    if cut_type == "MultiCut":
        return MultiCut.from_dict(raw_cut)
    if cut_type == "PaddingCut":
        return PaddingCut.from_dict(raw_cut)
    if cut_type == "Cut":
        warnings.warn(
            "Your manifest was created with Lhotse version earlier than v0.8, when MonoCut was called Cut. "
            "Please re-generate it with Lhotse v0.8 as it might stop working in a future version "
            "(using manifest.from_file() and then manifest.to_file() should be sufficient)."
        )
        return MonoCut.from_dict(raw_cut)
    if cut_type == "MixedCut":
        return MixedCut.from_dict(raw_cut)
    raise ValueError(f"Unexpected cut type during deserialization: '{cut_type}'")


def _cut_into_windows_single(
    cuts: CutSet,
    duration,
    hop,
    keep_excessive_supervisions,
) -> CutSet:
    return cuts.cut_into_windows(
        duration=duration,
        hop=hop,
        keep_excessive_supervisions=keep_excessive_supervisions,
    ).to_eager()


def _trim_to_supervisions_single(
    cuts: CutSet,
    keep_overlapping,
    min_duration,
    context_direction,
    keep_all_channels,
) -> CutSet:
    return cuts.trim_to_supervisions(
        keep_overlapping=keep_overlapping,
        min_duration=min_duration,
        context_direction=context_direction,
        keep_all_channels=keep_all_channels,
    ).to_eager()


def _trim_to_alignments_single(
    cuts: CutSet,
    type,
    max_pause,
    max_segment_duration,
    delimiter,
    keep_all_channels,
) -> CutSet:
    return cuts.trim_to_alignments(
        type=type,
        max_pause=max_pause,
        max_segment_duration=max_segment_duration,
        delimiter=delimiter,
        keep_all_channels=keep_all_channels,
    ).to_eager()


def _trim_to_supervision_groups_single(
    cuts: CutSet,
    max_pause: Seconds,
) -> CutSet:
    return cuts.trim_to_supervision_groups(
        max_pause=max_pause,
    ).to_eager()


def _add_recording_path_prefix_single(cut, path):
    return cut.with_recording_path_prefix(path)


def _add_features_path_prefix_single(cut, path):
    return cut.with_features_path_prefix(path)


def _with_id(cut, transform_fn):
    return cut.with_id(transform_fn(cut.id))


def _fill_supervision(cut, add_empty, shrink_ok):
    return cut.fill_supervision(add_empty=add_empty, shrink_ok=shrink_ok)


def _map_supervisions(cut, transform_fn):
    return cut.map_supervisions(transform_fn)


def _transform_text(sup, transform_fn):
    return sup.transform_text(transform_fn)


def _filter_supervisions(cut, predicate):
    return cut.filter_supervisions(predicate)


def _merge_supervisions(cut, merge_policy, custom_merge_fn):
    return cut.merge_supervisions(
        merge_policy=merge_policy, custom_merge_fn=custom_merge_fn
    )


def _pad(cut, *args, **kwargs):
    return cut.pad(*args, **kwargs)


def _extend_by(cut, *args, **kwargs):
    return cut.extend_by(*args, **kwargs)


def _resample(cut, *args, **kwargs):
    return cut.resample(*args, **kwargs)


def _perturb_speed(cut, *args, **kwargs):
    return cut.perturb_speed(*args, **kwargs)


def _perturb_tempo(cut, *args, **kwargs):
    return cut.perturb_tempo(*args, **kwargs)


def _perturb_volume(cut, *args, **kwargs):
    return cut.perturb_volume(*args, **kwargs)


def _reverb_rir(cut, *args, **kwargs):
    return cut.reverb_rir(*args, **kwargs)


def _normalize_loudness(cut, *args, **kwargs):
    return cut.normalize_loudness(*args, **kwargs)


def _dereverb_wpe(cut, *args, **kwargs):
    return cut.dereverb_wpe(*args, **kwargs)


def _drop_features(cut, *args, **kwargs):
    return cut.drop_features(*args, **kwargs)


def _drop_recordings(cut, *args, **kwargs):
    return cut.drop_recording(*args, **kwargs)


def _drop_alignments(cut, *args, **kwargs):
    return cut.drop_alignments(*args, **kwargs)


def _drop_supervisions(cut, *args, **kwargs):
    return cut.drop_supervisions(*args, **kwargs)


def _drop_in_memory_data(cut, *args, **kwargs):
    return cut.drop_in_memory_data(*args, **kwargs)


def _truncate_single(
    cut: Cut,
    max_duration: Seconds,
    offset_type: str,
    keep_excessive_supervisions: bool = True,
    preserve_id: bool = False,
    rng: Optional[random.Random] = None,
) -> Cut:
    if cut.duration <= max_duration:
        return cut

    def compute_offset():
        if offset_type == "start":
            return 0.0
        last_offset = cut.duration - max_duration
        if offset_type == "end":
            return last_offset
        if offset_type == "random":
            if rng is None:
                return random.uniform(0.0, last_offset)
            else:
                return rng.uniform(0.0, last_offset)
        raise ValueError(f"Unknown 'offset_type' option: {offset_type}")

    return cut.truncate(
        offset=compute_offset(),
        duration=max_duration,
        keep_excessive_supervisions=keep_excessive_supervisions,
        preserve_id=preserve_id,
    )


def _export_to_shar_single(
    cuts: CutSet,
    output_dir: Pathlike,
    shard_size: Optional[int],
    shard_offset: int,
    fields: Dict[str, str],
    warn_unused_fields: bool,
    include_cuts: bool,
    shard_suffix: Optional[str],
    verbose: bool,
    fault_tolerant: bool,
    preload: bool = False,
) -> Dict[str, List[str]]:
    from lhotse.shar import SharWriter

    pbar = tqdm(desc="Exporting to SHAR", disable=not verbose)

    if preload:
        # In the multi-threaded case we only read a single shard so it's quick,
        # and it allows us to overwrite a temporary cut manifest.
        cuts = cuts.to_eager()

    with SharWriter(
        output_dir=output_dir,
        fields=fields,
        shard_size=shard_size,
        shard_offset=shard_offset,
        warn_unused_fields=warn_unused_fields,
        include_cuts=include_cuts,
        shard_suffix=shard_suffix,
    ) as writer:
        for cut in cuts:
            try:
                writer.write(cut)
            except Exception as e:
                if fault_tolerant:
                    logging.warning(
                        f"Skipping: failed to load cut '{cut.id}'. Error message: {e}."
                    )
                else:
                    raise
            pbar.update()

    # Finally, return the list of output files.
    return writer.output_paths


class LazyCutMixer(Dillable):
    """
    Iterate over cuts from ``cuts`` CutSet while mixing randomly sampled ``mix_in_cuts`` into them.
    A typical application would be data augmentation with noise, music, babble, etc.

    :param cuts: a ``CutSet`` we are iterating over.
    :param mix_in_cuts: a ``CutSet`` containing other cuts to be mixed into ``cuts``.
    :param duration: an optional float in seconds.
        When ``None``, we will preserve the duration of the cuts in ``self``
        (i.e. we'll truncate the mix if it exceeded the original duration).
        Otherwise, we will keep sampling cuts to mix in until we reach the specified
        ``duration`` (and truncate to that value, should it be exceeded).
    :param allow_padding: an optional bool.
        When it is ``True``, we will allow the offset to be larger than the reference
        cut by padding the reference cut.
    :param snr: an optional float, or pair (range) of floats, in decibels.
        When it's a single float, we will mix all cuts with this SNR level
        (where cuts in ``self`` are treated as signals, and cuts in ``cuts`` are treated as noise).
        When it's a pair of floats, we will uniformly sample SNR values from that range.
        When ``None``, we will mix the cuts without any level adjustment
        (could be too noisy for data augmentation).
    :param preserve_id: optional string ("left", "right"). when specified, append will preserve the cut id
        of the left- or right-hand side argument. otherwise, a new random id is generated.
    :param mix_prob: an optional float in range [0, 1].
        Specifies the probability of performing a mix.
        Values lower than 1.0 mean that some cuts in the output will be unchanged.
    :param seed: an optional int or "trng". Random seed for choosing the cuts to mix and the SNR.
        If "trng" is provided, we'll use the ``secrets`` module for non-deterministic results
        on each iteration. You can also directly pass a ``random.Random`` instance here.
    :param random_mix_offset: an optional bool.
        When ``True`` and the duration of the to be mixed in cut in longer than the original cut,
         select a random sub-region from the to be mixed in cut.
    :param stateful: when True, each time this object is iterated we will shuffle the noise cuts
        using a different random seed. This is useful when you often re-start the iteration and
        don't want to keep seeing the same noise examples. Enabled by default.
    """

    def __init__(
        self,
        cuts: "CutSet",
        mix_in_cuts: "CutSet",
        duration: Optional[Seconds] = None,
        allow_padding: bool = False,
        snr: Optional[Union[Decibels, Sequence[Decibels]]] = 20,
        preserve_id: Optional[str] = None,
        mix_prob: float = 1.0,
        seed: Union[int, Literal["trng", "randomized"], random.Random] = 42,
        random_mix_offset: bool = False,
        stateful: bool = True,
    ) -> None:
        self.source = cuts
        self.mix_in_cuts = mix_in_cuts
        self.duration = duration
        self.allow_padding = allow_padding
        self.snr = snr
        self.preserve_id = preserve_id
        self.mix_prob = mix_prob
        self.seed = seed
        self.random_mix_offset = random_mix_offset
        self.stateful = stateful
        self.num_times_iterated = 0

        assert 0.0 <= self.mix_prob <= 1.0
        assert self.duration is None or self.duration > 0
        if isinstance(self.snr, (tuple, list)):
            assert (
                len(self.snr) == 2
            ), f"SNR range must be a list or tuple with exactly two values (got: {snr})"
        else:
            assert isinstance(self.snr, (type(None), int, float))

    def __iter__(self):
        from lhotse.dataset.dataloading import resolve_seed

        if isinstance(self.seed, random.Random):
            rng = self.seed
        else:
            rng = random.Random(resolve_seed(self.seed) + self.num_times_iterated)
        if self.stateful:
            self.num_times_iterated += 1

        if self.mix_in_cuts.is_lazy:
            # If the noise input is lazy, we'll shuffle it approximately.
            # We set the shuffling buffer size to 2000 because that's the size of MUSAN,
            # so even if the user forgets to convert MUSAN to an eager manifest, they will
            # get roughly the same quality of noise randomness.
            # Note: we can't just call .to_eager() as the noise CutSet can technically be
            #       very large, or even hold data in-memory in case of webdataset/Lhotse Shar sources.
            def noise_gen():
                yield from self.mix_in_cuts.repeat().shuffle(rng=rng, buffer_size=2000)

        else:
            # Eager nose cuts are just fully reshuffled in a different order on each noise "epoch".
            def noise_gen():
                #
                while True:
                    yield from self.mix_in_cuts.shuffle(rng=rng)

        mix_in_cuts = iter(noise_gen())
        for cut in self.source:
            # Check whether we're going to mix something into the current cut
            # or pass it through unchanged.
            if not is_cut(cut) or rng.uniform(0.0, 1.0) > self.mix_prob:
                yield cut
                continue
            # Determine the SNR - either it's specified or we need to sample one.
            cut_snr = (
                rng.uniform(*self.snr)
                if isinstance(self.snr, (list, tuple))
                else self.snr
            )
            # Note: we subtract 0.05s (50ms) from the target duration to avoid edge cases
            #       where we mix in some noise cut that effectively has 0 frames of features.
            target_mixed_duration = round(
                self.duration if self.duration is not None else cut.duration - 0.05,
                ndigits=8,
            )
            # Actual mixing
            to_mix = next(mix_in_cuts)
            to_mix = self._maybe_truncate_cut(to_mix, target_mixed_duration, rng)
            mixed = cut.mix(other=to_mix, snr=cut_snr, preserve_id=self.preserve_id)
            # Did the user specify a duration?
            # If yes, we will ensure that shorter cuts have more noise mixed in
            # to "pad" them with at the end.
            # If no, we will mix in as many noise cuts as needed to cover complete
            # duration.
            mixed_in_duration = to_mix.duration
            # Keep sampling until we mixed in a "duration" amount of noise.
            # Note: we subtract 0.05s (50ms) from the target duration to avoid edge cases
            #       where we mix in some noise cut that effectively has 0 frames of features.
            while mixed_in_duration < target_mixed_duration - 0.05:
                to_mix = next(mix_in_cuts)
                to_mix = self._maybe_truncate_cut(
                    to_mix, target_mixed_duration - mixed_in_duration, rng
                )
                # Keep the SNR constant for each cut from "self".
                mixed = mixed.mix(
                    other=to_mix,
                    snr=cut_snr,
                    offset_other_by=mixed_in_duration,
                    allow_padding=self.allow_padding,
                    preserve_id=self.preserve_id,
                )
                # Since we're adding floats, we can be off by an epsilon and trigger
                # some assertions for exceeding duration; do precautionary rounding here.
                mixed_in_duration = round(
                    mixed_in_duration + to_mix.duration, ndigits=8
                )
            # We truncate the mixed to either the original duration or the requested duration.
            # Note: we don't use 'target_mixed_duration' here because it may have subtracted
            #       a tiny bit of actual target duration to avoid errors related to edge effects.
            mixed = mixed.truncate(
                duration=self.duration if self.duration is not None else cut.duration,
                preserve_id=self.preserve_id is not None,
            )
            yield mixed

    def _maybe_truncate_cut(
        self, cut: Cut, target_duration: Seconds, rng: random.Random
    ) -> Cut:
        if self.random_mix_offset and cut.duration > target_duration:
            cut = cut.truncate(
                offset=rng.uniform(0, cut.duration - target_duration),
                duration=target_duration,
            )
        return cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other) -> "LazyIteratorChain":
        return LazyIteratorChain(self, other)
