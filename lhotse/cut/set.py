import itertools
import logging
import pickle
import random
import warnings
from collections import Counter, defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor
from functools import partial, reduce
from itertools import chain, islice
from math import ceil
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
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
from typing_extensions import Literal

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
from lhotse.lazy import AlgorithmMixin
from lhotse.serialization import Serializable
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    DEFAULT_PADDING_VALUE,
    LOG_EPSILON,
    Decibels,
    Pathlike,
    Seconds,
    TimeSpan,
    compute_num_frames,
    compute_num_samples,
    deprecated,
    exactly_one_not_null,
    fastcopy,
    ifnone,
    index_by_id_and_check,
    is_module_available,
    split_manifest_lazy,
    split_sequence,
    uuid4,
)

FW = TypeVar("FW", bound=FeaturesWriter)


class CutSet(Serializable, AlgorithmMixin):
    """
    :class:`~lhotse.cut.CutSet` represents a collection of cuts, indexed by cut IDs.
    CutSet ties together all types of data -- audio, features and supervisions, and is suitable to represent
    training/dev/test sets.

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

    def __init__(self, cuts: Optional[Mapping[str, Cut]] = None) -> None:
        self.cuts = ifnone(cuts, {})

    def __eq__(self, other: "CutSet") -> bool:
        return self.cuts == other.cuts

    @property
    def data(self) -> Union[Dict[str, Cut], Iterable[Cut]]:
        """Alias property for ``self.cuts``"""
        return self.cuts

    @property
    def mixed_cuts(self) -> Dict[str, MixedCut]:
        return {id_: cut for id_, cut in self.cuts.items() if isinstance(cut, MixedCut)}

    @property
    def simple_cuts(self) -> Dict[str, MonoCut]:
        return {id_: cut for id_, cut in self.cuts.items() if isinstance(cut, MonoCut)}

    @property
    def multi_cuts(self) -> Dict[str, MultiCut]:
        return {id_: cut for id_, cut in self.cuts.items() if isinstance(cut, MultiCut)}

    @property
    def ids(self) -> Iterable[str]:
        return self.cuts.keys()

    @property
    def speakers(self) -> FrozenSet[str]:
        return frozenset(
            supervision.speaker for cut in self for supervision in cut.supervisions
        )

    @staticmethod
    def from_cuts(cuts: Iterable[Cut]) -> "CutSet":
        return CutSet(cuts=index_by_id_and_check(cuts))

    from_items = from_cuts

    @staticmethod
    def from_manifests(
        recordings: Optional[RecordingSet] = None,
        supervisions: Optional[SupervisionSet] = None,
        features: Optional[FeatureSet] = None,
        output_path: Optional[Pathlike] = None,
        random_ids: bool = False,
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
            )
        else:
            return create_cut_set_eager(
                recordings=recordings,
                supervisions=supervisions,
                features=features,
                output_path=output_path,
                random_ids=random_ids,
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
        Example::

        >>> cuts = LazySharIterator({
        ...     "cuts": ["pipe:curl https://my.page/cuts.000000.jsonl.gz"],
        ...     "recording": ["pipe:curl https://my.page/recording.000000.tar"],
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
        warn_unused_fields: bool = True,
        include_cuts: bool = True,
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
        The function returns a dict that maps field names to lists of saved shard paths.

        When ``shard_size`` is set to ``None``, we will disable automatic sharding and the
        shard number suffix will be omitted from the file names.

        The option ``warn_unused_fields`` will emit a warning when cuts have some data attached to them
        (e.g., recording, features, or custom arrays) but saving it was not specified via ``fields``.

        The option ``include_cuts`` controls whether we store the cuts alongside ``fields`` (true by default).
        Turning it off is useful when extending existing dataset with new fields/feature types,
        but the original cuts do not require any modification.

        See also: :class:`~lhotse.shar.writers.shar.SharWriter`,
            :meth:`~lhotse.cut.set.CutSet.to_shar`.
        """
        from lhotse.shar import SharWriter

        with SharWriter(
            output_dir=output_dir,
            fields=fields,
            shard_size=shard_size,
            warn_unused_fields=warn_unused_fields,
            include_cuts=include_cuts,
        ) as writer:
            for cut in self:
                writer.write(cut)

        return writer.output_paths

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
        if not is_module_available("tabulate"):
            raise ValueError(
                "Since Lhotse v1.11, this function requires the `tabulate` package to be "
                "installed. Please run 'pip install tabulate' to continue."
            )
        from tabulate import tabulate

        def convert_(seconds: float) -> Tuple[int, int, int]:
            hours, seconds = divmod(seconds, 3600)
            minutes, seconds = divmod(seconds, 60)
            return int(hours), int(minutes), ceil(seconds)

        def time_as_str_(seconds: float) -> str:
            h, m, s = convert_(seconds)
            return f"{h:02d}:{m:02d}:{s:02d}"

        def total_duration_(segments: List[TimeSpan]) -> float:
            return sum(segment.duration for segment in segments)

        cntrs = defaultdict(int)
        cut_custom, sup_custom = Counter(), Counter()
        cut_durations = []

        # The following is to store statistics about speaker times in the cuts
        speaking_time_durations, speech_durations = [], []

        if full:
            durations_by_num_speakers = defaultdict(list)
            single_durations, overlapped_durations = [], []

        for c in self:
            cut_durations.append(c.duration)
            if hasattr(c, "custom"):
                for key in ifnone(c.custom, ()):
                    cut_custom[key] += 1
            cntrs["recordings"] += int(c.has_recording)
            cntrs["features"] += int(c.has_features)

            # Total speaking time duration is computed by summing the duration of all
            # supervisions in the cut.
            for s in c.trimmed_supervisions:
                speaking_time_durations.append(s.duration)
                cntrs["supervisions"] += 1
                for key in ifnone(s.custom, ()):
                    sup_custom[key] += 1

            # Total speech duration is the sum of intervals where 1 or more speakers are
            # active.
            speech_durations.append(
                total_duration_(find_segments_with_speaker_count(c, min_speakers=1))
            )

            if full:
                # Duration of single-speaker segments
                single_durations.append(
                    total_duration_(
                        find_segments_with_speaker_count(
                            c, min_speakers=1, max_speakers=1
                        )
                    )
                )
                # Duration of overlapped segments
                overlapped_durations.append(
                    total_duration_(
                        find_segments_with_speaker_count(
                            c, min_speakers=2, max_speakers=None
                        )
                    )
                )
                # Durations by number of speakers (we assume that overlaps can happen between
                # at most 4 speakers. This is a reasonable assumption for most datasets.)
                durations_by_num_speakers[1].append(single_durations[-1])
                for num_spk in range(2, 5):
                    durations_by_num_speakers[num_spk].append(
                        total_duration_(
                            find_segments_with_speaker_count(
                                c, min_speakers=num_spk, max_speakers=num_spk
                            )
                        )
                    )

        total_sum = np.array(cut_durations).sum()

        cut_stats = []
        cut_stats.append(["Cuts count:", len(cut_durations)])
        cut_stats.append(["Total duration (hh:mm:ss)", time_as_str_(total_sum)])
        cut_stats.append(["mean", f"{np.mean(cut_durations):.1f}"])
        cut_stats.append(["std", f"{np.std(cut_durations):.1f}"])
        cut_stats.append(["min", f"{np.min(cut_durations):.1f}"])
        cut_stats.append(["25%", f"{np.percentile(cut_durations, 25):.1f}"])
        cut_stats.append(["50%", f"{np.median(cut_durations):.1f}"])
        cut_stats.append(["75%", f"{np.percentile(cut_durations, 75):.1f}"])
        cut_stats.append(["99%", f"{np.percentile(cut_durations, 99):.1f}"])
        cut_stats.append(["99.5%", f"{np.percentile(cut_durations, 99.5):.1f}"])
        cut_stats.append(["99.9%", f"{np.percentile(cut_durations, 99.9):.1f}"])
        cut_stats.append(["max", f"{np.max(cut_durations):.1f}"])

        for key, val in cntrs.items():
            cut_stats.append([f"{key.title()} available:", val])

        print("Cut statistics:")
        print(tabulate(cut_stats, tablefmt="fancy_grid"))

        if cut_custom:
            print("CUT custom fields:")
            for key, val in cut_custom.most_common():
                print(f"- {key} (in {val} cuts)")

        if sup_custom:
            print("SUPERVISION custom fields:")
            for key, val in sup_custom.most_common():
                cut_stats.append(f"- {key} (in {val} cuts)")

        total_speech = np.array(speech_durations).sum()
        total_speaking_time = np.array(speaking_time_durations).sum()
        total_silence = total_sum - total_speech
        speech_stats = []
        speech_stats.append(
            [
                "Total speech duration",
                time_as_str_(total_speech),
                f"{total_speech / total_sum:.2%} of recording",
            ]
        )
        speech_stats.append(
            [
                "Total speaking time duration",
                time_as_str_(total_speaking_time),
                f"{total_speaking_time / total_sum:.2%} of recording",
            ]
        )
        speech_stats.append(
            [
                "Total silence duration",
                time_as_str_(total_silence),
                f"{total_silence / total_sum:.2%} of recording",
            ]
        )
        if full:
            total_single = np.array(single_durations).sum()
            total_overlap = np.array(overlapped_durations).sum()
            speech_stats.append(
                [
                    "Single-speaker duration",
                    time_as_str_(total_single),
                    f"{total_single / total_sum:.2%} ({total_single / total_speech:.2%} of speech)",
                ]
            )
            speech_stats.append(
                [
                    "Overlapped speech duration",
                    time_as_str_(total_overlap),
                    f"{total_overlap / total_sum:.2%} ({total_overlap / total_speech:.2%} of speech)",
                ]
            )
        print("Speech duration statistics:")
        print(tabulate(speech_stats, tablefmt="fancy_grid"))

        if not full:
            return

        # Additional statistics for full report
        speaker_stats = [
            [
                "Number of speakers",
                "Duration (hh:mm:ss)",
                "Speaking time (hh:mm:ss)",
                "% of speech",
                "% of speaking time",
            ]
        ]
        for num_spk, durations in durations_by_num_speakers.items():
            speaker_sum = np.array(durations).sum()
            speaking_time = num_spk * speaker_sum
            speaker_stats.append(
                [
                    num_spk,
                    time_as_str_(speaker_sum),
                    time_as_str_(speaking_time),
                    f"{speaker_sum / total_speech:.2%}",
                    f"{speaking_time / total_speaking_time:.2%}",
                ]
            )

        speaker_stats.append(
            [
                "Total",
                time_as_str_(total_speech),
                time_as_str_(total_speaking_time),
                "100.00%",
                "100.00%",
            ]
        )

        print("Speech duration statistics by number of speakers:")
        print(tabulate(speaker_stats, headers="firstrow", tablefmt="fancy_grid"))

    def split(
        self, num_splits: int, shuffle: bool = False, drop_last: bool = False
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
            CutSet.from_cuts(subset)
            for subset in split_sequence(
                self, num_splits=num_splits, shuffle=shuffle, drop_last=drop_last
            )
        ]

    def split_lazy(
        self, output_dir: Pathlike, chunk_size: int, prefix: str = ""
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
        :return: a list of lazily opened chunk manifests.
        """
        return split_manifest_lazy(
            self, output_dir=output_dir, chunk_size=chunk_size, prefix=prefix
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
            if len(out) < first:
                logging.warning(
                    f"CutSet has only {len(out)} items but first {first} were requested."
                )
            return out

        if last is not None:
            assert last > 0
            if last > len(self):
                logging.warning(
                    f"CutSet has only {len(self)} items but last {last} required; not doing anything."
                )
                return self
            cut_ids = list(self.ids)[-last:]
            return CutSet.from_cuts(self[cid] for cid in cut_ids)

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
            cuts = CutSet.from_cuts(cut for cut in self if cut.id in id_set)
            if len(cuts) < len(cut_ids):
                logging.warning(
                    f"In CutSet.subset(cut_ids=...): expected {len(cut_ids)} cuts but got {len(cuts)} "
                    f"instead ({len(cut_ids) - len(cuts)} cut IDs were not in the CutSet)."
                )
            # Restore the requested cut_ids order.
            return CutSet.from_cuts(cuts[cid] for cid in cut_ids)

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
        return self.map(lambda cut: cut.filter_supervisions(predicate))

    def merge_supervisions(
        self, custom_merge_fn: Optional[Callable[[str, Iterable[Any]], Any]] = None
    ) -> "CutSet":
        """
        Return a copy of the cut that has all of its supervisions merged into
        a single segment.

        The new start is the start of the earliest superivion, and the new duration
        is a minimum spanning duration for all the supervisions.

        The text fields are concatenated with a whitespace, and all other string fields
        (including IDs) are prefixed with "cat#" and concatenated with a hash symbol "#".
        This is also applied to ``custom`` fields. Fields with a ``None`` value are omitted.

        :param custom_merge_fn: a function that will be called to merge custom fields values.
            We expect ``custom_merge_fn`` to handle all possible custom keys.
            When not provided, we will treat all custom values as strings.
            It will be called roughly like:
            ``custom_merge_fn(custom_key, [s.custom[custom_key] for s in sups])``
        """
        return self.map(
            lambda cut: cut.merge_supervisions(custom_merge_fn=custom_merge_fn)
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
            from lhotse.lazy import LazyFlattener, LazyMapper

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

    def trim_to_unsupervised_segments(self) -> "CutSet":
        """
        Return a new CutSet with Cuts created from segments that have no supervisions (likely
        silence or noise).

        :return: a ``CutSet``.
        """
        cuts = []
        for cut in self:
            segments = find_segments_with_speaker_count(
                cut, min_speakers=0, max_speakers=0
            )
            for span in segments:
                cuts.append(cut.truncate(offset=span.start, duration=span.duration))
        return CutSet.from_cuts(cuts)

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

    def sort_by_duration(self, ascending: bool = False) -> "CutSet":
        """
        Sort the CutSet according to cuts duration and return the result. Descending by default.
        """
        return CutSet.from_cuts(
            sorted(self, key=(lambda cut: cut.duration), reverse=not ascending)
        )

    def sort_like(self, other: "CutSet") -> "CutSet":
        """
        Sort the CutSet according to the order of cut IDs in ``other`` and return the result.
        """
        assert set(self.ids) == set(
            other.ids
        ), "sort_like() expects both CutSet's to have identical cut IDs."
        return CutSet.from_cuts(self[cid] for cid in other.ids)

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
            lambda cut: cut.pad(
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
        truncated_cuts = []
        for cut in self:
            if cut.duration <= max_duration:
                truncated_cuts.append(cut)
                continue

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

            truncated_cuts.append(
                cut.truncate(
                    offset=compute_offset(),
                    duration=max_duration,
                    keep_excessive_supervisions=keep_excessive_supervisions,
                    preserve_id=preserve_id,
                )
            )
        return CutSet.from_cuts(truncated_cuts)

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
            lambda cut: cut.extend_by(
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
            from lhotse.lazy import LazyFlattener, LazyMapper

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
        # TODO: We might want to make this more efficient in the future
        #  by holding a cached list of cut ids as a member of CutSet...
        cut_indices = random.sample(range(len(self)), min(n_cuts, len(self)))
        cuts = [self[idx] for idx in cut_indices]
        if n_cuts == 1:
            return cuts[0]
        return CutSet.from_cuts(cuts)

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
        return self.map(lambda cut: cut.resample(sampling_rate, affix_id=affix_id))

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
        return self.map(lambda cut: cut.perturb_speed(factor=factor, affix_id=affix_id))

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
        return self.map(lambda cut: cut.perturb_tempo(factor=factor, affix_id=affix_id))

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
        return self.map(
            lambda cut: cut.perturb_volume(factor=factor, affix_id=affix_id)
        )

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
            lambda cut: cut.reverb_rir(
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
        seed: int = 42,
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
        :param seed: an optional int. Random seed for choosing the cuts to mix and the SNR.
        :return: a new ``CutSet`` with mixed cuts.
        """
        assert 0.0 <= mix_prob <= 1.0
        assert duration is None or duration > 0
        if isinstance(snr, (tuple, list)):
            assert (
                len(snr) == 2
            ), f"SNR range must be a list or tuple with exactly two values (got: {snr})"
        else:
            assert isinstance(snr, (type(None), int, float))
        assert not cuts.is_lazy, (
            "Mixing of two CutSets does not support a lazy mixed-in CutSet ('cuts' argument), "
            "as it would be extremely inefficient. "
            "You can use 'cuts.to_eager()' on the function argument to fix this."
        )
        rng = random.Random(seed)
        mixed_cuts = []
        for cut in self:
            # Check whether we're going to mix something into the current cut
            # or pass it through unchanged.
            if rng.uniform(0.0, 1.0) > mix_prob:
                mixed_cuts.append(cut)
                continue
            to_mix = cuts.sample()
            # Determine the SNR - either it's specified or we need to sample one.
            cut_snr = rng.uniform(*snr) if isinstance(snr, (list, tuple)) else snr
            # Actual mixing
            mixed = cut.mix(other=to_mix, snr=cut_snr, preserve_id=preserve_id)
            # Did the user specify a duration?
            # If yes, we will ensure that shorter cuts have more noise mixed in
            # to "pad" them with at the end.
            # If no, we will mix in as many noise cuts as needed to cover complete
            # duration.
            mixed_in_duration = to_mix.duration
            # Keep sampling until we mixed in a "duration" amount of noise.
            # Note: we subtract 0.05s (50ms) from the target duration to avoid edge cases
            #       where we mix in some noise cut that effectively has 0 frames of features.
            while mixed_in_duration < (
                duration if duration is not None else cut.duration - 0.05
            ):
                to_mix = cuts.sample()
                # Keep the SNR constant for each cut from "self".
                mixed = mixed.mix(
                    other=to_mix,
                    snr=cut_snr,
                    offset_other_by=mixed_in_duration,
                    allow_padding=allow_padding,
                    preserve_id=preserve_id,
                )
                # Since we're adding floats, we can be off by an epsilon and trigger
                # some assertions for exceeding duration; do precautionary rounding here.
                mixed_in_duration = round(
                    mixed_in_duration + to_mix.duration, ndigits=8
                )
            # We truncate the mixed to either the original duration or the requested duration.
            mixed = mixed.truncate(
                duration=duration if duration is not None else cut.duration,
                preserve_id=preserve_id is not None,
            )
            mixed_cuts.append(mixed)
        return CutSet.from_cuts(mixed_cuts)

    def drop_features(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its extracted features.
        """
        return self.map(lambda cut: cut.drop_features())

    def drop_recordings(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its recordings.
        """
        return self.map(lambda cut: cut.drop_recording())

    def drop_supervisions(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its supervisions.
        """
        return self.map(lambda cut: cut.drop_supervisions())

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
        from lhotse.lazy import LazySlicer
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
            executor = ProcessPoolExecutor(num_jobs)

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
        import torch
        from torch.utils.data import DataLoader

        from lhotse.dataset import SingleCutSampler, UnsupervisedWaveformDataset
        from lhotse.qa import validate_features

        frame_shift = extractor.frame_shift

        # We're opening a sequential cuts writer that can resume previously interrupted
        # operation. It scans the input JSONL file for cut IDs that should be ignored.
        # Note: this only works when ``manifest_path`` argument was specified, otherwise
        # we hold everything in memory and start from scratch.
        cuts_writer = CutSet.open_writer(manifest_path, overwrite=overwrite)

        # We tell the sampler to ignore cuts that were already processed.
        # It will avoid I/O for reading them in the DataLoader.
        sampler = SingleCutSampler(self, max_duration=batch_duration)
        sampler.filter(lambda cut: cut.id not in cuts_writer.ignore_ids)
        dataset = UnsupervisedWaveformDataset(collate=False)
        dloader = DataLoader(
            dataset, batch_size=None, sampler=sampler, num_workers=num_workers
        )

        with cuts_writer, storage_type(
            storage_path, mode="w" if overwrite else "a"
        ) as feats_writer, tqdm(
            desc="Computing features in batches", total=sampler.num_cuts
        ) as progress:
            # Display progress bar correctly.
            progress.update(len(cuts_writer.ignore_ids))
            for batch in dloader:
                cuts = batch["cuts"]
                waves = batch["audio"]

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
                        waves, sampling_rate=cuts[0].sampling_rate
                    )

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
                        # the features attached.
                        feat_manifest.recording_id = cut.id
                        cut = MonoCut(
                            id=cut.id,
                            start=0,
                            duration=cut.duration,
                            channel=0,
                            # Update supervisions recording_id for consistency
                            supervisions=[
                                fastcopy(s, recording_id=cut.id)
                                for s in cut.supervisions
                            ],
                            features=feat_manifest,
                            recording=None,
                        )
                    cuts_writer.write(cut, flush=True)

                progress.update(len(cuts))

        # If ``manifest_path`` was provided, this is a lazy manifest;
        # otherwise everything is in memory.
        return cuts_writer.open_manifest()

    def save_audios(
        self,
        storage_path: Pathlike,
        format: str = "wav",
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        num_jobs: Optional[int] = None,
        executor: Optional[Executor] = None,
        augment_fn: Optional[AugmentFn] = None,
        progress_bar: bool = True,
    ) -> "CutSet":
        """
        Store waveforms of all cuts as audio recordings to disk.

        :param storage_path: The path to location where we will store the audio recordings.
            For each cut, a sub-directory will be created that starts with the first 3
            characters of the cut's ID. The audio recording is then stored in the sub-directory
            using filename ``{cut.id}.{format}``
        :param format: Audio format argument supported by ``torchaudio.save``. Default is ``wav``.
        :param encoding: Audio encoding argument supported by ``torchaudio.save``. See
            https://pytorch.org/audio/stable/backend.html#save (sox_io backend) and
            https://pytorch.org/audio/stable/backend.html#id3 (soundfile backend) for more details.
        :param bits_per_sample: Audio bits_per_sample argument supported by ``torchaudio.save``. See
            https://pytorch.org/audio/stable/backend.html#save (sox_io backend) and
            https://pytorch.org/audio/stable/backend.html#id3 (soundfile backend) for more details.
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
            return CutSet.from_cuts(
                progress(
                    cut.save_audio(
                        storage_path=file_storage_path(cut, storage_path),
                        encoding=encoding,
                        bits_per_sample=bits_per_sample,
                        augment_fn=augment_fn,
                    )
                    for cut in self
                )
            )

        # Parallel execution: prepare the CutSet splits
        cut_sets = self.split(num_jobs, shuffle=True)

        # Initialize the default executor if None was given
        if executor is None:
            executor = ProcessPoolExecutor(num_jobs)

        # Submit the chunked tasks to parallel workers.
        # Each worker runs the non-parallel version of this function inside.
        futures = [
            executor.submit(
                CutSet.save_audios,
                cs,
                storage_path=storage_path,
                encoding=encoding,
                bits_per_sample=bits_per_sample,
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
        return self.map(lambda cut: cut.with_id(transform_fn(cut.id)))

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
            lambda cut: cut.fill_supervision(add_empty=add_empty, shrink_ok=shrink_ok)
        )

    def map_supervisions(
        self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]
    ) -> "CutSet":
        """
        Modify the SupervisionSegments by `transform_fn` in this CutSet.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a new, modified CutSet.
        """
        return self.map(lambda cut: cut.map_supervisions(transform_fn))

    def transform_text(self, transform_fn: Callable[[str], str]) -> "CutSet":
        """
        Return a copy of this ``CutSet`` with all ``SupervisionSegments`` text transformed with ``transform_fn``.
        Useful for text normalization, phonetic transcription, etc.

        :param transform_fn: a function that accepts a string and returns a string.
        :return: a new, modified CutSet.
        """
        return self.map_supervisions(lambda s: s.transform_text(transform_fn))

    def __repr__(self) -> str:
        try:
            len_val = len(self)
        except:
            len_val = "<unknown>"
        return f"CutSet(len={len_val}) [underlying data type: {type(self.data)}]"

    def __contains__(self, item: Union[str, Cut]) -> bool:
        if isinstance(item, str):
            return item in self.cuts
        else:
            return item.id in self.cuts

    def __getitem__(self, cut_id_or_index: Union[int, str]) -> "Cut":
        if isinstance(cut_id_or_index, str):
            return self.cuts[cut_id_or_index]
        # ~100x faster than list(dict.values())[index] for 100k elements
        return next(
            val for idx, val in enumerate(self.cuts.values()) if idx == cut_id_or_index
        )

    def __len__(self) -> int:
        return len(self.cuts)

    def __iter__(self) -> Iterable[Cut]:
        return iter(self.cuts.values())


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

    if reference_cut.num_features is not None:
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

    # When the left_cut is a MixedCut, take its existing tracks, otherwise create a new track.
    if isinstance(reference_cut, MixedCut):
        old_tracks = reference_cut.tracks
    elif isinstance(reference_cut, (DataCut, PaddingCut)):
        old_tracks = [MixTrack(cut=reference_cut)]
    else:
        raise ValueError(f"Unsupported type of cut in mix(): {type(reference_cut)}")

    # When the right_cut is a MixedCut, adapt its existing tracks with the new offset and snr,
    # otherwise create a new track.
    if isinstance(mixed_in_cut, MixedCut):
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
        raise ValueError(f"Unsupported type of cut in mix(): {type(reference_cut)}")

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

    The user can choose to pad either to a specific `duration`; a specific number of frames `max_frames`;
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

    padding_cut = PaddingCut(
        id=str(uuid4()),
        duration=round(duration - cut.duration, ndigits=8),
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
                    supervisions=list(
                        supervisions.find(
                            recording_id=recording.id,
                        )
                    )
                    if sup_ok
                    else [],
                )
            )
    cuts = CutSet.from_cuts(cuts)
    if output_path is not None:
        cuts.to_file(output_path)
    return cuts


def create_cut_set_lazy(
    output_path: Pathlike,
    recordings: Optional[RecordingSet] = None,
    supervisions: Optional[SupervisionSet] = None,
    features: Optional[FeatureSet] = None,
    random_ids: bool = False,
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


def find_segments_with_speaker_count(
    cut: Cut, min_speakers: int = 0, max_speakers: Optional[int] = None
) -> List[TimeSpan]:
    """
    Given a Cut, find a list of intervals that contain the specified number of speakers.

    :param cuts: the Cut to search.
    :param min_speakers: the minimum number of speakers.
    :param max_speakers: the maximum number of speakers.
    :return: a list of TimeSpans.
    """
    if max_speakers is None:
        max_speakers = float("inf")

    assert (
        min_speakers >= 0 and min_speakers <= max_speakers
    ), f"min_speakers={min_speakers} and max_speakers={max_speakers} are not valid."

    # First take care of trivial cases.
    if min_speakers == 0 and max_speakers == float("inf"):
        return [TimeSpan(0, cut.duration)]
    if len(cut.supervisions) == 0:
        return [] if min_speakers > 0 else [TimeSpan(0, cut.duration)]

    # We collect all the timestamps of the supervisions in the cut. Each timestamp is
    # a tuple of (time, is_speaker_start).
    timestamps = []
    # Add timestamp for cut start
    timestamps.append((0.0, None))
    for segment in cut.supervisions:
        timestamps.append((segment.start, True))
        timestamps.append((segment.end, False))
    # Add timestamp for cut end
    timestamps.append((cut.duration, None))

    # Sort the timestamps. We need the following priority order:
    # 1. Time mark of the timestamp: lower time mark comes first.
    # 2. For timestamps with the same time mark, None < False < True.
    timestamps.sort(key=lambda x: (x[0], x[1] is not None, x[1] is True))

    # We remove the timestamps that are not relevant for the search. The desired range
    # is given by the range of the cut start and end timestamps.
    cut_start_idx, cut_end_idx = [i for i, t in enumerate(timestamps) if t[1] is None]
    timestamps = timestamps[cut_start_idx : cut_end_idx + 1]

    # Now we iterate over the timestamps and count the number of speakers in any
    # given time interval. If the number of speakers is in the desired range,
    # we keep the interval.
    num_speakers = 0
    seg_start = 0.0
    intervals = []
    for timestamp, is_start in timestamps[1:]:
        if num_speakers >= min_speakers and num_speakers <= max_speakers:
            intervals.append((seg_start, timestamp))
        if is_start is not None:
            num_speakers += 1 if is_start else -1
        seg_start = timestamp

    # Merge consecutive intervals and remove empty intervals.
    merged_intervals = []
    for start, end in intervals:
        if start == end:
            continue
        if merged_intervals and merged_intervals[-1][1] == start:
            merged_intervals[-1] = (merged_intervals[-1][0], end)
        else:
            merged_intervals.append((start, end))

    return [TimeSpan(start, end) for start, end in merged_intervals]


def _cut_into_windows_single(
    cuts: CutSet, duration, hop, keep_excessive_supervisions
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


def _add_recording_path_prefix_single(cut, path):
    return cut.with_recording_path_prefix(path)


def _add_features_path_prefix_single(cut, path):
    return cut.with_features_path_prefix(path)


def _call(obj, member_fn: str, *args, **kwargs) -> Callable:
    return getattr(obj, member_fn)(*args, **kwargs)
