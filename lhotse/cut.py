import itertools
import logging
import random
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial, reduce
from itertools import chain, islice
from math import ceil, floor, isclose
from operator import add
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
from intervaltree import Interval, IntervalTree
from tqdm.auto import tqdm
from typing_extensions import Literal

from lhotse.audio import (
    AudioMixer,
    AudioSource,
    Recording,
    RecordingSet,
    audio_energy,
    null_result_on_audio_loading_error,
)
from lhotse.augmentation import AugmentFn
from lhotse.features import (
    FeatureExtractor,
    FeatureMixer,
    FeatureSet,
    Features,
    create_default_feature_extractor,
)
from lhotse.features.base import compute_global_stats
from lhotse.features.io import FeaturesWriter, LilcomChunkyWriter, LilcomFilesWriter
from lhotse.lazy import AlgorithmMixin
from lhotse.serialization import Serializable
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    DEFAULT_PADDING_VALUE,
    Decibels,
    LOG_EPSILON,
    NonPositiveEnergyError,
    Pathlike,
    Seconds,
    SetContainingAnything,
    TimeSpan,
    add_durations,
    asdict_nonull,
    compute_num_frames,
    compute_num_samples,
    compute_num_windows,
    compute_start_duration_for_extended_cut,
    deprecated,
    exactly_one_not_null,
    fastcopy,
    ifnone,
    index_by_id_and_check,
    measure_overlap,
    overlaps,
    overspans,
    perturb_num_samples,
    rich_exception_info,
    split_manifest_lazy,
    split_sequence,
    uuid4,
)

# One of the design principles for Cuts is a maximally "lazy" implementation, e.g. when mixing Cuts,
# we'd rather sum the feature matrices only after somebody actually calls "load_features". It helps to avoid
# an excessive storage size for data augmented in various ways.


FW = TypeVar("FW", bound=FeaturesWriter)


class Cut:
    """
    .. caution::
        :class:`~lhotse.cut.Cut` is just an abstract class -- the actual logic is implemented by its child classes (scroll down for references).

    :class:`~lhotse.cut.Cut` is a base class for audio cuts.
    An "audio cut" is a subset of a :class:`~lhotse.audio.Recording` -- it can also be thought of as a "view"
    or a pointer to a chunk of audio.
    It is not limited to audio data -- cuts may also point to (sub-spans of) precomputed
    :class:`~lhotse.features.base.Features`.

    Cuts are different from :class:`~lhotse.supervision.SupervisionSegment` in that they may be arbitrarily
    longer or shorter than supervisions; cuts may even contain multiple supervisions for creating contextual
    training data, and unsupervised regions that provide real or synthetic acoustic background context
    for the supervised segments.

    The following example visualizes how a cut may represent a part of a single-channel recording with
    two utterances and some background noise in between::

                          Recording
        |-------------------------------------------|
        "Hey, Matt!"     "Yes?"        "Oh, nothing"
        |----------|     |----|        |-----------|
                   Cut1
        |------------------------|

    This scenario can be represented in code, using :class:`~lhotse.cut.MonoCut`, as::

        >>> from lhotse import Recording, SupervisionSegment, MonoCut
        >>> rec = Recording(id='rec1', duration=10.0, sampling_rate=8000, num_samples=80000, sources=[...])
        >>> sups = [
        ...     SupervisionSegment(id='sup1', recording_id='rec1', start=0, duration=3.37, text='Hey, Matt!'),
        ...     SupervisionSegment(id='sup2', recording_id='rec1', start=4.5, duration=0.9, text='Yes?'),
        ...     SupervisionSegment(id='sup3', recording_id='rec1', start=6.9, duration=2.9, text='Oh, nothing'),
        ... ]
        >>> cut = MonoCut(id='rec1-cut1', start=0.0, duration=6.0, channel=0, recording=rec,
        ...     supervisions=[sups[0], sups[1]])

    .. note::
        All Cut classes assume that the :class:`~lhotse.supervision.SupervisionSegment` time boundaries are relative
        to the beginning of the cut.
        E.g. if the underlying :class:`~lhotse.audio.Recording` starts at 0s (always true), the cut starts at 100s,
        and the SupervisionSegment inside the cut starts at 3s, it really did start at 103rd second of the recording.
        In some cases, the supervision might have a negative start, or a duration exceeding the duration of the cut;
        this means that the supervision in the recording extends beyond the cut.

    Cut allows to check and read audio data or features data::

        >>> assert cut.has_recording
        >>> samples = cut.load_audio()
        >>> if cut.has_features:
        ...     feats = cut.load_features()

    It can be visualized, and listened to, inside Jupyter Notebooks::

        >>> cut.plot_audio()
        >>> cut.play_audio()
        >>> cut.plot_features()

    Cuts can be used with Lhotse's :class:`~lhotse.features.base.FeatureExtractor` to compute features.

        >>> from lhotse import Fbank
        >>> feats = cut.compute_features(extractor=Fbank())

    It is also possible to use a :class:`~lhotse.features.io.FeaturesWriter` to store the features and attach
    their manifest to a copy of the cut::

        >>> from lhotse import LilcomChunkyWriter
        >>> with LilcomChunkyWriter('feats.lca') as storage:
        ...     cut_with_feats = cut.compute_and_store_features(
        ...         extractor=Fbank(),
        ...         storage=storage
        ...     )

    Cuts have several methods that allow their manipulation, transformation, and mixing.
    Some examples (see the respective methods documentation for details)::

        >>> cut_2_to_4s = cut.truncate(offset=2, duration=2)
        >>> cut_padded = cut.pad(duration=10.0)
        >>> cut_extended = cut.extend_by(duration=5.0, direction='both')
        >>> cut_mixed = cut.mix(other_cut, offset_other_by=5.0, snr=20)
        >>> cut_append = cut.append(other_cut)
        >>> cut_24k = cut.resample(24000)
        >>> cut_sp = cut.perturb_speed(1.1)
        >>> cut_vp = cut.perturb_volume(2.)
        >>> cut_rvb = cut.reverb_rir(rir_recording)

    .. note::
        All cut transformations are performed lazily, on-the-fly, upon calling ``load_audio`` or ``load_features``.
        The stored waveforms and features are untouched.

    .. caution::
        Operations on cuts are not mutating -- they return modified copies of :class:`.Cut` objects,
        leaving the original object unmodified.

    A :class:`.Cut` that contains multiple segments (:class:`SupervisionSegment`) can be decayed into
    smaller cuts that correspond directly to supervisions::

        >>> smaller_cuts = cut.trim_to_supervisions()

    Cuts can be detached from parts of their metadata::

        >>> cut_no_feat = cut.drop_features()
        >>> cut_no_rec = cut.drop_recording()
        >>> cut_no_sup = cut.drop_supervisions()

    Finally, cuts provide convenience methods to compute feature frame and audio sample masks for supervised regions::

        >>> sup_frames = cut.supervisions_feature_mask()
        >>> sup_samples = cut.supervisions_audio_mask()

    See also:

        - :class:`lhotse.cut.MonoCut`
        - :class:`lhotse.cut.MixedCut`
        - :class:`lhotse.cut.CutSet`
    """

    # The following is the list of members and properties implemented by the child classes.
    # They are not abstract properties because dataclasses do not work well with the "abc" module.
    id: str
    start: Seconds
    duration: Seconds
    sampling_rate: int
    supervisions: List[SupervisionSegment]
    num_samples: Optional[int]
    num_frames: Optional[int]
    num_features: Optional[int]
    frame_shift: Optional[Seconds]
    features_type: Optional[str]
    has_recording: bool
    has_features: bool

    # The following is the list of methods implemented by the child classes.
    # They are not abstract methods because dataclasses do not work well with the "abc" module.
    # Check a specific child class for their documentation.
    from_dict: Callable[[Dict], "Cut"]
    load_audio: Callable[[], np.ndarray]
    load_features: Callable[[], np.ndarray]
    compute_and_store_features: Callable
    drop_features: Callable
    drop_recording: Callable
    drop_supervisions: Callable
    truncate: Callable
    pad: Callable
    extend_by: Callable
    resample: Callable
    perturb_speed: Callable
    perturb_tempo: Callable
    perturb_volume: Callable
    reverb_rir: Callable
    map_supervisions: Callable
    merge_supervisions: Callable
    filter_supervisions: Callable
    fill_supervision: Callable
    with_features_path_prefix: Callable
    with_recording_path_prefix: Callable

    @property
    def end(self) -> Seconds:
        return add_durations(
            self.start, self.duration, sampling_rate=self.sampling_rate
        )

    def to_dict(self) -> dict:
        d = asdict_nonull(self)
        return {**d, "type": type(self).__name__}

    @property
    def trimmed_supervisions(self) -> List[SupervisionSegment]:
        """
        Return the supervisions in this Cut that have modified time boundaries so as not to exceed
        the Cut's start or end.

        Note that when ``cut.supervisions`` is called, the supervisions may have negative ``start``
        values that indicate the supervision actually begins before the cut, or ``end`` values
        that exceed the Cut's duration (it means the supervision continued in the original recording
        after the Cut's ending).

        .. caution::
            For some tasks such as speech recognition (ASR), trimmed supervisions
            could result in corrupted training data. This is because a part of the transcript
            might actually reside outside of the cut.
        """
        return [s.trim(self.duration) for s in self.supervisions]

    def split(self, timestamp: Seconds) -> Tuple["Cut", "Cut"]:
        """
        Split a cut into two cuts at ``timestamp``, which is measured from the start of the cut.
        For example, a [0s - 10s] cut split at 4s yields:
            - left cut [0s - 4s]
            - right cut [4s - 10s]
        """
        assert 0 < timestamp < self.duration, f"0 < {timestamp} < {self.duration}"
        left = self.truncate(duration=timestamp)
        right = self.truncate(offset=timestamp)
        return left, right

    def mix(
        self,
        other: "Cut",
        offset_other_by: Seconds = 0.0,
        allow_padding: bool = False,
        snr: Optional[Decibels] = None,
        preserve_id: Optional[str] = None,
    ) -> "MixedCut":
        """Refer to :function:`~lhotse.cut.mix` documentation."""
        return mix(
            self,
            other,
            offset=offset_other_by,
            allow_padding=allow_padding,
            snr=snr,
            preserve_id=preserve_id,
        )

    def append(
        self,
        other: "Cut",
        snr: Optional[Decibels] = None,
        preserve_id: Optional[str] = None,
    ) -> "MixedCut":
        """
        Append the ``other`` Cut after the current Cut. Conceptually the same as ``mix`` but with an offset
        matching the current cuts length. Optionally scale down (positive SNR) or scale up (negative SNR)
        the ``other`` cut.
        Returns a MixedCut, which only keeps the information about the mix; actual mixing is performed
        during the call to ``load_features``.

        :param preserve_id: optional string ("left", "right"). When specified, append will preserve the cut ID
            of the left- or right-hand side argument. Otherwise, a new random ID is generated.
        """
        return mix(self, other, offset=self.duration, snr=snr, preserve_id=preserve_id)

    def compute_features(
        self,
        extractor: FeatureExtractor,
        augment_fn: Optional[AugmentFn] = None,
    ) -> np.ndarray:
        """
        Compute the features from this cut. This cut has to be able to load audio.

        :param extractor: a ``FeatureExtractor`` instance used to compute the features.
        :param augment_fn: optional ``WavAugmenter`` instance for audio augmentation.
        :return: a numpy ndarray with the computed features.
        """
        samples = self.load_audio()
        if augment_fn is not None:
            samples = augment_fn(samples, self.sampling_rate)
        return extractor.extract(samples, self.sampling_rate)

    def plot_audio(self):
        """
        Display a plot of the waveform. Requires matplotlib to be installed.
        """
        import matplotlib.pyplot as plt

        samples = self.load_audio().squeeze()
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, self.duration, len(samples)), samples)
        for supervision in self.supervisions:
            supervision = supervision.trim(self.duration)
            ax.axvspan(supervision.start, supervision.end, color="green", alpha=0.1)
        return ax

    def play_audio(self):
        """
        Display a Jupyter widget that allows to listen to the waveform.
        Works only in Jupyter notebook/lab or similar (e.g. Colab).
        """
        from IPython.display import Audio

        samples = self.load_audio().squeeze()
        return Audio(samples, rate=self.sampling_rate)

    def plot_features(self):
        """
        Display the feature matrix as an image. Requires matplotlib to be installed.
        """
        import matplotlib.pyplot as plt

        features = np.flip(self.load_features().transpose(1, 0), 0)
        return plt.matshow(features)

    def plot_alignment(self, alignment_type: str = "word"):
        """
        Display the alignment on top of a spectrogram. Requires matplotlib to be installed.
        """
        import matplotlib.pyplot as plt
        from lhotse import Fbank
        from lhotse.utils import compute_num_frames

        assert (
            len(self.supervisions) == 1
        ), "Cannot plot alignment: there has to be exactly one supervision in a Cut."
        sup = self.supervisions[0]
        assert (
            sup.alignment is not None and alignment_type in sup.alignment
        ), f"Cannot plot alignment: missing alignment field or alignment type '{alignment_type}'"

        fbank = Fbank()

        feats = self.compute_features(fbank)
        speaker = sup.speaker
        language = sup.language

        fig = plt.matshow(np.flip(feats.transpose(1, 0), 0))
        plt.title(
            "Cut ID:" + self.id + ", Speaker:" + speaker + ", Language:" + language
        )
        plt.tick_params(
            axis="both",
            which="major",
            labelbottom=True,
            labeltop=False,
            bottom=True,
            top=False,
        )

        for idx, item in enumerate(sup.alignment[alignment_type]):
            is_even = bool(idx % 2)
            end_frame = compute_num_frames(
                item.end,
                frame_shift=fbank.frame_shift,
                sampling_rate=self.sampling_rate,
            )
            plt.text(
                end_frame - 4,
                70 if is_even else 45,
                item.symbol,
                fontsize=12,
                color="w",
                rotation="vertical",
            )
            plt.axvline(end_frame, color="k")

        plt.show()

    def trim_to_supervisions(
        self,
        keep_overlapping: bool = True,
        min_duration: Optional[Seconds] = None,
        context_direction: Literal["center", "left", "right", "random"] = "center",
    ) -> List["Cut"]:
        """
        Splits the current :class:`.Cut` into as many cuts as there are supervisions (:class:`.SupervisionSegment`).
        These cuts have identical start times and durations as the supervisions.
        When there are overlapping supervisions, they can be kept or discarded via ``keep_overlapping`` flag.

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
        :return: a list of cuts.
        """
        cuts = []
        supervisions_index = self.index_supervisions(index_mixed_tracks=True)
        for segment in self.supervisions:
            if min_duration is None:
                # Cut boundaries are equal to the supervision segment boundaries.
                new_start, new_duration = segment.start, segment.duration
            else:
                # Cut boundaries will be extended with some acoustic context.
                new_start, new_duration = compute_start_duration_for_extended_cut(
                    start=segment.start,
                    duration=segment.duration,
                    new_duration=min_duration,
                    direction=context_direction,
                )
            trimmed = self.truncate(
                offset=new_start,
                duration=new_duration,
                keep_excessive_supervisions=keep_overlapping,
                _supervisions_index=supervisions_index,
            )
            if not keep_overlapping:
                # Ensure that there is exactly one supervision per cut.
                trimmed = trimmed.filter_supervisions(lambda s: s.id == segment.id)
            cuts.append(trimmed)
        return cuts

    def cut_into_windows(
        self,
        duration: Seconds,
        hop: Optional[Seconds] = None,
        keep_excessive_supervisions: bool = True,
    ) -> List["Cut"]:
        """
        Return a list of shorter cuts, made by traversing this cut in windows of
        ``duration`` seconds by ``hop`` seconds.

        The last window might have a shorter duration if there was not enough audio,
        so you might want to use either filter or pad the results.

        :param duration: Desired duration of the new cuts in seconds.
        :param hop: Shift between the windows in the new cuts in seconds.
        :param keep_excessive_supervisions: bool. When a cut is truncated in the
            middle of a supervision segment, should the supervision be kept.
        :return: a list of cuts made from shorter duration windows.
        """
        if not hop:
            hop = duration
        new_cuts = []
        n_windows = compute_num_windows(self.duration, duration, hop)
        for i in range(n_windows):
            new_cuts.append(
                self.truncate(
                    offset=hop * i,
                    duration=duration,
                    keep_excessive_supervisions=keep_excessive_supervisions,
                )
            )
        return new_cuts

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
        keep_ids = ifnone(keep_ids, SetContainingAnything())
        indexed = {
            self.id: IntervalTree(
                Interval(s.start, s.end, s)
                for s in self.supervisions
                if s.id in keep_ids
            )
        }
        if index_mixed_tracks:
            if isinstance(self, MixedCut):
                for track in self.tracks:
                    indexed[track.cut.id] = IntervalTree(
                        Interval(s.start, s.end, s)
                        for s in track.cut.supervisions
                        if s.id in keep_ids
                    )
        return indexed

    @deprecated(
        "Cut.compute_and_store_recording will be removed in a future release. Please use save_audio() instead."
    )
    def compute_and_store_recording(
        self,
        storage_path: Pathlike,
        augment_fn: Optional[AugmentFn] = None,
    ) -> "MonoCut":
        """
        Store this cut's waveform as audio recording to disk.

        :param storage_path: The path to location where we will store the audio recordings.
        :param augment_fn: an optional callable used for audio augmentation.
            Be careful with the types of augmentations used: if they modify
            the start/end/duration times of the cut and its supervisions,
            you will end up with incorrect supervision information when using this API.
            E.g. for speed perturbation, use ``CutSet.perturb_speed()`` instead.
        :return: a new MonoCut instance.
        """
        return self.save_audio(storage_path=storage_path, augment_fn=augment_fn)

    def save_audio(
        self,
        storage_path: Pathlike,
        augment_fn: Optional[AugmentFn] = None,
    ) -> "MonoCut":
        """
        Store this cut's waveform as audio recording to disk.

        :param storage_path: The path to location where we will store the audio recordings.
        :param augment_fn: an optional callable used for audio augmentation.
            Be careful with the types of augmentations used: if they modify
            the start/end/duration times of the cut and its supervisions,
            you will end up with incorrect supervision information when using this API.
            E.g. for speed perturbation, use ``CutSet.perturb_speed()`` instead.
        :return: a new MonoCut instance.
        """
        import torchaudio

        storage_path = Path(storage_path)
        samples = self.load_audio()
        if augment_fn is not None:
            samples = augment_fn(samples, self.sampling_rate)

        torchaudio.save(
            str(storage_path), torch.as_tensor(samples), sample_rate=self.sampling_rate
        )
        recording = Recording(
            id=storage_path.stem,
            sampling_rate=self.sampling_rate,
            num_samples=samples.shape[1],
            duration=samples.shape[1] / self.sampling_rate,
            sources=[
                AudioSource(
                    type="file",
                    channels=[0],
                    source=str(storage_path),
                )
            ],
        )
        return MonoCut(
            id=self.id,
            start=0,
            duration=recording.duration,
            channel=0,
            supervisions=self.supervisions,
            recording=recording,
            custom=self.custom if hasattr(self, "custom") else None,
        )

    def speakers_feature_mask(
        self,
        min_speaker_dim: Optional[int] = None,
        speaker_to_idx_map: Optional[Dict[str, int]] = None,
        use_alignment_if_exists: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return a matrix of per-speaker activity in a cut. The matrix shape is (num_speakers, num_frames),
        and its values are 0 for nonspeech **frames** and 1 for speech **frames** for each respective speaker.

        This is somewhat inspired by the TS-VAD setup: https://arxiv.org/abs/2005.07272

        :param min_speaker_dim: optional int, when specified it will enforce that the matrix shape is at least
            that value (useful for datasets like CHiME 6 where the number of speakers is always 4, but some cuts
            might have less speakers than that).
        :param speaker_to_idx_map: optional dict mapping speaker names (strings) to their global indices (ints).
            Useful when you want to preserve the order of the speakers (e.g. speaker XYZ is always mapped to index 2)
        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        assert self.has_features, (
            f"No features available. "
            f"Can't compute supervisions feature mask for cut with ID: {self.id}."
        )
        if speaker_to_idx_map is None:
            speaker_to_idx_map = {
                spk: idx
                for idx, spk in enumerate(
                    sorted(set(s.speaker for s in self.supervisions))
                )
            }
        num_speakers = len(speaker_to_idx_map)
        if min_speaker_dim is not None:
            num_speakers = min(min_speaker_dim, num_speakers)
        mask = np.zeros((num_speakers, self.num_frames))
        for supervision in self.supervisions:
            speaker_idx = speaker_to_idx_map[supervision.speaker]
            if (
                use_alignment_if_exists
                and supervision.alignment
                and use_alignment_if_exists in supervision.alignment
            ):
                for ali in supervision.alignment[use_alignment_if_exists]:
                    st = round(ali.start / self.frame_shift) if ali.start > 0 else 0
                    et = (
                        round(ali.end / self.frame_shift)
                        if ali.end < self.duration
                        else self.num_frames
                    )
                    mask[speaker_idx, st:et] = 1
            else:
                st = (
                    round(supervision.start / self.frame_shift)
                    if supervision.start > 0
                    else 0
                )
                et = (
                    round(supervision.end / self.frame_shift)
                    if supervision.end < self.duration
                    else self.num_frames
                )
                mask[speaker_idx, st:et] = 1
        return mask

    def speakers_audio_mask(
        self,
        min_speaker_dim: Optional[int] = None,
        speaker_to_idx_map: Optional[Dict[str, int]] = None,
        use_alignment_if_exists: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return a matrix of per-speaker activity in a cut. The matrix shape is (num_speakers, num_samples),
        and its values are 0 for nonspeech **samples** and 1 for speech **samples** for each respective speaker.

        This is somewhat inspired by the TS-VAD setup: https://arxiv.org/abs/2005.07272

        :param min_speaker_dim: optional int, when specified it will enforce that the matrix shape is at least
            that value (useful for datasets like CHiME 6 where the number of speakers is always 4, but some cuts
            might have less speakers than that).
        :param speaker_to_idx_map: optional dict mapping speaker names (strings) to their global indices (ints).
            Useful when you want to preserve the order of the speakers (e.g. speaker XYZ is always mapped to index 2)
        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        assert self.has_recording, (
            f"No recording available. "
            f"Can't compute supervisions audio mask for cut with ID: {self.id}."
        )
        if speaker_to_idx_map is None:
            speaker_to_idx_map = {
                spk: idx
                for idx, spk in enumerate(
                    sorted(set(s.speaker for s in self.supervisions))
                )
            }
        num_speakers = len(speaker_to_idx_map)
        if min_speaker_dim is not None:
            num_speakers = min(min_speaker_dim, num_speakers)
        mask = np.zeros((num_speakers, self.num_samples))
        for supervision in self.supervisions:
            speaker_idx = speaker_to_idx_map[supervision.speaker]
            if (
                use_alignment_if_exists
                and supervision.alignment
                and use_alignment_if_exists in supervision.alignment
            ):
                for ali in supervision.alignment[use_alignment_if_exists]:
                    st = (
                        compute_num_samples(ali.start, self.sampling_rate)
                        if ali.start > 0
                        else 0
                    )
                    et = (
                        compute_num_samples(ali.end, self.sampling_rate)
                        if ali.end < self.duration
                        else compute_num_samples(self.duration, self.sampling_rate)
                    )
                    mask[speaker_idx, st:et] = 1
            else:
                st = (
                    compute_num_samples(supervision.start, self.sampling_rate)
                    if supervision.start > 0
                    else 0
                )
                et = (
                    compute_num_samples(supervision.end, self.sampling_rate)
                    if supervision.end < self.duration
                    else compute_num_samples(self.duration, self.sampling_rate)
                )
                mask[speaker_idx, st:et] = 1
        return mask

    def supervisions_feature_mask(
        self, use_alignment_if_exists: Optional[str] = None
    ) -> np.ndarray:
        """
        Return a 1D numpy array with value 1 for **frames** covered by at least one supervision,
        and 0 for **frames** not covered by any supervision.

        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        return compute_supervisions_frame_mask(
            self, use_alignment_if_exists=use_alignment_if_exists
        )

    def supervisions_audio_mask(
        self, use_alignment_if_exists: Optional[str] = None
    ) -> np.ndarray:
        """
        Return a 1D numpy array with value 1 for **samples** covered by at least one supervision,
        and 0 for **samples** not covered by any supervision.

        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        assert self.has_recording, (
            f"No recording available. "
            f"Can't compute supervisions audio mask for cut with ID: {self.id}."
        )
        mask = np.zeros(self.num_samples, dtype=np.float32)
        for supervision in self.supervisions:
            if (
                use_alignment_if_exists
                and supervision.alignment
                and use_alignment_if_exists in supervision.alignment
            ):
                for ali in supervision.alignment[use_alignment_if_exists]:
                    st = round(ali.start * self.sampling_rate) if ali.start > 0 else 0
                    et = (
                        round(ali.end * self.sampling_rate)
                        if ali.end < self.duration
                        else self.duration * self.sampling_rate
                    )
                    mask[st:et] = 1.0
            else:
                st = (
                    round(supervision.start * self.sampling_rate)
                    if supervision.start > 0
                    else 0
                )
                et = (
                    round(supervision.end * self.sampling_rate)
                    if supervision.end < self.duration
                    else self.duration * self.sampling_rate
                )
                mask[st:et] = 1.0
        return mask

    def with_id(self, id_: str) -> "Cut":
        """Return a copy of the Cut with a new ID."""
        return fastcopy(self, id=id_)


@dataclass
class MonoCut(Cut):
    """
    :class:`~lhotse.cut.MonoCut` is a :class:`~lhotse.cut.Cut` of a single channel of
    a :class:`~lhotse.audio.Recording`. In addition to Cut, it has a specified channel attribute. This is the most commonly used type of cut.

    Please refer to the documentation of :class:`~lhotse.cut.Cut` to learn more about using cuts.

    See also:

        - :class:`lhotse.cut.Cut`
        - :class:`lhotse.cut.MixedCut`
        - :class:`lhotse.cut.CutSet`
    """

    id: str

    # Begin and duration are needed to specify which chunk of features/recording to load.
    start: Seconds
    duration: Seconds
    channel: int

    # Supervisions that will be used as targets for model training later on. They don't have to cover the whole
    # cut duration. They also might overlap.
    supervisions: List[SupervisionSegment] = field(default_factory=list)

    # The features can span longer than the actual cut - the Features object "knows" its start and end time
    # within the underlying recording. We can expect the interval [begin, begin + duration] to be a subset of the
    # interval represented in features.
    features: Optional[Features] = None

    # For the cases that the model was trained by raw audio instead of features
    recording: Optional[Recording] = None

    # Store anything else the user might want.
    custom: Optional[Dict[str, Any]] = None

    def __setattr__(self, key: str, value: Any):
        """
        This magic function is called when the user tries to set an attribute.
        We use it as syntactic sugar to store custom attributes in ``self.custom``
        field, so that they can be (de)serialized later.
        """
        if key in self.__dataclass_fields__:
            super().__setattr__(key, value)
        else:
            custom = ifnone(self.custom, {})
            custom[key] = value
            self.custom = custom

    def __getattr__(self, name: str) -> Any:
        """
        This magic function is called when the user tries to access an attribute
        of :class:`.MonoCut` that doesn't exist. It is used for accessing the custom
        attributes of cuts.

        We use it to look up the ``custom`` field: when it's None or empty,
        we'll just raise AttributeError as usual.
        If ``item`` is found in ``custom``, we'll return ``custom[item]``.
        If ``item`` starts with "load_", we'll assume the name of the relevant
        attribute comes after that, and that value of that field is of type
        :class:`~lhotse.array.Array` or :class:`~lhotse.array.TemporalArray`.
        We'll return its ``load`` method to call by the user.

        Example of attaching and reading an alignment as TemporalArray::

            >>> cut = MonoCut('cut1', start=0, duration=4, channel=0)
            >>> cut.alignment = TemporalArray(...)
            >>> ali = cut.load_alignment()

        """
        custom = self.custom
        if custom is None:
            raise AttributeError(f"No such attribute: {name}")
        if name in custom:
            # Somebody accesses raw [Temporal]Array manifest
            # or wrote a custom piece of metadata into MonoCut.
            return self.custom[name]
        elif name.startswith("load_"):
            # Return the method for loading [Temporal]Arrays,
            # to be invoked by the user.
            attr_name = name[5:]
            return partial(self.load_custom, attr_name)
        raise AttributeError(f"No such attribute: {name}")

    def load_custom(self, name: str) -> np.ndarray:
        """
        Load custom data as numpy array. The custom data is expected to have
        been stored in cuts ``custom`` field as an :class:`~lhotse.array.Array` or
        :class:`~lhotse.array.TemporalArray` manifest.

        .. note:: It works with Array manifests stored via attribute assignments,
            e.g.: ``cut.my_custom_data = Array(...)``.

        :param name: name of the custom attribute.
        :return: a numpy array with the data.
        """
        from lhotse.array import Array, TemporalArray

        value = self.custom.get(name)
        if isinstance(value, Array):
            # Array does not support slicing.
            return value.load()
        elif isinstance(value, TemporalArray):
            # TemporalArray supports slicing.
            return value.load(start=self.start, duration=self.duration)
        elif isinstance(value, Recording):
            # Recording supports slicing.
            return value.load_audio(
                channels=self.channel, offset=self.start, duration=self.duration
            )
        else:
            raise ValueError(
                f"To load {name}, the cut needs to have field {name} (or cut.custom['{name}']) "
                f"defined, and its value has to be a manifest of type Array or TemporalArray."
            )

    @property
    def recording_id(self) -> str:
        return self.recording.id if self.has_recording else self.features.recording_id

    @property
    def has_features(self) -> bool:
        return self.features is not None

    @property
    def has_recording(self) -> bool:
        return self.recording is not None

    @property
    def frame_shift(self) -> Optional[Seconds]:
        return self.features.frame_shift if self.has_features else None

    @property
    def num_frames(self) -> Optional[int]:
        return (
            compute_num_frames(
                duration=self.duration,
                frame_shift=self.frame_shift,
                sampling_rate=self.sampling_rate,
            )
            if self.has_features
            else None
        )

    @property
    def num_samples(self) -> Optional[int]:
        return (
            compute_num_samples(self.duration, self.sampling_rate)
            if self.has_recording
            else None
        )

    @property
    def num_features(self) -> Optional[int]:
        return self.features.num_features if self.has_features else None

    @property
    def features_type(self) -> Optional[str]:
        return self.features.type if self.has_features else None

    @property
    def sampling_rate(self) -> int:
        return (
            self.features.sampling_rate
            if self.has_features
            else self.recording.sampling_rate
        )

    @rich_exception_info
    def load_features(self) -> Optional[np.ndarray]:
        """
        Load the features from the underlying storage and cut them to the relevant
        [begin, duration] region of the current MonoCut.
        """
        if self.has_features:
            feats = self.features.load(start=self.start, duration=self.duration)
            # Note: we forgive off-by-one errors in the feature matrix frames
            #       due to various hard-to-predict floating point arithmetic issues.
            #       If needed, we will remove or duplicate the last frame to be
            #       consistent with the manifests declared "num_frames".
            if feats.shape[0] - self.num_frames == 1:
                feats = feats[: self.num_frames, :]
            elif feats.shape[0] - self.num_frames == -1:
                feats = np.concatenate((feats, feats[-1:, :]), axis=0)
            return feats
        return None

    @rich_exception_info
    def load_audio(self) -> Optional[np.ndarray]:
        """
        Load the audio by locating the appropriate recording in the supplied RecordingSet.
        The audio is trimmed to the [begin, end] range specified by the MonoCut.

        :return: a numpy ndarray with audio samples, with shape (1 <channel>, N <samples>)
        """
        if self.has_recording:
            return self.recording.load_audio(
                channels=self.channel,
                offset=self.start,
                duration=self.duration,
            )
        return None

    def move_to_memory(
        self,
        audio_format: str = "flac",
        load_audio: bool = True,
        load_features: bool = True,
        load_custom: bool = True,
    ) -> "MonoCut":
        """
        Load data (audio, features, or custom arrays) into memory and attach them
        to a copy of the manifest. This is useful when you want to store cuts together
        with the actual data in some binary format that enables sequential data reads.

        Audio is encoded with ``audio_format`` (compatible with ``torchaudio.save``),
        floating point features are encoded with lilcom, and other arrays are pickled.
        """

        # Handle moving audio to memory.
        if not load_audio or not self.has_recording:
            recording = self.recording
        else:
            recording = self.recording.move_to_memory(
                channels=self.channel,
                offset=self.start,
                duration=self.duration,
                format=audio_format,
            )

        # Handle moving features to memory.
        if not load_features or not self.has_features:
            features = self.features
        else:
            features = self.features.move_to_memory(
                start=self.start, duration=self.duration
            )

        # Handle moving custom arrays to memory.
        if not load_custom or self.custom is None:
            custom = self.custom
        else:
            from lhotse.array import Array, TemporalArray

            custom = {
                # Case 1: Array
                k: v.move_to_memory() if isinstance(v, Array)
                # Case 2: TemporalArray
                else v.move_to_memory(start=self.start, duration=self.duration)
                if isinstance(v, TemporalArray)
                # Case 3: anything else
                else v
                for k, v in self.custom.items()
            }

        cut = fastcopy(
            self,
            # note: cut's start is relative to the start of the recording/features;
            # since we moved to memory only a subset of recording/features that
            # corresponds to this cut, the start is always 0.
            start=0.0,
            recording=recording,
            features=features,
            custom=custom,
        )
        return cut

    def drop_features(self) -> "MonoCut":
        """Return a copy of the current :class:`.MonoCut`, detached from ``features``."""
        assert (
            self.has_recording
        ), f"Cannot detach features from a MonoCut with no Recording (cut ID = {self.id})."
        return fastcopy(self, features=None)

    def drop_recording(self) -> "MonoCut":
        """Return a copy of the current :class:`.MonoCut`, detached from ``recording``."""
        assert (
            self.has_features
        ), f"Cannot detach recording from a MonoCut with no Features (cut ID = {self.id})."
        return fastcopy(self, recording=None)

    def drop_supervisions(self) -> "MonoCut":
        """Return a copy of the current :class:`.MonoCut`, detached from ``supervisions``."""
        return fastcopy(self, supervisions=[])

    def fill_supervision(
        self, add_empty: bool = True, shrink_ok: bool = False
    ) -> "MonoCut":
        """
        Fills the whole duration of a cut with a supervision segment.

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
        if len(self.supervisions) == 0:
            if not add_empty:
                return self
            sups = [
                SupervisionSegment(
                    id=self.id,
                    recording_id=self.recording_id,
                    start=0,
                    duration=self.duration,
                    channel=self.channel,
                )
            ]
        else:
            assert (
                len(self.supervisions) == 1
            ), f"Cannot expand more than one supervision (found {len(self.supervisions)}."
            old_sup = self.supervisions[0]
            if isclose(old_sup.start, 0) and isclose(old_sup.duration, self.duration):
                return self
            if old_sup.start < 0 or old_sup.end > self.end and not shrink_ok:
                raise ValueError(
                    f"Cannot shrink supervision (start={old_sup.start}, end={old_sup.end}) to cut "
                    f"(start=0, duration={self.duration}) because the argument `shrink_ok` is `False`. "
                    f"Note: this check prevents accidental data loss for speech recognition, "
                    f"as supervision exceeding a cut indicates there might be some spoken content "
                    f"beyond cuts start or end (an ASR model would be trained to predict more text than "
                    f"spoken in the audio). If this is okay, set `shrink_ok` to `True`."
                )
            sups = [fastcopy(old_sup, start=0, duration=self.duration)]

        return fastcopy(self, supervisions=sups)

    def compute_and_store_features(
        self,
        extractor: FeatureExtractor,
        storage: FeaturesWriter,
        augment_fn: Optional[AugmentFn] = None,
        *args,
        **kwargs,
    ) -> Cut:
        """
        Compute the features from this cut, store them on disk, and attach a feature manifest to this cut.
        This cut has to be able to load audio.

        :param extractor: a ``FeatureExtractor`` instance used to compute the features.
        :param storage: a ``FeaturesWriter`` instance used to write the features to a storage.
        :param augment_fn: an optional callable used for audio augmentation.
        :return: a new ``MonoCut`` instance with a ``Features`` manifest attached to it.
        """
        features_info = extractor.extract_from_samples_and_store(
            samples=self.load_audio(),
            storage=storage,
            sampling_rate=self.sampling_rate,
            offset=self.start,
            channel=self.channel,
            augment_fn=augment_fn,
        )
        # The fastest way to instantiate a copy of the cut with a Features object attached
        return fastcopy(self, features=features_info)

    def truncate(
        self,
        *,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        keep_excessive_supervisions: bool = True,
        preserve_id: bool = False,
        _supervisions_index: Optional[Dict[str, IntervalTree]] = None,
    ) -> "MonoCut":
        """
        Returns a new MonoCut that is a sub-region of the current MonoCut.

        Note that no operation is done on the actual features or recording -
        it's only during the call to :meth:`MonoCut.load_features` / :meth:`MonoCut.load_audio`
        when the actual changes happen (a subset of features/audio is loaded).

        .. hint::

            To extend a cut by a fixed duration, use the :meth:`MonoCut.extend_by` method.

        :param offset: float (seconds), controls the start of the new cut relative to the current MonoCut's start.
            E.g., if the current MonoCut starts at 10.0, and offset is 2.0, the new start is 12.0.
        :param duration: optional float (seconds), controls the duration of the resulting MonoCut.
            By default, the duration is (end of the cut before truncation) - (offset).
        :param keep_excessive_supervisions: bool. Since trimming may happen inside a SupervisionSegment,
            the caller has an option to either keep or discard such supervisions.
        :param preserve_id: bool. Should the truncated cut keep the same ID or get a new, random one.
        :param _supervisions_index: an IntervalTree; when passed, allows to speed up processing of Cuts with a very
            large number of supervisions. Intended as an internal parameter.
        :return: a new MonoCut instance. If the current MonoCut is shorter than the duration, return None.
        """
        assert (
            offset >= 0
        ), f"Offset for truncate must be non-negative (provided {offset})."
        new_start = max(
            add_durations(self.start, offset, sampling_rate=self.sampling_rate), 0
        )
        until = add_durations(
            offset,
            duration if duration is not None else self.duration,
            sampling_rate=self.sampling_rate,
        )
        new_duration = add_durations(until, -offset, sampling_rate=self.sampling_rate)
        assert new_duration > 0.0, f"new_duration={new_duration}"
        # duration_past_end = (new_start + new_duration) - (self.start + self.duration)
        duration_past_end = add_durations(
            new_start,
            new_duration,
            -self.start,
            -self.duration,
            sampling_rate=self.sampling_rate,
        )
        if duration_past_end > 0:
            # When the end of the MonoCut has been exceeded, trim the new duration to not exceed the old MonoCut's end.
            new_duration = add_durations(
                new_duration, -duration_past_end, sampling_rate=self.sampling_rate
            )

        if _supervisions_index is None:
            criterion = overlaps if keep_excessive_supervisions else overspans
            new_time_span = TimeSpan(start=0, end=new_duration)
            new_supervisions = (
                segment.with_offset(-offset) for segment in self.supervisions
            )
            supervisions = [
                segment
                for segment in new_supervisions
                if criterion(new_time_span, segment)
            ]
        else:
            tree = _supervisions_index[self.id]
            # Below we select which method should be called on the IntervalTree object.
            # The result of calling that method with a range of (begin, end) is an iterable
            # of Intervals that contain the SupervisionSegments matching our criterion.
            # We call "interval.data" to obtain the underlying SupervisionSegment.
            # Additionally, when the method is tree.envelop, we use a small epsilon to
            # extend the searched boundaries to account for possible float arithmetic errors.
            if keep_excessive_supervisions:
                intervals = tree.overlap(begin=offset, end=offset + new_duration)
            else:
                intervals = tree.envelop(
                    begin=offset - 1e-3, end=offset + new_duration + 1e-3
                )
            supervisions = []
            for interval in intervals:
                # We are going to measure the overlap ratio of the supervision with the "truncated" cut
                # and reject segments that overlap less than 1%. This way we can avoid quirks and errors
                # of limited float precision.
                olap_ratio = measure_overlap(
                    interval.data, TimeSpan(offset, offset + new_duration)
                )
                if olap_ratio > 0.01:
                    supervisions.append(interval.data.with_offset(-offset))

        return fastcopy(
            self,
            id=self.id if preserve_id else str(uuid4()),
            start=new_start,
            duration=new_duration,
            supervisions=sorted(supervisions, key=lambda s: s.start),
        )

    def extend_by(
        self,
        *,
        duration: Seconds,
        direction: str = "both",
        preserve_id: bool = False,
    ) -> "MonoCut":
        """
        Returns a new MonoCut that is an extended region of the current MonoCut by extending
        the cut by a fixed duration in the specified direction.

        Note that no operation is done on the actual features or recording -
        it's only during the call to :meth:`MonoCut.load_features` / :meth:`MonoCut.load_audio`
        when the actual changes happen (an extended version of features/audio is loaded).

        .. hint::

            This method extends a cut by a given duration, either to the left or to the right (or both), using
            the "real" content of the recording that the cut is part of. For example, a MonoCut spanning
            the region from 2s to 5s in a recording, when extended by 2s to the right, will now span
            the region from 2s to 7s in the same recording (provided the recording length exceeds 7s).
            If the recording is shorter, the cut will only be extended up to the duration of the recording.
            To "expand" a cut by padding, use :meth:`MonoCut.pad`. To "truncate" a cut, use :meth:`MonoCut.truncate`.

        .. hint::

            If `direction` is "both", the resulting cut will be extended by the specified duration in
            both directions. This is different from the usage in :meth:`MonoCut.pad` where a padding
            equal to 0.5*duration is added to both sides.

        :param duration: float (seconds), specifies the duration by which the cut should be extended.
        :param direction: string, 'left', 'right' or 'both'. Determines whether to extend on the left,
            right, or both sides. If 'both', extend on both sides by the duration specified in `duration`.
        :param preserve_id: bool. Should the extended cut keep the same ID or get a new, random one.
        :return: a new MonoCut instance.
        """
        from lhotse.array import TemporalArray

        assert duration >= 0, f"Duration must be non-negative (provided {duration})."

        new_start, new_end = self.start, self.end
        if direction == "left" or direction == "both":
            new_start = max(self.start - duration, 0)
        if direction == "right" or direction == "both":
            new_end = min(self.end + duration, self.recording.duration)

        new_duration = add_durations(
            new_end, -new_start, sampling_rate=self.sampling_rate
        )

        new_supervisions = (
            segment.with_offset(
                add_durations(self.start, -new_start, sampling_rate=self.sampling_rate)
            )
            for segment in self.supervisions
        )

        def _this_exceeds_duration(attribute: Union[Features, TemporalArray]) -> bool:
            # We compare in terms of frames, not seconds, to avoid rounding errors.
            # We also allow a tolerance of 1 frame on either side.
            new_start_frames = compute_num_frames(
                new_start, attribute.frame_shift, self.sampling_rate
            )
            new_end_frames = compute_num_frames(
                new_end, attribute.frame_shift, self.sampling_rate
            )
            attribute_start = compute_num_frames(
                attribute.start, attribute.frame_shift, self.sampling_rate
            )
            attribute_end = attribute_start + attribute.num_frames
            return (new_start_frames < attribute_start - 1) or (
                new_end_frames > attribute_end + 1
            )

        feature_kwargs = {}
        if self.has_features:
            if _this_exceeds_duration(self.features):
                logging.warning(
                    "Attempting to extend a MonoCut that exceeds the range of pre-computed features. "
                    "The feature manifest will be detached."
                )
                feature_kwargs["features"] = None

        custom_kwargs = {}
        if self.custom is not None:
            for name, array in self.custom.items():
                custom_kwargs[name] = array
                if isinstance(array, TemporalArray):
                    if _this_exceeds_duration(array):
                        logging.warning(
                            f"Attempting to extend a MonoCut that exceeds the range of pre-computed custom data '{name}'. "
                            "The custom data will be detached."
                        )
                        custom_kwargs[name] = None

        return fastcopy(
            self,
            id=self.id if preserve_id else str(uuid4()),
            start=new_start,
            duration=new_duration,
            supervisions=sorted(new_supervisions, key=lambda s: s.start),
            **feature_kwargs,
            custom=custom_kwargs,
        )

    def pad(
        self,
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
        return pad(
            self,
            duration=duration,
            num_frames=num_frames,
            num_samples=num_samples,
            pad_feat_value=pad_feat_value,
            direction=direction,
            preserve_id=preserve_id,
            pad_value_dict=pad_value_dict,
        )

    def resample(self, sampling_rate: int, affix_id: bool = False) -> "MonoCut":
        """
        Return a new ``MonoCut`` that will lazily resample the audio while reading it.
        This operation will drop the feature manifest, if attached.
        It does not affect the supervision.

        :param sampling_rate: The new sampling rate.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the current ``MonoCut``.
        """
        assert self.has_recording, "Cannot resample a MonoCut without Recording."
        return fastcopy(
            self,
            id=f"{self.id}_rs{sampling_rate}" if affix_id else self.id,
            recording=self.recording.resample(sampling_rate),
            features=None,
        )

    def perturb_speed(self, factor: float, affix_id: bool = True) -> "MonoCut":
        """
        Return a new ``MonoCut`` that will lazily perturb the speed while loading audio.
        The ``num_samples``, ``start`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.
        We are also updating the time markers of the underlying ``Recording`` and the supervisions.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``MonoCut.id`` field
            by affixing it with "_sp{factor}".
        :return: a modified copy of the current ``MonoCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb speed on a MonoCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb speed on a MonoCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "speed perturbation."
            )
            self.features = None
        # Actual audio perturbation.
        recording_sp = self.recording.perturb_speed(factor=factor, affix_id=affix_id)
        # Match the supervision's start and duration to the perturbed audio.
        # Since SupervisionSegment "start" is relative to the MonoCut's, it's okay (and necessary)
        # to perturb it as well.
        supervisions_sp = [
            s.perturb_speed(
                factor=factor, sampling_rate=self.sampling_rate, affix_id=affix_id
            )
            for s in self.supervisions
        ]
        # New start and duration have to be computed through num_samples to be accurate
        start_samples = perturb_num_samples(
            compute_num_samples(self.start, self.sampling_rate), factor
        )
        new_start = start_samples / self.sampling_rate
        new_num_samples = perturb_num_samples(self.num_samples, factor)
        new_duration = new_num_samples / self.sampling_rate
        return fastcopy(
            self,
            id=f"{self.id}_sp{factor}" if affix_id else self.id,
            recording=recording_sp,
            supervisions=supervisions_sp,
            duration=new_duration,
            start=new_start,
        )

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> "MonoCut":
        """
        Return a new ``MonoCut`` that will lazily perturb the tempo while loading audio.

        Compared to speed perturbation, tempo preserves pitch.
        The ``num_samples``, ``start`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.
        We are also updating the time markers of the underlying ``Recording`` and the supervisions.

        :param factor: The tempo will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``MonoCut.id`` field
            by affixing it with "_tp{factor}".
        :return: a modified copy of the current ``MonoCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb speed on a MonoCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb tempo on a MonoCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "speed perturbation."
            )
            self.features = None
        # Actual audio perturbation.
        recording_sp = self.recording.perturb_tempo(factor=factor, affix_id=affix_id)
        # Match the supervision's start and duration to the perturbed audio.
        # Since SupervisionSegment "start" is relative to the MonoCut's, it's okay (and necessary)
        # to perturb it as well.
        supervisions_sp = [
            s.perturb_tempo(
                factor=factor, sampling_rate=self.sampling_rate, affix_id=affix_id
            )
            for s in self.supervisions
        ]
        # New start and duration have to be computed through num_samples to be accurate
        start_samples = perturb_num_samples(
            compute_num_samples(self.start, self.sampling_rate), factor
        )
        new_start = start_samples / self.sampling_rate
        new_num_samples = perturb_num_samples(self.num_samples, factor)
        new_duration = new_num_samples / self.sampling_rate
        return fastcopy(
            self,
            id=f"{self.id}_tp{factor}" if affix_id else self.id,
            recording=recording_sp,
            supervisions=supervisions_sp,
            duration=new_duration,
            start=new_start,
        )

    def perturb_volume(self, factor: float, affix_id: bool = True) -> "MonoCut":
        """
        Return a new ``MonoCut`` that will lazily perturb the volume while loading audio.

        :param factor: The volume will be adjusted this many times (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``MonoCut.id`` field
            by affixing it with "_vp{factor}".
        :return: a modified copy of the current ``MonoCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb volume on a MonoCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb volume on a MonoCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "volume perturbation."
            )
            self.features = None
        # Actual audio perturbation.
        recording_vp = self.recording.perturb_volume(factor=factor, affix_id=affix_id)
        # Match the supervision's id (and it's underlying recording id).
        supervisions_vp = [
            s.perturb_volume(factor=factor, affix_id=affix_id)
            for s in self.supervisions
        ]

        return fastcopy(
            self,
            id=f"{self.id}_vp{factor}" if affix_id else self.id,
            recording=recording_vp,
            supervisions=supervisions_vp,
        )

    def reverb_rir(
        self,
        rir_recording: "Recording",
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: List[int] = [0],
    ) -> Union["MonoCut", "MixedCut"]:
        """
        Return a new ``MonoCut`` that will convolve the audio with the provided impulse response.
        If the `rir_recording` is multi-channel, the `rir_channels` argument determines which channels
        will be used. By default, we use the first channel and return a MonoCut.

        :param rir_recording: The impulse response to use for convolving.
        :param normalize_output: When true, output will be normalized to have energy as input.
        :param early_only: When true, only the early reflections (first 50 ms) will be used.
        :param affix_id: When true, we will modify the ``MonoCut.id`` field
            by affixing it with "_rvb".
        :param rir_channels: The channels of the impulse response to use. First channel is used by default.
            If multiple channels are specified, this will produce a MixedCut instead of a MonoCut.
        :return: a modified copy of the current ``MonoCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot apply reverberation on a MonoCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to reverberate a MonoCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "reverberation."
            )
            self.features = None

        assert all(
            c < rir_recording.num_channels for c in rir_channels
        ), "Invalid channel index in `rir_channels`."
        if rir_recording.num_channels == 1 or len(rir_channels) == 1:
            # reverberation will return a MonoCut
            recording_rvb = self.recording.reverb_rir(
                rir_recording=rir_recording,
                normalize_output=normalize_output,
                early_only=early_only,
                affix_id=affix_id,
                rir_channels=rir_channels,
            )
            # Match the supervision's id (and it's underlying recording id).
            supervisions_rvb = [
                s.reverb_rir(
                    affix_id=affix_id,
                )
                for s in self.supervisions
            ]

            return fastcopy(
                self,
                id=f"{self.id}_rvb" if affix_id else self.id,
                recording=recording_rvb,
                supervisions=supervisions_rvb,
            )
        else:
            # we will return a MixedCut where each track represents the MonoCut convolved
            # with a single channel of the RIR
            new_tracks = [
                MixTrack(
                    cut=fastcopy(
                        self,
                        recording=self.recording.reverb_rir(
                            rir_recording=rir_recording,
                            normalize_output=normalize_output,
                            early_only=early_only,
                            affix_id=affix_id,
                            rir_channels=[channel],
                        ),
                        supervisions=[
                            s.reverb_rir(
                                affix_id=affix_id,
                            )
                            for s in self.supervisions
                        ],
                    ),
                    offset=0,
                )
                for channel in rir_channels
            ]
            return MixedCut(
                id=f"{self.id}_rvb" if affix_id else self.id, tracks=new_tracks
            )

    def map_supervisions(
        self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]
    ) -> "MonoCut":
        """
        Return a copy of the cut that has its supervisions transformed by ``transform_fn``.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a modified MonoCut.
        """
        new_cut = fastcopy(
            self, supervisions=[s.map(transform_fn) for s in self.supervisions]
        )
        return new_cut

    def filter_supervisions(
        self, predicate: Callable[[SupervisionSegment], bool]
    ) -> "MonoCut":
        """
        Return a copy of the cut that only has supervisions accepted by ``predicate``.

        Example::

            >>> cut = cut.filter_supervisions(lambda s: s.id in supervision_ids)
            >>> cut = cut.filter_supervisions(lambda s: s.duration < 5.0)
            >>> cut = cut.filter_supervisions(lambda s: s.text is not None)

        :param predicate: A callable that accepts `SupervisionSegment` and returns bool
        :return: a modified MonoCut
        """
        new_cut = fastcopy(
            self, supervisions=[s for s in self.supervisions if predicate(s)]
        )
        return new_cut

    def merge_supervisions(
        self, custom_merge_fn: Optional[Callable[[str, Iterable[Any]], Any]] = None
    ) -> "MonoCut":
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
        # "m" stands for merged in variable names below
        return merge_supervisions(self, custom_merge_fn=custom_merge_fn)

    @staticmethod
    def from_dict(data: dict) -> "MonoCut":
        from lhotse.serialization import deserialize_custom_field

        # Remove "type" field if exists.
        data.pop("type", None)

        features = (
            Features.from_dict(data.pop("features")) if "features" in data else None
        )
        recording = (
            Recording.from_dict(data.pop("recording")) if "recording" in data else None
        )
        supervision_infos = data.pop("supervisions") if "supervisions" in data else []

        if "custom" in data:
            deserialize_custom_field(data["custom"])

        if "type" in data:
            data.pop("type")

        return MonoCut(
            **data,
            features=features,
            recording=recording,
            supervisions=[SupervisionSegment.from_dict(s) for s in supervision_infos],
        )

    def with_features_path_prefix(self, path: Pathlike) -> "MonoCut":
        if not self.has_features:
            return self
        return fastcopy(self, features=self.features.with_path_prefix(path))

    def with_recording_path_prefix(self, path: Pathlike) -> "MonoCut":
        if not self.has_recording:
            return self
        return fastcopy(self, recording=self.recording.with_path_prefix(path))


@dataclass
class PaddingCut(Cut):
    """
    :class:`~lhotse.cut.PaddingCut` is a dummy :class:`~lhotse.cut.Cut` that doesn't refer to
    actual recordings or features --it simply returns zero samples in the time domain
    and a specified features value in the feature domain.
    Its main role is to be appended to other cuts to make them evenly sized.

    Please refer to the documentation of :class:`~lhotse.cut.Cut` to learn more about using cuts.

    See also:

        - :class:`lhotse.cut.Cut`
        - :class:`lhotse.cut.MonoCut`
        - :class:`lhotse.cut.MixedCut`
        - :class:`lhotse.cut.CutSet`
    """

    id: str
    duration: Seconds
    sampling_rate: int
    feat_value: float

    # For frequency domain
    num_frames: Optional[int] = None
    num_features: Optional[int] = None
    frame_shift: Optional[float] = None

    # For time domain
    num_samples: Optional[int] = None

    # Dict for storing padding values for custom array attributes
    custom: Optional[dict] = None

    @property
    def start(self) -> Seconds:
        return 0

    @property
    def supervisions(self):
        return []

    @property
    def has_features(self) -> bool:
        return self.num_frames is not None

    @property
    def has_recording(self) -> bool:
        return self.num_samples is not None

    @property
    def recording_id(self) -> str:
        return "PAD"

    # noinspection PyUnusedLocal
    def load_features(self, *args, **kwargs) -> Optional[np.ndarray]:
        if self.has_features:
            return (
                np.ones((self.num_frames, self.num_features), np.float32)
                * self.feat_value
            )
        return None

    # noinspection PyUnusedLocal
    def load_audio(self, *args, **kwargs) -> Optional[np.ndarray]:
        if self.has_recording:
            return np.zeros(
                (1, compute_num_samples(self.duration, self.sampling_rate)), np.float32
            )
        return None

    # noinspection PyUnusedLocal
    def truncate(
        self,
        *,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        keep_excessive_supervisions: bool = True,
        preserve_id: bool = False,
        **kwargs,
    ) -> "PaddingCut":
        new_duration = self.duration - offset if duration is None else duration
        assert new_duration > 0.0
        return fastcopy(
            self,
            id=self.id if preserve_id else str(uuid4()),
            duration=new_duration,
            feat_value=self.feat_value,
            num_frames=compute_num_frames(
                duration=new_duration,
                frame_shift=self.frame_shift,
                sampling_rate=self.sampling_rate,
            )
            if self.num_frames is not None
            else None,
            num_samples=compute_num_samples(
                duration=new_duration, sampling_rate=self.sampling_rate
            )
            if self.num_samples is not None
            else None,
        )

    # noinspection PyUnusedLocal
    def extend_by(
        self,
        *,
        duration: Seconds,
        direction: str = "both",
        preserve_id: bool = False,
    ) -> "PaddingCut":
        """
        Return a new PaddingCut with region extended by the specified duration.

        :param duration: The duration by which to extend the cut.
        :param direction: string, 'left', 'right' or 'both'. Determines whether the cut should
            be extended to the left, right or both sides. By default, the cut is extended by
            the specified duration on both sides.
        :param preserve_id: When ``True``, preserves the cut ID from before padding.
            Otherwise, generates a new random ID (default).
        :return: an extended PaddingCut.
        """
        new_duration = self.duration + duration
        if direction == "both":
            new_duration += duration
        assert new_duration > 0.0
        return fastcopy(
            self,
            id=self.id if preserve_id else str(uuid4()),
            duration=new_duration,
            feat_value=self.feat_value,
            num_frames=compute_num_frames(
                duration=new_duration,
                frame_shift=self.frame_shift,
                sampling_rate=self.sampling_rate,
            )
            if self.num_frames is not None
            else None,
            num_samples=compute_num_samples(
                duration=new_duration, sampling_rate=self.sampling_rate
            )
            if self.num_samples is not None
            else None,
        )

    def pad(
        self,
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

        :param duration: The cut's minimal duration after padding.
        :param num_frames: The cut's total number of frames after padding.
        :param num_samples: The cut's total number of samples after padding.
        :param pad_feat_value: A float value that's used for padding the features.
            By default we assume a log-energy floor of approx. -23 (1e-10 after exp).
        :param direction: string, 'left', 'right' or 'both'. Determines whether the padding is added before or after
            the cut.
        :param preserve_id: When ``True``, preserves the cut ID from before padding.
            Otherwise, generates a new random ID (default).
        :param pad_value_dict: Optional dict that specifies what value should be used
            for padding arrays in custom attributes.
        :return: a padded MixedCut if duration is greater than this cut's duration, otherwise ``self``.
        """
        return pad(
            self,
            duration=duration,
            num_frames=num_frames,
            num_samples=num_samples,
            pad_feat_value=pad_feat_value,
            direction=direction,
            preserve_id=preserve_id,
            pad_value_dict=pad_value_dict,
        )

    def resample(self, sampling_rate: int, affix_id: bool = False) -> "PaddingCut":
        """
        Return a new ``MonoCut`` that will lazily resample the audio while reading it.
        This operation will drop the feature manifest, if attached.
        It does not affect the supervision.

        :param sampling_rate: The new sampling rate.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the current ``MonoCut``.
        """
        assert self.has_recording, "Cannot resample a MonoCut without Recording."
        return fastcopy(
            self,
            id=f"{self.id}_rs{sampling_rate}" if affix_id else self.id,
            sampling_rate=sampling_rate,
            num_samples=compute_num_samples(self.duration, sampling_rate),
            num_frames=None,
            num_features=None,
            frame_shift=None,
        )

    def perturb_speed(self, factor: float, affix_id: bool = True) -> "PaddingCut":
        """
        Return a new ``PaddingCut`` that will "mimic" the effect of speed perturbation
        on ``duration`` and ``num_samples``.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``PaddingCut.id`` field
            by affixing it with "_sp{factor}".
        :return: a modified copy of the current ``PaddingCut``.
        """
        # Pre-conditions
        if self.has_features:
            logging.warning(
                "Attempting to perturb speed on a MonoCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "speed perturbation."
            )
            new_num_frames = None
            new_num_features = None
            new_frame_shift = None
        else:
            new_num_frames = self.num_frames
            new_num_features = self.num_features
            new_frame_shift = self.frame_shift
        new_num_samples = perturb_num_samples(self.num_samples, factor)
        new_duration = new_num_samples / self.sampling_rate
        return fastcopy(
            self,
            id=f"{self.id}_sp{factor}" if affix_id else self.id,
            num_samples=new_num_samples,
            duration=new_duration,
            num_frames=new_num_frames,
            num_features=new_num_features,
            frame_shift=new_frame_shift,
        )

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> "PaddingCut":
        """
        Return a new ``PaddingCut`` that will "mimic" the effect of tempo perturbation
        on ``duration`` and ``num_samples``.

        Compared to speed perturbation, tempo preserves pitch.
        :param factor: The tempo will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``PaddingCut.id`` field
            by affixing it with "_tp{factor}".
        :return: a modified copy of the current ``PaddingCut``.
        """
        # Pre-conditions
        if self.has_features:
            logging.warning(
                "Attempting to perturb tempo on a MonoCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "tempo perturbation."
            )
            new_num_frames = None
            new_num_features = None
            new_frame_shift = None
        else:
            new_num_frames = self.num_frames
            new_num_features = self.num_features
            new_frame_shift = self.frame_shift
        new_num_samples = perturb_num_samples(self.num_samples, factor)
        new_duration = new_num_samples / self.sampling_rate
        return fastcopy(
            self,
            id=f"{self.id}_tp{factor}" if affix_id else self.id,
            num_samples=new_num_samples,
            duration=new_duration,
            num_frames=new_num_frames,
            num_features=new_num_features,
            frame_shift=new_frame_shift,
        )

    def perturb_volume(self, factor: float, affix_id: bool = True) -> "PaddingCut":
        """
        Return a new ``PaddingCut`` that will "mimic" the effect of volume perturbation
        on amplitude of samples.

        :param factor: The volume will be adjusted this many times (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``PaddingCut.id`` field
            by affixing it with "_vp{factor}".
        :return: a modified copy of the current ``PaddingCut``.
        """

        return fastcopy(self, id=f"{self.id}_vp{factor}" if affix_id else self.id)

    def reverb_rir(
        self,
        rir_recording: "Recording",
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: List[int] = [0],
    ) -> "PaddingCut":
        """
        Return a new ``PaddingCut`` that will "mimic" the effect of reverberation with impulse response
        on original samples.

        :param rir_recording: The impulse response to use for convolving.
        :param normalize_output: When true, output will be normalized to have energy as input.
        :param early_only: When true, only the early reflections (first 50 ms) will be used.
        :param affix_id: When true, we will modify the ``PaddingCut.id`` field
            by affixing it with "_rvb".
        :param rir_channels: The channels of the impulse response to use.
        :return: a modified copy of the current ``PaddingCut``.
        """

        return fastcopy(self, id=f"{self.id}_rvb" if affix_id else self.id)

    def drop_features(self) -> "PaddingCut":
        """Return a copy of the current :class:`.PaddingCut`, detached from ``features``."""
        assert (
            self.has_recording
        ), f"Cannot detach features from a MonoCut with no Recording (cut ID = {self.id})."
        return fastcopy(self, num_frames=None, num_features=None, frame_shift=None)

    def drop_recording(self) -> "PaddingCut":
        """Return a copy of the current :class:`.PaddingCut`, detached from ``recording``."""
        assert (
            self.has_features
        ), f"Cannot detach recording from a PaddingCut with no Features (cut ID = {self.id})."
        return fastcopy(self, num_samples=None)

    def drop_supervisions(self) -> "PaddingCut":
        """Return a copy of the current :class:`.PaddingCut`, detached from ``supervisions``."""
        return self

    def compute_and_store_features(
        self, extractor: FeatureExtractor, *args, **kwargs
    ) -> Cut:
        """
        Returns a new PaddingCut with updates information about the feature dimension and number of
        feature frames, depending on the ``extractor`` properties.
        """
        return fastcopy(
            self,
            num_features=extractor.feature_dim(self.sampling_rate),
            num_frames=compute_num_frames(
                duration=self.duration,
                frame_shift=extractor.frame_shift,
                sampling_rate=self.sampling_rate,
            ),
            frame_shift=extractor.frame_shift,
        )

    def fill_supervision(self, *args, **kwargs) -> "PaddingCut":
        """
        Just for consistency with :class`.MonoCut` and :class:`.MixedCut`.
        """
        return self

    def map_supervisions(self, transform_fn: Callable[[Any], Any]) -> "PaddingCut":
        """
        Just for consistency with :class:`.MonoCut` and :class:`.MixedCut`.

        :param transform_fn: a dummy function that would be never called actually.
        :return: the PaddingCut itself.
        """
        return self

    def merge_supervisions(self, *args, **kwargs) -> "PaddingCut":
        """
        Just for consistency with :class:`.MonoCut` and :class:`.MixedCut`.

        :return: the PaddingCut itself.
        """
        return self

    def filter_supervisions(
        self, predicate: Callable[[SupervisionSegment], bool]
    ) -> "PaddingCut":
        """
        Just for consistency with :class:`.MonoCut` and :class:`.MixedCut`.

        :param predicate: A callable that accepts `SupervisionSegment` and returns bool
        :return: a modified MonoCut
        """
        return self

    @staticmethod
    def from_dict(data: dict) -> "PaddingCut":
        # Remove "type" field if exists
        data.pop("type", None)
        return PaddingCut(**data)

    def with_features_path_prefix(self, path: Pathlike) -> "PaddingCut":
        return self

    def with_recording_path_prefix(self, path: Pathlike) -> "PaddingCut":
        return self


@dataclass
class MixTrack:
    """
    Represents a single track in a mix of Cuts. Points to a specific MonoCut and holds information on
    how to mix it with other Cuts, relative to the first track in a mix.
    """

    cut: Union[MonoCut, PaddingCut]
    offset: Seconds = 0.0
    snr: Optional[Decibels] = None

    @staticmethod
    def from_dict(data: dict):
        raw_cut = data.pop("cut")
        try:
            cut = MonoCut.from_dict(raw_cut)
        except TypeError:
            cut = PaddingCut.from_dict(raw_cut)
        return MixTrack(cut, **data)


@dataclass
class MixedCut(Cut):
    """
    :class:`~lhotse.cut.MixedCut` is a :class:`~lhotse.cut.Cut` that actually consists of multiple other cuts.
    It can be interpreted as a multi-channel cut, but its primary purpose is to allow
    time-domain and feature-domain augmentation via mixing the training cuts with noise, music, and babble cuts.
    The actual mixing operations are performed on-the-fly.

    Internally, :class:`~lhotse.cut.MixedCut` holds other cuts in multiple trakcs (:class:`~lhotse.cut.MixTrack`),
    each with its own offset and SNR that is relative to the first track.

    Please refer to the documentation of :class:`~lhotse.cut.Cut` to learn more about using cuts.

    In addition to methods available in :class:`~lhotse.cut.Cut`, :class:`~lhotse.cut.MixedCut` provides the methods to
    read all of its tracks audio and features as separate channels:

        >>> cut = MixedCut(...)
        >>> mono_features = cut.load_features()
        >>> assert len(mono_features.shape) == 2
        >>> multi_features = cut.load_features(mixed=False)
        >>> # Now, the first dimension is the channel.
        >>> assert len(multi_features.shape) == 3

    See also:

        - :class:`lhotse.cut.Cut`
        - :class:`lhotse.cut.MonoCut`
        - :class:`lhotse.cut.CutSet`
    """

    id: str
    tracks: List[MixTrack]

    @property
    def supervisions(self) -> List[SupervisionSegment]:
        """
        Lists the supervisions of the underlying source cuts.
        Each segment start time will be adjusted by the track offset.
        """
        return [
            segment.with_offset(track.offset)
            for track in self.tracks
            for segment in track.cut.supervisions
        ]

    @property
    def start(self) -> Seconds:
        return 0

    @property
    def duration(self) -> Seconds:
        track_durations = (track.offset + track.cut.duration for track in self.tracks)
        return round(max(track_durations), ndigits=8)

    @property
    def has_features(self) -> bool:
        return self._first_non_padding_cut.has_features

    @property
    def has_recording(self) -> bool:
        return self._first_non_padding_cut.has_recording

    @property
    def num_frames(self) -> Optional[int]:
        if self.has_features:
            return compute_num_frames(
                duration=self.duration,
                frame_shift=self.frame_shift,
                sampling_rate=self.sampling_rate,
            )
        return None

    @property
    def frame_shift(self) -> Optional[Seconds]:
        return self.tracks[0].cut.frame_shift

    @property
    def sampling_rate(self) -> Optional[int]:
        return self.tracks[0].cut.sampling_rate

    @property
    def num_samples(self) -> Optional[int]:
        return compute_num_samples(self.duration, self.sampling_rate)

    @property
    def num_features(self) -> Optional[int]:
        return self.tracks[0].cut.num_features

    @property
    def features_type(self) -> Optional[str]:
        return self._first_non_padding_cut.features.type if self.has_features else None

    def __getattr__(self, name: str) -> Any:
        """
        This magic function is called when the user tries to access an attribute
        of :class:`.MixedCut` that doesn't exist. It is used for accessing the custom
        attributes of cuts. We support exactly one scenario for mixed cuts:

        If :attr:`tracks` contains exactly one :class:`.MonoCut` object (and an arbitrary
        number of :class:`.PaddingCut` objects), we will look up the custom attributes
        of that cut.

        If one of the custom attributes is of type :class:`~lhotse.array.Array` or
        :class:`~lhotse.array.TemporalArray` we'll also support loading those arrays
        (see example below). Additionally, we will incorporate extra padding as
        dictated by padding cuts.

        Example:

            >>> cut = MonoCut('cut1', start=0, duration=4, channel=0)
            >>> cut.alignment = TemporalArray(...)
            >>> mixed_cut = cut.pad(10, pad_value_dict={'alignment': -1})
            >>> ali = mixed_cut.load_alignment()

        """
        # Python will sometimes try to call undefined magic functions,
        # just fail for them (e.g. __setstate__ when pickling).
        if name.startswith("__"):
            raise AttributeError()

        # Loading a custom array attribute + performing padding.
        if name.startswith("load_"):
            attr_name = name[5:]
            return partial(self.load_custom, attr_name)

        # Returning the contents of "mono_cut.custom[name]",
        # or raising AttributeError.
        try:
            (
                non_padding_idx,
                mono_cut,
            ) = self._assert_one_mono_cut_with_attr_and_return_it_with_track_index(name)
            return getattr(mono_cut, name)
        except AssertionError:
            raise AttributeError(
                f"No such attribute: '{name}' (note: custom attributes are not supported "
                f"when a MixedCut consists of more than one MonoCut with that attribute)."
            )

    def load_custom(self, name: str) -> np.ndarray:
        """
        Load custom data as numpy array. The custom data is expected to have
        been stored in cuts ``custom`` field as an :class:`~lhotse.array.Array` or
        :class:`~lhotse.array.TemporalArray` manifest.

        .. note:: It works with Array manifests stored via attribute assignments,
            e.g.: ``cut.my_custom_data = Array(...)``.

        .. warning:: For :class:`.MixedCut`, this will only work if the mixed cut
            consists of a single :class:`.MonoCut` and an arbitrary number of
            :class:`.PaddingCuts`. This is because it is generally undefined how to
            mix arbitrary arrays.

        :param name: name of the custom attribute.
        :return: a numpy array with the data (after padding).
        """

        from lhotse.array import Array, pad_array

        (
            non_padding_idx,
            mono_cut,
        ) = self._assert_one_mono_cut_with_attr_and_return_it_with_track_index(name)

        # Use getattr to propagate AttributeError if "name" is not defined.
        manifest = getattr(mono_cut, name)

        # Check if the corresponding manifest for 'load_something' is of type
        # Array; if yes, just return the loaded data.
        # This is likely an embedding without a temporal dimension.
        if isinstance(manifest, Array):
            return mono_cut.load_custom(name)

        # We are loading either an array with a temporal dimension, or a recording:
        # We need to pad it.
        left_padding = self.tracks[non_padding_idx].offset
        padded_duration = self.duration

        # Then, check if it's a Recording. In that case we convert it to a cut,
        # leverage existing padding methods, and load padded audio data.
        if isinstance(manifest, Recording):
            return (
                manifest.to_cut()
                .pad(duration=manifest.duration + left_padding, direction="left")
                .pad(duration=padded_duration, direction="right")
                .load_audio()
            )

        # Load the array and retrieve the manifest from the only non-padding cut.
        # We'll also need to fetch the dict defining what padding value to use (if present).
        array = mono_cut.load_custom(name)
        try:
            pad_value_dict = [
                t.cut for t in self.tracks if isinstance(t.cut, PaddingCut)
            ][0].custom
            pad_value = pad_value_dict[name]
        except:
            pad_value = DEFAULT_PADDING_VALUE

        return pad_array(
            array,
            temporal_dim=manifest.temporal_dim,
            frame_shift=manifest.frame_shift,
            offset=left_padding,
            padded_duration=padded_duration,
            pad_value=pad_value,
        )

    def _assert_one_mono_cut_with_attr_and_return_it_with_track_index(
        self,
        attr_name: str,
    ) -> Tuple[int, MonoCut]:
        # TODO(pzelasko): consider relaxing this condition to
        #                 supporting mixed cuts that are not overlapping
        non_padding_cuts = [
            (idx, t.cut)
            for idx, t in enumerate(self.tracks)
            if isinstance(t.cut, MonoCut)
        ]
        non_padding_cuts_with_custom_attr = [
            (idx, cut)
            for idx, cut in non_padding_cuts
            if cut.custom is not None and attr_name in cut.custom
        ]
        assert len(non_padding_cuts_with_custom_attr) == 1, (
            f"This MixedCut has {len(non_padding_cuts_with_custom_attr)} non-padding cuts "
            f"with a custom attribute '{attr_name}'. We currently don't support mixing custom attributes. "
            f"Consider dropping the attribute on all but one of MonoCuts. Problematic cut:\n{self}"
        )
        non_padding_idx, mono_cut = non_padding_cuts_with_custom_attr[0]
        return non_padding_idx, mono_cut

    def truncate(
        self,
        *,
        offset: Seconds = 0.0,
        duration: Optional[Seconds] = None,
        keep_excessive_supervisions: bool = True,
        preserve_id: bool = False,
        _supervisions_index: Optional[Dict[str, IntervalTree]] = None,
    ) -> Cut:
        """
        Returns a new MixedCut that is a sub-region of the current MixedCut. This method truncates the underlying Cuts
        and modifies their offsets in the mix, as needed. Tracks that do not fit in the truncated cut are removed.

        Note that no operation is done on the actual features - it's only during the call to load_features()
        when the actual changes happen (a subset of features is loaded).

        :param offset: float (seconds), controls the start of the new cut relative to the current MixedCut's start.
        :param duration: optional float (seconds), controls the duration of the resulting MixedCut.
            By default, the duration is (end of the cut before truncation) - (offset).
        :param keep_excessive_supervisions: bool. Since trimming may happen inside a SupervisionSegment, the caller has
            an option to either keep or discard such supervisions.
        :param preserve_id: bool. Should the truncated cut keep the same ID or get a new, random one.
        :return: a new MixedCut instance.
        """

        assert (
            offset >= 0
        ), f"Offset for truncate must be non-negative (provided {offset})."
        new_tracks = []
        old_duration = self.duration
        new_mix_end = (
            add_durations(old_duration, -offset, sampling_rate=self.sampling_rate)
            if duration is None
            else add_durations(offset, duration, sampling_rate=self.sampling_rate)
        )

        for track in sorted(self.tracks, key=lambda t: t.offset):
            # First, determine how much of the beginning of the current track we're going to truncate:
            # when the track offset is larger than the truncation offset, we are not truncating the cut;
            # just decreasing the track offset.

            # 'cut_offset' determines how much we're going to truncate the Cut for the current track.
            cut_offset = max(
                add_durations(offset, -track.offset, sampling_rate=self.sampling_rate),
                0,
            )
            # 'track_offset' determines the new track's offset after truncation.
            track_offset = max(
                add_durations(track.offset, -offset, sampling_rate=self.sampling_rate),
                0,
            )
            # 'track_end' is expressed relative to the beginning of the mix
            # (not to be confused with the 'start' of the underlying MonoCut)
            track_end = add_durations(
                track.offset, track.cut.duration, sampling_rate=self.sampling_rate
            )

            if track_end < offset:
                # Omit a MonoCut that ends before the truncation offset.
                continue

            cut_duration_decrease = 0
            if track_end > new_mix_end:
                if duration is not None:
                    cut_duration_decrease = add_durations(
                        track_end, -new_mix_end, sampling_rate=self.sampling_rate
                    )
                else:
                    cut_duration_decrease = add_durations(
                        track_end, -old_duration, sampling_rate=self.sampling_rate
                    )

            # Compute the new MonoCut's duration after trimming the start and the end.
            new_duration = add_durations(
                track.cut.duration,
                -cut_offset,
                -cut_duration_decrease,
                sampling_rate=self.sampling_rate,
            )
            if new_duration <= 0:
                # Omit a MonoCut that is completely outside the time span of the new truncated MixedCut.
                continue

            new_tracks.append(
                MixTrack(
                    cut=track.cut.truncate(
                        offset=cut_offset,
                        duration=new_duration,
                        keep_excessive_supervisions=keep_excessive_supervisions,
                        preserve_id=preserve_id,
                        _supervisions_index=_supervisions_index,
                    ),
                    offset=track_offset,
                    snr=track.snr,
                )
            )
        if len(new_tracks) == 1:
            # The truncation resulted in just a single cut - simply return it.
            return new_tracks[0].cut

        new_cut = MixedCut(
            id=self.id if preserve_id else str(uuid4()), tracks=new_tracks
        )

        # Final edge-case check. Scenario:
        # - some of the original MixedCut tracks had specified an SNR
        # - we truncated away the track that served as an SNR reference
        # - we are left only with PaddingCuts and MonoCuts that have specified SNR
        # Solution:
        # - find first non padding cut and reset its SNR to None (make it the new reference)
        if all(
            t.snr is not None or isinstance(t.cut, PaddingCut) for t in new_cut.tracks
        ):
            first_non_padding_track_idx = [
                idx
                for idx, t in enumerate(new_cut.tracks)
                if not isinstance(t.cut, PaddingCut)
            ][0]
            new_cut.tracks[first_non_padding_track_idx] = fastcopy(
                new_cut.tracks[first_non_padding_track_idx], snr=None
            )

        return new_cut

    def extend_by(
        self,
        *,
        duration: Seconds,
        direction: str = "both",
        preserve_id: bool = False,
    ) -> "MixedCut":
        """
        This raises a ValueError since extending a MixedCut is not defined.

        :param duration: float (seconds), duration (in seconds) to extend the MixedCut.
        :param direction: string, 'left', 'right' or 'both'. Determines whether to extend on the left,
            right, or both sides. If 'both', extend on both sides by the duration specified in `duration`.
        :param preserve_id: bool. Should the extended cut keep the same ID or get a new, random one.
        :return: a new MixedCut instance.
        """
        raise ValueError("The extend_by() method is not defined for a MixedCut.")

    def pad(
        self,
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

        :param duration: The cut's minimal duration after padding.
        :param num_frames: The cut's total number of frames after padding.
        :param num_samples: The cut's total number of samples after padding.
        :param pad_feat_value: A float value that's used for padding the features.
            By default we assume a log-energy floor of approx. -23 (1e-10 after exp).
        :param direction: string, 'left', 'right' or 'both'. Determines whether the padding is added before or after
            the cut.
        :param preserve_id: When ``True``, preserves the cut ID from before padding.
            Otherwise, generates a new random ID (default).
        :param pad_value_dict: Optional dict that specifies what value should be used
            for padding arrays in custom attributes.
        :return: a padded MixedCut if duration is greater than this cut's duration, otherwise ``self``.
        """
        return pad(
            self,
            duration=duration,
            num_frames=num_frames,
            num_samples=num_samples,
            pad_feat_value=pad_feat_value,
            direction=direction,
            preserve_id=preserve_id,
            pad_value_dict=pad_value_dict,
        )

    def resample(self, sampling_rate: int, affix_id: bool = False) -> "MixedCut":
        """
        Return a new ``MixedCut`` that will lazily resample the audio while reading it.
        This operation will drop the feature manifest, if attached.
        It does not affect the supervision.

        :param sampling_rate: The new sampling rate.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the current ``MixedCut``.
        """
        assert self.has_recording, "Cannot resample a MixedCut without Recording."
        return MixedCut(
            id=f"{self.id}_rs{sampling_rate}" if affix_id else self.id,
            tracks=[
                fastcopy(t, cut=t.cut.resample(sampling_rate)) for t in self.tracks
            ],
        )

    def perturb_speed(self, factor: float, affix_id: bool = True) -> "MixedCut":
        """
        Return a new ``MixedCut`` that will lazily perturb the speed while loading audio.
        The ``num_samples``, ``start`` and ``duration`` fields of the underlying Cuts
        (and their Recordings and SupervisionSegments) are updated to reflect
        the shrinking/extending effect of speed.
        We are also updating the offsets of all underlying tracks.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``MixedCut.id`` field
            by affixing it with "_sp{factor}".
        :return: a modified copy of the current ``MixedCut``.
        """
        # TODO(pzelasko): test most extensively for edge cases
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb speed on a MonoCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb speed on a MixedCut that references pre-computed features. "
                "The feature manifest(s) will be detached, as we do not support feature-domain "
                "speed perturbation."
            )
        return MixedCut(
            id=f"{self.id}_sp{factor}" if affix_id else self.id,
            tracks=[
                MixTrack(
                    cut=track.cut.perturb_speed(factor=factor, affix_id=affix_id),
                    offset=round(
                        perturb_num_samples(
                            num_samples=compute_num_samples(
                                track.offset, self.sampling_rate
                            ),
                            factor=factor,
                        )
                        / self.sampling_rate,
                        ndigits=8,
                    ),
                    snr=track.snr,
                )
                for track in self.tracks
            ],
        )

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> "MixedCut":
        """
        Return a new ``MixedCut`` that will lazily perturb the tempo while loading audio.

        Compared to speed perturbation, tempo preserves pitch.
        The ``num_samples``, ``start`` and ``duration`` fields of the underlying Cuts
        (and their Recordings and SupervisionSegments) are updated to reflect
        the shrinking/extending effect of tempo.
        We are also updating the offsets of all underlying tracks.

        :param factor: The tempo will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``MixedCut.id`` field
            by affixing it with "_tp{factor}".
        :return: a modified copy of the current ``MixedCut``.
        """
        # TODO(pzelasko): test most extensively for edge cases
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb tempo on a MonoCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb tempo on a MixedCut that references pre-computed features. "
                "The feature manifest(s) will be detached, as we do not support feature-domain "
                "tempo perturbation."
            )
        return MixedCut(
            id=f"{self.id}_tp{factor}" if affix_id else self.id,
            tracks=[
                MixTrack(
                    cut=track.cut.perturb_tempo(factor=factor, affix_id=affix_id),
                    offset=round(
                        perturb_num_samples(
                            num_samples=compute_num_samples(
                                track.offset, self.sampling_rate
                            ),
                            factor=factor,
                        )
                        / self.sampling_rate,
                        ndigits=8,
                    ),
                    snr=track.snr,
                )
                for track in self.tracks
            ],
        )

    def perturb_volume(self, factor: float, affix_id: bool = True) -> "MixedCut":
        """
        Return a new ``MixedCut`` that will lazily perturb the volume while loading audio.
        Recordings of the underlying Cuts are updated to reflect volume change.

        :param factor: The volume will be adjusted this many times (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``MixedCut.id`` field
            by affixing it with "_vp{factor}".
        :return: a modified copy of the current ``MixedCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb volume on a MonoCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb volume on a MixedCut that references pre-computed features. "
                "The feature manifest(s) will be detached, as we do not support feature-domain "
                "volume perturbation."
            )
        return MixedCut(
            id=f"{self.id}_vp{factor}" if affix_id else self.id,
            tracks=[
                fastcopy(
                    track,
                    cut=track.cut.perturb_volume(factor=factor, affix_id=affix_id),
                )
                for track in self.tracks
            ],
        )

    def reverb_rir(
        self,
        rir_recording: "Recording",
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: List[int] = [0],
    ) -> "MixedCut":
        """
        Return a new ``MixedCut`` that will convolve the audio with the provided impulse response.

        :param rir_recording: The impulse response to use for convolving.
        :param normalize_output: When true, output will be normalized to have energy as input.
        :param early_only: When true, only the early reflections (first 50 ms) will be used.
        :param affix_id: When true, we will modify the ``MixedCut.id`` field
            by affixing it with "_rvb".
        :param rir_channels: The channels of the impulse response to use. By default, first channel is used.
            If only one channel is specified, all tracks will be convolved with this channel. If a list
            is provided, it must contain as many channels as there are tracks such that each track will
            be convolved with one of the specified channels.
        :return: a modified copy of the current ``MixedCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot apply reverberation on a MixedCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to reverberate a MixedCut that references pre-computed features. "
                "The feature manifest(s) will be detached, as we do not support feature-domain "
                "reverberation."
            )

        assert all(
            c < rir_recording.num_channels for c in rir_channels
        ), "Invalid channel index in `rir_channels`."
        assert len(rir_channels) == 1 or len(rir_channels) == len(
            self.tracks
        ), "Invalid number of channels in `rir_channels`. Must be 1 or equal to number of tracks."

        if len(rir_channels) == 1:
            rir_channels = rir_channels * len(self.tracks)

        return MixedCut(
            id=f"{self.id}_rvb" if affix_id else self.id,
            tracks=[
                fastcopy(
                    track,
                    cut=track.cut.reverb_rir(
                        rir_recording=rir_recording,
                        normalize_output=normalize_output,
                        early_only=early_only,
                        affix_id=affix_id,
                        rir_channels=[channel],
                    ),
                )
                for track, channel in zip(self.tracks, rir_channels)
            ],
        )

    @rich_exception_info
    def load_features(self, mixed: bool = True) -> Optional[np.ndarray]:
        """
        Loads the features of the source cuts and mixes them on-the-fly.

        :param mixed: when True (default), returns a 2D array of features mixed in the feature domain.
            Otherwise returns a 3D array with the first dimension equal to the number of tracks.
        :return: A numpy ndarray with features and with shape ``(num_frames, num_features)``,
            or ``(num_tracks, num_frames, num_features)``
        """
        if not self.has_features:
            return None
        first_cut = self.tracks[0].cut

        # First, check for a simple scenario: just a single cut with padding.
        # When that is the case, we don't have to instantiate a feature extractor,
        # because we are not performing any actual mixing.
        # That makes life simpler for the users who have a custom feature extractor,
        # but don't actually care about feature-domain mixing; just want to pad.
        if mixed and all(isinstance(t.cut, PaddingCut) for t in self.tracks[1:]):
            padding_val = self.tracks[1].cut.feat_value
            feats = np.ones((self.num_frames, self.num_features)) * padding_val
            feats[: first_cut.num_frames, :] = first_cut.load_features()
            return feats

        # When there is more than one "regular" cut, we will perform an actual mix.

        # First, we have to make sure that the reference energy levels are appropriate.
        # They might not be if the first track is a padding track.
        reference_feats = None
        reference_energy = None
        reference_pos, reference_cut = [
            (idx, t.cut)
            for idx, t in enumerate(self.tracks)
            if not isinstance(t.cut, PaddingCut) and t.snr is None
        ][0]
        feature_extractor = create_default_feature_extractor(
            reference_cut.features.type
        )
        if first_cut.id != reference_cut.id:
            reference_feats = reference_cut.load_features()
            reference_energy = feature_extractor.compute_energy(reference_feats)

        # The mix itself.
        mixer = FeatureMixer(
            feature_extractor=create_default_feature_extractor(
                self._first_non_padding_cut.features.type
            ),
            base_feats=first_cut.load_features(),
            frame_shift=first_cut.frame_shift,
            reference_energy=reference_energy,
        )
        for pos, track in enumerate(self.tracks[1:], start=1):
            try:
                if pos == reference_pos and reference_feats is not None:
                    feats = reference_feats  # manual caching to avoid duplicated I/O
                else:
                    feats = track.cut.load_features()
                mixer.add_to_mix(
                    feats=feats,
                    snr=track.snr,
                    offset=track.offset,
                    sampling_rate=track.cut.sampling_rate,
                )
            except NonPositiveEnergyError as e:
                logging.warning(
                    str(e) + f' MonoCut with id "{track.cut.id}" will not be mixed in.'
                )

        if mixed:
            # Checking for some edge cases below.
            feats = mixer.mixed_feats
            # Note: The slicing below is a work-around for an edge-case
            #  when two cuts have durations that ended with 0.005 (e.g. 10.125 and 5.715)
            #  - then, the feature extractor "squeezed in" a last extra frame and the simple
            #  relationship between num_frames and duration we strived for is not true;
            #  i.e. the duration is 10.125 + 5.715 = 15.84, but the number of frames is
            #  1013 + 572 = 1585. If the frame_shift is 0.01, we have gained an extra 0.01s...
            if feats.shape[0] - self.num_frames == 1:
                feats = feats[: self.num_frames, :]
            # TODO(pzelasko): This can sometimes happen in a MixedCut with >= 5 different Cuts,
            #   with a regular MonoCut at the end, when the mix offsets are floats with a lot of decimals.
            #   For now we're duplicating the last frame to match the declared "num_frames" of this cut.
            if feats.shape[0] - self.num_frames == -1:
                feats = np.concatenate((feats, feats[-1:, :]), axis=0)
            assert feats.shape[0] == self.num_frames, (
                "Inconsistent number of frames in a MixedCut: please report "
                "this issue at https://github.com/lhotse-speech/lhotse/issues "
                "showing the output of print(cut) or str(cut) on which"
                "load_features() was called."
            )
            return feats
        else:
            return mixer.unmixed_feats

    @rich_exception_info
    def load_audio(self, mixed: bool = True) -> Optional[np.ndarray]:
        """
        Loads the audios of the source cuts and mix them on-the-fly.

        :param mixed: When True (default), returns a mono mix of the underlying tracks.
            Otherwise returns a numpy array with the number of channels equal to the number of tracks.
        :return: A numpy ndarray with audio samples and with shape ``(num_channels, num_samples)``
        """
        if not self.has_recording:
            return None
        first_cut = self.tracks[0].cut

        # First, we have to make sure that the reference energy levels are appropriate.
        # They might not be if the first track is a padding track.
        reference_audio = None
        reference_energy = None
        reference_pos, reference_cut = [
            (idx, t.cut)
            for idx, t in enumerate(self.tracks)
            if not isinstance(t.cut, PaddingCut) and t.snr is None
        ][0]
        if first_cut.id != reference_cut.id:
            reference_audio = reference_cut.load_audio()
            reference_energy = audio_energy(reference_audio)

        try:
            mixer = AudioMixer(
                self.tracks[0].cut.load_audio(),
                sampling_rate=self.tracks[0].cut.sampling_rate,
                reference_energy=reference_energy,
            )
        except NonPositiveEnergyError as e:
            logging.warning(
                f"{e}\nNote: we cannot mix signal with a given SNR to the reference audio with zero energy. "
                f'Cut ID: "{self.tracks[0].cut.id}"'
            )
            raise

        for pos, track in enumerate(self.tracks[1:], start=1):
            try:
                if pos == reference_pos and reference_audio is not None:
                    audio = reference_audio  # manual caching to avoid duplicated I/O
                else:
                    audio = track.cut.load_audio()
                mixer.add_to_mix(
                    audio=audio,
                    snr=track.snr,
                    offset=track.offset,
                )
            except NonPositiveEnergyError as e:
                logging.warning(
                    f'{e} MonoCut with id "{track.cut.id}" will not be mixed in.'
                )

        if mixed:
            # Off-by-one errors can happen during mixing due to imperfect float arithmetic and rounding;
            # we will fix them on-the-fly so that the manifest does not lie about the num_samples.
            audio = mixer.mixed_audio
            if audio.shape[1] - self.num_samples == 1:
                audio = audio[:, : self.num_samples]
            if audio.shape[1] - self.num_samples == -1:
                audio = np.concatenate((audio, audio[:, -1:]), axis=1)
            assert audio.shape[1] == self.num_samples, (
                f"Inconsistent number of samples in a MixedCut: please report "
                f"this issue at https://github.com/lhotse-speech/lhotse/issues "
                f"showing the cut below. MixedCut:\n{self}"
            )
        else:
            audio = mixer.unmixed_audio

        return audio

    def plot_tracks_features(self):
        """
        Display the feature matrix as an image. Requires matplotlib to be installed.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(self.tracks))
        features = self.load_features(mixed=False)
        fmin, fmax = features.min(), features.max()
        for idx, ax in enumerate(axes):
            ax.imshow(np.flip(features[idx].transpose(1, 0), 0), vmin=fmin, vmax=fmax)
        return axes

    def plot_tracks_audio(self):
        """
        Display plots of the individual tracks' waveforms. Requires matplotlib to be installed.
        """
        import matplotlib.pyplot as plt

        audio = self.load_audio(mixed=False)
        fig, axes = plt.subplots(len(self.tracks), sharex=False, sharey=True)
        for idx, (track, ax) in enumerate(zip(self.tracks, axes)):
            samples = audio[idx, :]
            ax.plot(np.linspace(0, self.duration, len(samples)), samples)
            for supervision in track.cut.supervisions:
                supervision = supervision.trim(track.cut.duration)
                ax.axvspan(
                    track.offset + supervision.start,
                    track.offset + supervision.end,
                    color="green",
                    alpha=0.1,
                )
        return axes

    def drop_features(self) -> "MixedCut":
        """Return a copy of the current :class:`MixedCut`, detached from ``features``."""
        assert (
            self.has_recording
        ), f"Cannot detach features from a MixedCut with no Recording (cut ID = {self.id})."
        return fastcopy(
            self, tracks=[fastcopy(t, cut=t.cut.drop_features()) for t in self.tracks]
        )

    def drop_recording(self) -> "MixedCut":
        """Return a copy of the current :class:`.MixedCut`, detached from ``recording``."""
        assert (
            self.has_features
        ), f"Cannot detach recording from a MixedCut with no Features (cut ID = {self.id})."
        return fastcopy(
            self, tracks=[fastcopy(t, cut=t.cut.drop_recording()) for t in self.tracks]
        )

    def drop_supervisions(self) -> "MixedCut":
        """Return a copy of the current :class:`.MixedCut`, detached from ``supervisions``."""
        return fastcopy(
            self,
            tracks=[fastcopy(t, cut=t.cut.drop_supervisions()) for t in self.tracks],
        )

    def compute_and_store_features(
        self,
        extractor: FeatureExtractor,
        storage: FeaturesWriter,
        augment_fn: Optional[AugmentFn] = None,
        mix_eagerly: bool = True,
    ) -> Cut:
        """
        Compute the features from this cut, store them on disk, and create a new `MonoCut` object with the
        feature manifest attached. This cut has to be able to load audio.

        :param extractor: a ``FeatureExtractor`` instance used to compute the features.
        :param storage: a ``FeaturesWriter`` instance used to store the features.
        :param augment_fn: an optional callable used for audio augmentation.
        :param mix_eagerly: when False, extract and store the features for each track separately,
            and mix them dynamically when loading the features.
            When True, mix the audio first and store the mixed features, returning a new ``MonoCut`` instance
            with the same ID. The returned ``MonoCut`` will not have a ``Recording`` attached.
        :return: a new ``MonoCut`` instance if ``mix_eagerly`` is True, or returns ``self``
            with each of the tracks containing the ``Features`` manifests.
        """
        if mix_eagerly:
            features_info = extractor.extract_from_samples_and_store(
                samples=self.load_audio(),
                storage=storage,
                sampling_rate=self.sampling_rate,
                offset=0,
                channel=0,
                augment_fn=augment_fn,
            )
            features_info.recording_id = self.id
            return MonoCut(
                id=self.id,
                start=0,
                duration=self.duration,
                channel=0,
                supervisions=[
                    fastcopy(s, recording_id=self.id) for s in self.supervisions
                ],
                features=features_info,
                recording=None,
                custom=self.custom if hasattr(self, "custom") else None,
            )
        else:  # mix lazily
            new_tracks = [
                MixTrack(
                    cut=track.cut.compute_and_store_features(
                        extractor=extractor,
                        storage=storage,
                        augment_fn=augment_fn,
                    ),
                    offset=track.offset,
                    snr=track.snr,
                )
                for track in self.tracks
            ]
            return MixedCut(id=self.id, tracks=new_tracks)

    def fill_supervision(
        self, add_empty: bool = True, shrink_ok: bool = False
    ) -> "MixedCut":
        """
        Fills the whole duration of a cut with a supervision segment.

        If the cut has one supervision, its start is set to 0 and duration is set to ``cut.duration``.
        Note: this may either expand a supervision that was shorter than a cut, or shrink a supervision
        that exceeds the cut.

        If there are no supervisions, we will add an empty one when ``add_empty==True``, otherwise
        we won't change anything.

        If there are two or more supervisions, we will raise an exception.

        .. note:: For :class:`.MixedCut`, we expect that only one track contains a supervision.
            That supervision will be expanded to cover the full MixedCut's duration.

        :param add_empty: should we add an empty supervision with identical time bounds as the cut.
        :param shrink_ok: should we raise an error if a supervision would be shrank as a result
            of calling this method.
        """
        n_sups = len(self.supervisions)
        if n_sups == 0:
            if not add_empty:
                return self
            first_non_padding_idx = [
                idx for idx, t in enumerate(self.tracks) if isinstance(t.cut, MonoCut)
            ][0]
            new_tracks = [
                fastcopy(
                    t,
                    cut=fastcopy(
                        t.cut,
                        supervisions=[
                            SupervisionSegment(
                                id=self.id,
                                recording_id=t.cut.recording_id,
                                start=-t.offset,
                                duration=self.duration,
                                channel=-1,
                            )
                        ],
                    ),
                )
                if idx == first_non_padding_idx
                else t
                for idx, t in enumerate(self.tracks)
            ]
        else:
            assert (
                n_sups == 1
            ), f"Cannot expand more than one supervision (found {len(self.supervisions)}."
            new_tracks = []
            for t in self.tracks:
                if len(t.cut.supervisions) == 0:
                    new_tracks.append(t)
                else:
                    sup = t.cut.supervisions[0]
                    if not shrink_ok and (
                        sup.start < -t.offset or sup.end > self.duration
                    ):
                        raise ValueError(
                            f"Cannot shrink supervision (start={sup.start}, end={sup.end}) to cut "
                            f"(start=0, duration={t.cut.duration}) because the argument `shrink_ok` is `False`. "
                            f"Note: this check prevents accidental data loss for speech recognition, "
                            f"as supervision exceeding a cut indicates there might be some spoken content "
                            f"beyond cuts start or end (an ASR model would be trained to predict more text than "
                            f"spoken in the audio). If this is okay, set `shrink_ok` to `True`."
                        )
                    new_tracks.append(
                        fastcopy(
                            t,
                            cut=fastcopy(
                                t.cut,
                                supervisions=[
                                    fastcopy(
                                        sup, start=-t.offset, duration=self.duration
                                    )
                                ],
                            ),
                        )
                    )
        return fastcopy(self, tracks=new_tracks)

    def map_supervisions(
        self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]
    ) -> Cut:
        """
        Modify the SupervisionSegments by `transform_fn` of this MixedCut.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a modified MixedCut.
        """
        new_mixed_cut = fastcopy(self)
        for track in new_mixed_cut.tracks:
            track.cut.supervisions = [
                segment.map(transform_fn) for segment in track.cut.supervisions
            ]
        return new_mixed_cut

    def merge_supervisions(
        self, custom_merge_fn: Optional[Callable[[str, Iterable[Any]], Any]] = None
    ) -> "MixedCut":
        """
        Return a copy of the cut that has all of its supervisions merged into
        a single segment.

        The new start is the start of the earliest superivion, and the new duration
        is a minimum spanning duration for all the supervisions.

        The text fields are concatenated with a whitespace, and all other string fields
        (including IDs) are prefixed with "cat#" and concatenated with a hash symbol "#".
        This is also applied to ``custom`` fields. Fields with a ``None`` value are omitted.

        .. note:: If you're using individual tracks of a mixed cut, note that this transform
             drops all the supervisions in individual tracks and assigns the merged supervision
             in the first :class:`.MonoCut` found in ``self.tracks``.

        :param custom_merge_fn: a function that will be called to merge custom fields values.
            We expect ``custom_merge_fn`` to handle all possible custom keys.
            When not provided, we will treat all custom values as strings.
            It will be called roughly like:
            ``custom_merge_fn(custom_key, [s.custom[custom_key] for s in sups])``
        """
        return merge_supervisions(self, custom_merge_fn=custom_merge_fn)

    def filter_supervisions(
        self, predicate: Callable[[SupervisionSegment], bool]
    ) -> Cut:
        """
        Modify cut to store only supervisions accepted by `predicate`

        Example:
            >>> cut = cut.filter_supervisions(lambda s: s.id in supervision_ids)
            >>> cut = cut.filter_supervisions(lambda s: s.duration < 5.0)
            >>> cut = cut.filter_supervisions(lambda s: s.text is not None)

        :param predicate: A callable that accepts `SupervisionSegment` and returns bool
        :return: a modified MonoCut
        """
        new_mixed_cut = fastcopy(
            self,
            tracks=[
                fastcopy(track, cut=track.cut.filter_supervisions(predicate))
                for track in self.tracks
            ],
        )
        return new_mixed_cut

    @staticmethod
    def from_dict(data: dict) -> "MixedCut":
        if "type" in data:
            data.pop("type")
        return MixedCut(
            id=data["id"],
            tracks=[MixTrack.from_dict(track) for track in data["tracks"]],
        )

    def with_features_path_prefix(self, path: Pathlike) -> "MixedCut":
        if not self.has_features:
            return self
        return MixedCut(
            id=self.id,
            tracks=[
                fastcopy(t, cut=t.cut.with_features_path_prefix(path))
                for t in self.tracks
            ],
        )

    def with_recording_path_prefix(self, path: Pathlike) -> "MixedCut":
        if not self.has_recording:
            return self
        return MixedCut(
            id=self.id,
            tracks=[
                fastcopy(t, cut=t.cut.with_recording_path_prefix(path))
                for t in self.tracks
            ],
        )

    @property
    def _first_non_padding_cut(self) -> MonoCut:
        return self._first_non_padding_track.cut

    @property
    def _first_non_padding_track(self) -> MixTrack:
        return [t for t in self.tracks if not isinstance(t.cut, PaddingCut)][0]


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
        def deserialize_one(raw_cut: dict) -> Cut:
            cut_type = raw_cut.pop("type")
            if cut_type == "MonoCut":
                return MonoCut.from_dict(raw_cut)
            if cut_type == "Cut":
                warnings.warn(
                    "Your manifest was created with Lhotse version earlier than v0.8, when MonoCut was called Cut. "
                    "Please re-generate it with Lhotse v0.8 as it might stop working in a future version "
                    "(using manifest.from_file() and then manifest.to_file() should be sufficient)."
                )
                return MonoCut.from_dict(raw_cut)
            if cut_type == "MixedCut":
                return MixedCut.from_dict(raw_cut)
            raise ValueError(
                f"Unexpected cut type during deserialization: '{cut_type}'"
            )

        return CutSet.from_cuts(deserialize_one(cut) for cut in data)

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

            def save(mono_cut: MonoCut):
                if mono_cut.has_recording and mono_cut.recording_id not in stored_rids:
                    rw.write(mono_cut.recording)
                    stored_rids.add(mono_cut.recording_id)
                if mono_cut.has_features:
                    # Note: we have no way of saying if features are unique,
                    #       so we will always write them.
                    fw.write(mono_cut.features)
                for sup in mono_cut.supervisions:
                    if sup.id not in stored_sids:
                        # Supervisions inside cuts are relative to cuts start,
                        # so we correct the offset.
                        sw.write(sup.with_offset(mono_cut.start))
                        stored_sids.add(sup.id)

            for cut in tqdm(self, desc="Decomposing cuts") if verbose else self:
                if isinstance(cut, MonoCut):
                    save(cut)
                elif isinstance(cut, MixedCut):
                    for track in cut.tracks:
                        if isinstance(track.cut, MonoCut):
                            save(track.cut)

        return rw.open_manifest(), sw.open_manifest(), fw.open_manifest()

    def describe(self) -> None:
        """
        Print a message describing details about the ``CutSet`` - the number of cuts and the
        duration statistics, including the total duration and the percentage of speech segments.

        Example output:
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
            99%     2500.0
            99.5%   2523.0
            99.9%   2601.0
            max     5415.0
            dtype: float64
        """
        durations = np.array([c.duration for c in self])
        speech_durations = np.array(
            [s.duration for c in self for s in c.trimmed_supervisions]
        )
        total_sum = durations.sum()
        speech_sum = speech_durations.sum()
        print("Cuts count:", len(durations))
        print(f"Total duration (hours): {total_sum / 3600:.1f}")
        print(
            f"Speech duration (hours): {speech_sum / 3600:.1f} ({speech_sum / total_sum:.1%})"
        )
        print("***")
        print("Duration statistics (seconds):")
        print(f"mean\t{np.mean(durations):.1f}")
        print(f"std\t{np.std(durations):.1f}")
        print(f"min\t{np.min(durations):.1f}")
        print(f"25%\t{np.percentile(durations, 25):.1f}")
        print(f"50%\t{np.median(durations):.1f}")
        print(f"75%\t{np.percentile(durations, 75):.1f}")
        print(f"99%\t{np.percentile(durations, 99):.1f}")
        print(f"99.5%\t{np.percentile(durations, 99.5):.1f}")
        print(f"99.9%\t{np.percentile(durations, 99.9):.1f}")
        print(f"max\t{np.max(durations):.1f}")

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
        return CutSet.from_cuts(cut.filter_supervisions(predicate) for cut in self)

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
        return CutSet.from_cuts(
            c.merge_supervisions(custom_merge_fn=custom_merge_fn) for c in self
        )

    def trim_to_supervisions(
        self,
        keep_overlapping: bool = True,
        min_duration: Optional[Seconds] = None,
        context_direction: Literal["center", "left", "right", "random"] = "center",
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
        :param num_jobs: Number of parallel workers to process the cuts.
        :return: a ``CutSet``.
        """
        if num_jobs == 1:
            return CutSet.from_cuts(
                # chain.from_iterable is a flatten operation: Iterable[Iterable[T]] -> Iterable[T]
                chain.from_iterable(
                    cut.trim_to_supervisions(
                        keep_overlapping=keep_overlapping,
                        min_duration=min_duration,
                        context_direction=context_direction,
                    )
                    for cut in self
                )
            )

        from lhotse.manipulation import split_parallelize_combine

        result = split_parallelize_combine(
            num_jobs,
            self,
            CutSet.trim_to_supervisions,
            keep_overlapping=keep_overlapping,
            min_duration=min_duration,
            context_direction=context_direction,
        )
        return result

    def trim_to_unsupervised_segments(self) -> "CutSet":
        """
        Return a new CutSet with Cuts created from segments that have no supervisions (likely
        silence or noise).

        :return: a ``CutSet``.
        """
        from cytoolz import sliding_window

        cuts = []
        for cut in self:
            segments = []
            supervisions = sorted(cut.supervisions, key=lambda s: s.start)
            # Check if there is an unsupervised segment at the start of the cut,
            # before the first supervision.
            if supervisions[0].start > 0:
                segments.append((0, supervisions[0].start))
            # Check if there are unsupervised segments between the supervisions.
            for left, right in sliding_window(2, supervisions):
                if overlaps(left, right) or left.end == right.start:
                    continue
                segments.append((left.end, right.start))
            # Check if there is an unsupervised segment after the last supervision,
            # before the cut ends.
            if supervisions[-1].end < cut.duration:
                segments.append((supervisions[-1].end, cut.duration))
            # Create cuts from all found unsupervised segments.
            for start, end in segments:
                cuts.append(cut.truncate(offset=start, duration=end - start))
        return CutSet.from_cuts(cuts)

    def mix_same_recording_channels(self) -> "CutSet":
        """
        Find cuts that come from the same recording and have matching start and end times, but
        represent different channels. Then, mix them together (in matching groups) and return
        a new ``CutSet`` that contains their mixes. This is useful for processing microphone array
        recordings.

        It is intended to be used as the first operation after creating a new ``CutSet`` (but
        might also work in other circumstances, e.g. if it was cut to windows first).

        Example:
            >>> ami = prepare_ami('path/to/ami')
            >>> cut_set = CutSet.from_manifests(recordings=ami['train']['recordings'])
            >>> multi_channel_cut_set = cut_set.mix_same_recording_channels()

        In the AMI example, the ``multi_channel_cut_set`` will yield MixedCuts that hold all single-channel
        Cuts together.
        """
        if self.mixed_cuts:
            raise ValueError(
                "This operation is not applicable to CutSet's containing MixedCut's."
            )
        from cytoolz.itertoolz import groupby

        groups = groupby(lambda cut: (cut.recording.id, cut.start, cut.end), self)
        return CutSet.from_cuts(mix_cuts(cuts) for cuts in groups.values())

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
        :return: a mapping from MonoCut ID to an interval tree of SupervisionSegments.
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

        return CutSet.from_cuts(
            cut.pad(
                duration=duration,
                num_frames=num_frames,
                num_samples=num_samples,
                pad_feat_value=pad_feat_value,
                direction=direction,
                preserve_id=preserve_id,
                pad_value_dict=pad_value_dict,
            )
            for cut in self
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
    ) -> "CutSet":
        """
        Returns a new CutSet with cuts extended by `duration` amount.

        :param duration: float (seconds), specifies the duration by which the CutSet is extended.
        :param direction: string, 'left', 'right' or 'both'. Determines whether to extend on the left,
            right, or both sides. If 'both', extend on both sides by the same duration (equal to `duration`).
        :param preserve_id: bool. Should the extended cut keep the same ID or get a new, random one.
        :return: a new CutSet instance.
        """
        return CutSet.from_cuts(
            cut.extend_by(
                duration=duration, direction=direction, preserve_id=preserve_id
            )
            for cut in self
        )

    def cut_into_windows(
        self,
        duration: Seconds,
        hop: Optional[Seconds] = None,
        keep_excessive_supervisions: bool = True,
        num_jobs: int = 1,
    ) -> "CutSet":
        """
        Return a new ``CutSet``, made by traversing each ``MonoCut`` in windows of ``duration`` seconds by ``hop`` seconds and
        creating new ``MonoCut`` out of them.

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
                        lambda cut: cut.cut_into_windows(
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
            partial(
                _cut_into_windows_single,
                duration=duration,
                hop=hop,
                keep_excessive_supervisions=keep_excessive_supervisions,
            ),
        )
        return result

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
        rir_recordings: "RecordingSet",
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

        :param rir_recordings: RecordingSet containing the room impulse responses.
        :param normalize_output: When true, output will be normalized to have energy as input.
        :param early_only: When true, only the early reflections (first 50 ms) will be used.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :param rir_channels: The channels of the impulse response to use. By default, first channel will be used.
            If it is a multi-channel RIR, applying RIR will produce MixedCut.
        :return: a modified copy of the ``CutSet``.
        """
        rir_recordings = list(rir_recordings)
        return CutSet.from_cuts(
            [
                cut.reverb_rir(
                    rir_recording=random.choice(rir_recordings),
                    normalize_output=normalize_output,
                    early_only=early_only,
                    affix_id=affix_id,
                    rir_channels=rir_channels,
                )
                for cut in self
            ]
        )

    def mix(
        self,
        cuts: "CutSet",
        duration: Optional[Seconds] = None,
        allow_padding: bool = False,
        snr: Optional[Union[Decibels, Sequence[Decibels]]] = 20,
        preserve_id: Optional[str] = None,
        mix_prob: float = 1.0,
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
        mixed_cuts = []
        for cut in self:
            # Check whether we're going to mix something into the current cut
            # or pass it through unchanged.
            if random.uniform(0.0, 1.0) > mix_prob:
                mixed_cuts.append(cut)
                continue
            # Randomly sample a new cut from "cuts" to mix in.
            to_mix = cuts.sample()
            # Determine the SNR - either it's specified or we need to sample one.
            snr = random.uniform(*snr) if isinstance(snr, (list, tuple)) else snr
            # Actual mixing
            mixed = cut.mix(other=to_mix, snr=snr, preserve_id=preserve_id)
            # Did the user specify a duration?
            # If yes, we will ensure that shorter cuts have more noise mixed in
            # to "pad" them with at the end.
            if duration is not None:
                mixed_in_duration = to_mix.duration
                # Keep sampling until we mixed in a "duration" amount of noise.
                # Note: we subtract 0.05s (50ms) from the target duration to avoid edge cases
                #       where we mix in some noise cut that effectively has 0 frames of features.
                while mixed_in_duration < (duration - 0.05):
                    to_mix = cuts.sample()
                    # Keep the SNR constant for each cut from "self".
                    mixed = mixed.mix(
                        other=to_mix,
                        snr=snr,
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
                duration=cut.duration if duration is None else duration,
                preserve_id=preserve_id is not None,
            )
            mixed_cuts.append(mixed)
        return CutSet.from_cuts(mixed_cuts)

    def drop_features(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its extracted features.
        """
        return CutSet.from_cuts(c.drop_features() for c in self)

    def drop_recordings(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its recordings.
        """
        return CutSet.from_cuts(c.drop_recording() for c in self)

    def drop_supervisions(self) -> "CutSet":
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its supervisions.
        """
        return CutSet.from_cuts(c.drop_supervisions() for c in self)

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
            returning a new ``MonoCut`` instance with the same ID.
            The returned ``MonoCut`` will not have a ``Recording`` attached.
        :param progress_bar: Should a progress bar be displayed (automatically turned off
            for parallel computation).
        :return: Returns a new ``CutSet`` with ``Features`` manifests attached to the cuts.
        """
        from lhotse.manipulation import combine
        from cytoolz import identity

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
        cut_sets = self.split(num_jobs, shuffle=True)

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
                        channels=0,
                        storage_type=feats_writer.name,
                        storage_path=str(feats_writer.storage_path),
                        storage_key=storage_key,
                    )
                    validate_features(feat_manifest, feats_data=feat_mat)

                    # Update the cut manifest.
                    if isinstance(cut, MonoCut):
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

    @deprecated(
        "CutSet.compute_and_store_recordings will be removed in a future release. Please use save_audios() instead."
    )
    def compute_and_store_recordings(
        self,
        storage_path: Pathlike,
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
            using the cut ID as filename and '.flac' as suffix.
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
        return self.save_audios(
            storage_path,
            num_jobs=num_jobs,
            executor=executor,
            augment_fn=augment_fn,
            progress_bar=progress_bar,
        )

    def save_audios(
        self,
        storage_path: Pathlike,
        format: str = "wav",
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
        from lhotse.manipulation import combine
        from cytoolz import identity

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
            return (subdir / cut.id).with_suffix(f".{format}")

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
        self, storage_path: Optional[Pathlike] = None, max_cuts: Optional[int] = None
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
        :return a dict of ``{'norm_means': np.ndarray, 'norm_stds': np.ndarray}`` with the
            shape of the arrays equal to the number of feature bins in this manifest.
        """
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
        return CutSet.from_cuts(c.with_features_path_prefix(path) for c in self)

    def with_recording_path_prefix(self, path: Pathlike) -> "CutSet":
        return CutSet.from_cuts(c.with_recording_path_prefix(path) for c in self)

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
                        if isinstance(t.cut, MonoCut):
                            t.cut.features = t.cut.features.copy_feats(writer=writer)
                    manifest_writer.write(cpy)

                elif isinstance(item, MonoCut):
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
        return CutSet.from_cuts(c.with_id(transform_fn(c.id)) for c in self)

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
        return CutSet.from_cuts(
            cut.fill_supervision(add_empty=add_empty, shrink_ok=shrink_ok)
            for cut in self
        )

    def map_supervisions(
        self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]
    ) -> "CutSet":
        """
        Modify the SupervisionSegments by `transform_fn` in this CutSet.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a new, modified CutSet.
        """
        return CutSet.from_cuts(cut.map_supervisions(transform_fn) for cut in self)

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


def make_windowed_cuts_from_features(
    feature_set: FeatureSet,
    cut_duration: Seconds,
    cut_shift: Optional[Seconds] = None,
    keep_shorter_windows: bool = False,
) -> CutSet:
    """
    Converts a FeatureSet to a CutSet by traversing each Features object in - possibly overlapping - windows, and
    creating a MonoCut out of that area. By default, the last window in traversal will be discarded if it cannot satisfy
    the `cut_duration` requirement.

    :param feature_set: a FeatureSet object.
    :param cut_duration: float, duration of created Cuts in seconds.
    :param cut_shift: optional float, specifies how many seconds are in between the starts of consecutive windows.
        Equals `cut_duration` by default.
    :param keep_shorter_windows: bool, when True, the last window will be used to create a MonoCut even if
        its duration is shorter than `cut_duration`.
    :return: a CutSet object.
    """
    if cut_shift is None:
        cut_shift = cut_duration
    round_fn = ceil if keep_shorter_windows else floor
    cuts = []
    for features in feature_set:
        # Determine the number of cuts, depending on `keep_shorter_windows` argument.
        # When its true, we'll want to include the residuals in the output; otherwise,
        # we provide only full duration cuts.
        n_cuts = round_fn(features.duration / cut_shift)
        if (
            (n_cuts - 1) * cut_shift + cut_duration > features.duration
            and not keep_shorter_windows
        ):
            n_cuts -= 1
        for idx in range(n_cuts):
            offset = features.start + idx * cut_shift
            duration = min(cut_duration, features.end - offset)
            cuts.append(
                MonoCut(
                    id=str(uuid4()),
                    start=offset,
                    duration=duration,
                    channel=features.channels,
                    features=features,
                    supervisions=[],
                )
            )
    return CutSet.from_cuts(cuts)


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
    elif isinstance(reference_cut, (MonoCut, PaddingCut)):
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
    elif isinstance(mixed_in_cut, (MonoCut, PaddingCut)):
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

    :param cut: MonoCut to be padded.
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
    if feat_ok:
        # Case I: Features are provided.
        # Use features to determine the cut boundaries and attach recordings and supervisions as available.
        cuts = CutSet.from_cuts(
            MonoCut(
                id=str(uuid4())
                if random_ids
                else f"{feats.recording_id}-{idx}-{feats.channels}",
                start=feats.start,
                duration=feats.duration,
                channel=feats.channels,
                features=feats,
                recording=recordings[feats.recording_id] if rec_ok else None,
                # The supervisions' start times are adjusted if the features object starts at time other than 0s.
                supervisions=list(
                    supervisions.find(
                        recording_id=feats.recording_id,
                        channel=feats.channels,
                        start_after=feats.start,
                        end_before=feats.end,
                        adjust_offset=True,
                    )
                )
                if sup_ok
                else [],
            )
            for idx, feats in enumerate(features)
        )
    else:
        # Case II: Recordings are provided (and features are not).
        # Use recordings to determine the cut boundaries.
        cuts = CutSet.from_cuts(
            MonoCut(
                id=str(uuid4()) if random_ids else f"{recording.id}-{ridx}-{cidx}",
                start=0,
                duration=recording.duration,
                channel=channel,
                recording=recording,
                supervisions=list(
                    supervisions.find(recording_id=recording.id, channel=channel)
                )
                if sup_ok
                else [],
            )
            for ridx, recording in enumerate(recordings)
            # A single cut always represents a single channel. When a recording has multiple channels,
            # we create a new cut for each channel separately.
            for cidx, channel in enumerate(recording.channel_ids)
        )
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
                cut = MonoCut(
                    id=str(uuid4())
                    if random_ids
                    else f"{feats.recording_id}-{idx}-{feats.channels}",
                    start=feats.start,
                    duration=feats.duration,
                    channel=feats.channels,
                    features=feats,
                    recording=rec,
                    # The supervisions' start times are adjusted if the features object starts at time other than 0s.
                    supervisions=list(
                        sups.find(
                            recording_id=feats.recording_id,
                            channel=feats.channels,
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

            # A single cut always represents a single channel. When a recording has multiple channels,
            # we create a new cut for each channel separately.
            for cidx, channel in enumerate(recording.channel_ids):
                cut = MonoCut(
                    id=str(uuid4()) if random_ids else f"{recording.id}-{ridx}-{cidx}",
                    start=0,
                    duration=recording.duration,
                    channel=channel,
                    recording=recording,
                    supervisions=list(
                        sups.find(recording_id=recording.id, channel=channel)
                    )
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


def merge_supervisions(
    cut: Cut, custom_merge_fn: Optional[Callable[[str, Iterable[Any]], Any]] = None
) -> Cut:
    """
    Return a copy of the cut that has all of its supervisions merged into
    a single segment.

    The new start is the start of the earliest superivion, and the new duration
    is a minimum spanning duration for all the supervisions.

    The text fields are concatenated with a whitespace, and all other string fields
    (including IDs) are prefixed with "cat#" and concatenated with a hash symbol "#".
    This is also applied to ``custom`` fields. Fields with a ``None`` value are omitted.

    .. note:: If you're using individual tracks of a :class:`MixedCut`, note that this transform
         drops all the supervisions in individual tracks and assigns the merged supervision
         in the first :class:`.MonoCut` found in ``self.tracks``.

    :param custom_merge_fn: a function that will be called to merge custom fields values.
        We expect ``custom_merge_fn`` to handle all possible custom keys.
        When not provided, we will treat all custom values as strings.
        It will be called roughly like:
        ``custom_merge_fn(custom_key, [s.custom[custom_key] for s in sups])``
    """
    # "m" stands for merged in variable names below

    def merge(values: Iterable[str]) -> Optional[str]:
        # e.g.
        # values = ["1125-76840-0001", "1125-53670-0003"]
        # return "cat#1125-76840-0001#1125-53670-0003"
        values = list(values)
        if len(values) == 0:
            return None
        if len(values) == 1:
            return values[0]
        return "#".join(chain(["cat"], values))

    if custom_merge_fn is not None:
        # Merge custom fields with the user-provided function.
        merge_custom = custom_merge_fn
    else:
        # Merge the string representations of custom fields.
        merge_custom = lambda k, vs: merge(map(str, vs))

    if isinstance(cut, PaddingCut):
        return cut

    sups = sorted(cut.supervisions, key=lambda s: s.start)

    if len(sups) <= 1:
        return cut

    # the sampling rate is arbitrary, ensures there are no float precision errors
    mstart = sups[0].start
    mend = sups[-1].end
    mduration = add_durations(mend, -mstart, sampling_rate=cut.sampling_rate)

    custom_keys = set(k for s in sups if s.custom is not None for k in s.custom.keys())
    alignment_keys = set(
        k for s in sups if s.alignment is not None for k in s.alignment.keys()
    )

    if any(overlaps(s1, s2) for s1, s2 in zip(sups, sups[1:])) and any(
        s.text is not None for s in sups
    ):
        warnings.warn(
            "You are merging overlapping supervisions that have text transcripts. "
            "The result is likely to be unusable if you are going to train speech "
            f"recognition models (cut id: {cut.id})."
        )

    is_mixed = isinstance(cut, MixedCut)

    msup = SupervisionSegment(
        id=merge(s.id for s in sups),
        # For MixedCut, make merged recording_id is a mix of recording_ids.
        # For MonoCut, the recording_id is always the same.
        recording_id=merge(s.recording_id for s in sups)
        if is_mixed
        else sups[0].recording_id,
        start=mstart,
        duration=mduration,
        # For MixedCut, hardcode -1 to indicate no specific channel,
        # as the supervisions might have come from different channels
        # in their original recordings.
        # For MonoCut, the channel is always the same.
        channel=-1 if is_mixed else sups[0].channel,
        text=" ".join(s.text for s in sups if s.text),
        speaker=merge(s.speaker for s in sups if s.speaker),
        language=merge(s.language for s in sups if s.language),
        gender=merge(s.gender for s in sups if s.gender),
        custom={
            k: merge_custom(
                k, (s.custom[k] for s in sups if s.custom is not None and k in s.custom)
            )
            for k in custom_keys
        },
        alignment={
            # Concatenate the lists of alignment units.
            k: reduce(
                add,
                (
                    s.alignment[k]
                    for s in sups
                    if s.alignment is not None and k in s.alignment
                ),
            )
            for k in alignment_keys
        },
    )

    if is_mixed:
        new_cut = cut.drop_supervisions()
        new_cut._first_non_padding_cut.supervisions = [msup]
        return new_cut
    else:
        return fastcopy(cut, supervisions=[msup])


def _cut_into_windows_single(
    cuts: CutSet, duration, hop, keep_excessive_supervisions
) -> CutSet:
    return cuts.cut_into_windows(
        duration=duration,
        hop=hop,
        keep_excessive_supervisions=keep_excessive_supervisions,
    ).to_eager()
