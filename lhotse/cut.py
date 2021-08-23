import logging
import random
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial, reduce
from itertools import chain, islice
from math import ceil, floor
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Type, TypeVar, \
    Union

import numpy as np
from cytoolz import sliding_window
from cytoolz.itertoolz import groupby
from intervaltree import Interval, IntervalTree
from tqdm.auto import tqdm
from typing_extensions import Literal

from lhotse.audio import AudioMixer, AudioSource, Recording, RecordingSet
from lhotse.augmentation import AugmentFn
from lhotse.features import FeatureExtractor, FeatureMixer, FeatureSet, Features, create_default_feature_extractor
from lhotse.features.base import compute_global_stats
from lhotse.features.io import FeaturesWriter, LilcomFilesWriter, LilcomHdf5Writer
from lhotse.serialization import Serializable
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (Decibels, LOG_EPSILON, NonPositiveEnergyError, Pathlike, Seconds, SetContainingAnything,
                          TimeSpan, asdict_nonull,
                          compute_num_frames, compute_num_samples, compute_start_duration_for_extended_cut,
                          exactly_one_not_null, fastcopy,
                          ifnone, index_by_id_and_check, measure_overlap, overlaps,
                          overspans, perturb_num_samples, split_sequence, uuid4)

# One of the design principles for Cuts is a maximally "lazy" implementation, e.g. when mixing Cuts,
# we'd rather sum the feature matrices only after somebody actually calls "load_features". It helps to avoid
# an excessive storage size for data augmented in various ways.


FW = TypeVar('FW', bound=FeaturesWriter)


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

        >>> from lhotse import LilcomHdf5Writer
        >>> with LilcomHdf5Writer('feats.h5') as storage:
        ...     cut_with_feats = cut.compute_and_store_features(
        ...         extractor=Fbank(),
        ...         storage=storage
        ...     )

    Cuts have several methods that allow their manipulation, transformation, and mixing.
    Some examples (see the respective methods documentation for details)::

        >>> cut_2_to_4s = cut.truncate(offset=2, duration=2)
        >>> cut_padded = cut.pad(duration=10.0)
        >>> cut_mixed = cut.mix(other_cut, offset_other_by=5.0, snr=20)
        >>> cut_append = cut.append(other_cut)
        >>> cut_24k = cut.resample(24000)
        >>> cut_sp = cut.perturb_speed(1.1)
        >>> cut_vp = cut.perturb_volume(2.)

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
    from_dict: Callable[[Dict], 'Cut']
    load_audio: Callable[[], np.ndarray]
    load_features: Callable[[], np.ndarray]
    compute_and_store_features: Callable
    drop_features: Callable
    drop_recording: Callable
    drop_supervisions: Callable
    truncate: Callable
    pad: Callable
    resample: Callable
    perturb_speed: Callable
    perturb_tempo: Callable
    perturb_volume: Callable
    map_supervisions: Callable
    filter_supervisions: Callable
    with_features_path_prefix: Callable
    with_recording_path_prefix: Callable

    def to_dict(self) -> dict:
        d = asdict_nonull(self)
        return {**d, 'type': type(self).__name__}

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

    def mix(self, other: 'Cut', offset_other_by: Seconds = 0.0, snr: Optional[Decibels] = None) -> 'MixedCut':
        """Refer to :function:`~lhotse.cut.mix` documentation."""
        return mix(self, other, offset=offset_other_by, snr=snr)

    def append(self, other: 'Cut', snr: Optional[Decibels] = None) -> 'MixedCut':
        """
        Append the ``other`` Cut after the current Cut. Conceptually the same as ``mix`` but with an offset
        matching the current cuts length. Optionally scale down (positive SNR) or scale up (negative SNR)
        the ``other`` cut.
        Returns a MixedCut, which only keeps the information about the mix; actual mixing is performed
        during the call to ``load_features``.
        """
        return mix(self, other, offset=self.duration, snr=snr)

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
            ax.axvspan(supervision.start, supervision.end, color='green', alpha=0.1)
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

    def trim_to_supervisions(
            self,
            keep_overlapping: bool = True,
            min_duration: Optional[Seconds] = None,
            context_direction: Literal['center', 'left', 'right', 'random'] = 'center',
    ) -> List['Cut']:
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
                    direction=context_direction
                )
            cuts.append(
                self.truncate(
                    offset=new_start,
                    duration=new_duration,
                    keep_excessive_supervisions=keep_overlapping,
                    _supervisions_index=supervisions_index,
                )
            )
        return cuts

    def index_supervisions(
            self,
            index_mixed_tracks: bool = False,
            keep_ids: Optional[Set[str]] = None
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

    def compute_and_store_recording(
            self,
            storage_path: Pathlike,
            augment_fn: Optional[AugmentFn] = None,
    ) -> 'MonoCut':
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
        storage_path = Path(storage_path)
        samples = self.load_audio()
        if augment_fn is not None:
            samples = augment_fn(samples, self.sampling_rate)
        # Store audio as FLAC
        import soundfile as sf
        sf.write(
            file=str(storage_path),
            data=samples.transpose(),
            samplerate=self.sampling_rate,
            format='FLAC'
        )
        recording = Recording(
            id=storage_path.stem,
            sampling_rate=self.sampling_rate,
            num_samples=samples.shape[1],
            duration=samples.shape[1] / self.sampling_rate,
            sources=[
                AudioSource(
                    type='file',
                    channels=[0],
                    source=str(storage_path),
                )
            ]
        )
        return MonoCut(
            id=self.id,
            start=0,
            duration=recording.duration,
            channel=0,
            supervisions=self.supervisions,
            recording=recording
        )

    def speakers_feature_mask(
            self,
            min_speaker_dim: Optional[int] = None,
            speaker_to_idx_map: Optional[Dict[str, int]] = None,
            use_alignment_if_exists: Optional[str] = None
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
        assert self.has_features, f"No features available. " \
                                  f"Can't compute supervisions feature mask for cut with ID: {self.id}."
        if speaker_to_idx_map is None:
            speaker_to_idx_map = {spk: idx for idx, spk in enumerate(sorted(set(s.speaker for s in self.supervisions)))}
        num_speakers = len(speaker_to_idx_map)
        if min_speaker_dim is not None:
            num_speakers = min(min_speaker_dim, num_speakers)
        mask = np.zeros((num_speakers, self.num_frames))
        for supervision in self.supervisions:
            speaker_idx = speaker_to_idx_map[supervision.speaker]
            if use_alignment_if_exists and supervision.alignment and use_alignment_if_exists in supervision.alignment:
                for ali in supervision.alignment[use_alignment_if_exists]:
                    st = round(ali.start / self.frame_shift) if ali.start > 0 else 0
                    et = round(ali.end / self.frame_shift) if ali.end < self.duration else self.num_frames
                    mask[speaker_idx, st:et] = 1
            else:
                st = round(supervision.start / self.frame_shift) if supervision.start > 0 else 0
                et = round(supervision.end / self.frame_shift) if supervision.end < self.duration else self.num_frames
                mask[speaker_idx, st:et] = 1
        return mask

    def speakers_audio_mask(
            self,
            min_speaker_dim: Optional[int] = None,
            speaker_to_idx_map: Optional[Dict[str, int]] = None,
            use_alignment_if_exists: Optional[str] = None
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
        assert self.has_recording, f"No recording available. " \
                                   f"Can't compute supervisions audio mask for cut with ID: {self.id}."
        if speaker_to_idx_map is None:
            speaker_to_idx_map = {spk: idx for idx, spk in enumerate(sorted(set(s.speaker for s in self.supervisions)))}
        num_speakers = len(speaker_to_idx_map)
        if min_speaker_dim is not None:
            num_speakers = min(min_speaker_dim, num_speakers)
        mask = np.zeros((num_speakers, self.num_samples))
        for supervision in self.supervisions:
            speaker_idx = speaker_to_idx_map[supervision.speaker]
            if use_alignment_if_exists and supervision.alignment and use_alignment_if_exists in supervision.alignment:
                for ali in supervision.alignment[use_alignment_if_exists]:
                    st = round(ali.start * self.sampling_rate) if ali.start > 0 else 0
                    et = (
                        round(ali.end * self.sampling_rate)
                        if ali.end < self.duration
                        else self.duration * self.sampling_rate
                    )
                    mask[speaker_idx, st:et] = 1
            else:
                st = round(supervision.start * self.sampling_rate) if supervision.start > 0 else 0
                et = (
                    round(supervision.end * self.sampling_rate)
                    if supervision.end < self.duration
                    else self.duration * self.sampling_rate
                )
                mask[speaker_idx, st:et] = 1
        return mask

    def supervisions_feature_mask(self, use_alignment_if_exists: Optional[str] = None) -> np.ndarray:
        """
        Return a 1D numpy array with value 1 for **frames** covered by at least one supervision,
        and 0 for **frames** not covered by any supervision.

        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        return compute_supervisions_frame_mask(self, use_alignment_if_exists=use_alignment_if_exists)

    def supervisions_audio_mask(self, use_alignment_if_exists: Optional[str] = None) -> np.ndarray:
        """
        Return a 1D numpy array with value 1 for **samples** covered by at least one supervision,
        and 0 for **samples** not covered by any supervision.

        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        assert self.has_recording, f"No recording available. " \
                                   f"Can't compute supervisions audio mask for cut with ID: {self.id}."
        mask = np.zeros(self.num_samples, dtype=np.float32)
        for supervision in self.supervisions:
            if use_alignment_if_exists and supervision.alignment and use_alignment_if_exists in supervision.alignment:
                for ali in supervision.alignment[use_alignment_if_exists]:
                    st = round(ali.start * self.sampling_rate) if ali.start > 0 else 0
                    et = (
                        round(ali.end * self.sampling_rate)
                        if ali.end < self.duration
                        else self.duration * self.sampling_rate
                    )
                    mask[st:et] = 1.0
            else:
                st = round(supervision.start * self.sampling_rate) if supervision.start > 0 else 0
                et = (
                    round(supervision.end * self.sampling_rate)
                    if supervision.end < self.duration
                    else self.duration * self.sampling_rate
                )
                mask[st:et] = 1.0
        return mask

    def with_id(self, id_: str) -> 'Cut':
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

    @property
    def recording_id(self) -> str:
        return self.recording.id if self.has_recording else self.features.recording_id

    @property
    def end(self) -> Seconds:
        return round(self.start + self.duration, ndigits=8)

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
        return compute_num_frames(duration=self.duration, frame_shift=self.frame_shift,
                                  sampling_rate=self.sampling_rate) if self.has_features else None

    @property
    def num_samples(self) -> Optional[int]:
        return compute_num_samples(self.duration, self.sampling_rate) if self.has_recording else None

    @property
    def num_features(self) -> Optional[int]:
        return self.features.num_features if self.has_features else None

    @property
    def features_type(self) -> Optional[str]:
        return self.features.type if self.has_features else None

    @property
    def sampling_rate(self) -> int:
        return self.features.sampling_rate if self.has_features else self.recording.sampling_rate

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
                feats = feats[:self.num_frames, :]
            elif feats.shape[0] - self.num_frames == -1:
                feats = np.concatenate((feats, feats[-1:, :]), axis=0)
            return feats
        return None

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

    def drop_features(self) -> 'MonoCut':
        """Return a copy of the current :class:`.MonoCut`, detached from ``features``."""
        assert self.has_recording, f"Cannot detach features from a MonoCut with no Recording (cut ID = {self.id})."
        return fastcopy(self, features=None)

    def drop_recording(self) -> 'MonoCut':
        """Return a copy of the current :class:`.MonoCut`, detached from ``recording``."""
        assert self.has_features, f"Cannot detach recording from a MonoCut with no Features (cut ID = {self.id})."
        return fastcopy(self, recording=None)

    def drop_supervisions(self) -> 'MonoCut':
        """Return a copy of the current :class:`.MonoCut`, detached from ``supervisions``."""
        return fastcopy(self, supervisions=[])

    def compute_and_store_features(
            self,
            extractor: FeatureExtractor,
            storage: FeaturesWriter,
            augment_fn: Optional[AugmentFn] = None,
            *args,
            **kwargs
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
            _supervisions_index: Optional[Dict[str, IntervalTree]] = None
    ) -> 'MonoCut':
        """
        Returns a new MonoCut that is a sub-region of the current MonoCut.

        Note that no operation is done on the actual features or recording -
        it's only during the call to :meth:`MonoCut.load_features` / :meth:`MonoCut.load_audio`
        when the actual changes happen (a subset of features/audio is loaded).

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
        # Note: technically, truncate's code can be used for "expanding" the cut as well:
        #       In that case, we must ensure that the start of MonoCut is not before the start
        #       of the actual Recording, hence max(..., 0).
        new_start = max(self.start + offset, 0)
        until = offset + (duration if duration is not None else self.duration)
        new_duration = self.duration - new_start if duration is None else until - offset
        assert new_duration > 0.0
        duration_past_end = (new_start + new_duration) - (self.start + self.duration)
        if duration_past_end > 0:
            # When the end of the MonoCut has been exceeded, trim the new duration to not exceed the old MonoCut's end.
            new_duration -= duration_past_end
        # Round the duration to avoid the possible loss of a single audio sample due to floating point
        # additions and subtractions.
        new_duration = round(new_duration, ndigits=8)

        if _supervisions_index is None:
            criterion = overlaps if keep_excessive_supervisions else overspans
            new_time_span = TimeSpan(start=0, end=new_duration)
            new_supervisions = (segment.with_offset(-offset) for segment in self.supervisions)
            supervisions = [
                segment for segment in new_supervisions if criterion(new_time_span, segment)
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
                intervals = tree.envelop(begin=offset - 1e-3, end=offset + new_duration + 1e-3)
            supervisions = []
            for interval in intervals:
                # We are going to measure the overlap ratio of the supervision with the "truncated" cut
                # and reject segments that overlap less than 1%. This way we can avoid quirks and errors
                # of limited float precision.
                olap_ratio = measure_overlap(interval.data, TimeSpan(offset, offset + new_duration))
                if olap_ratio > 0.01:
                    supervisions.append(interval.data.with_offset(-offset))

        return MonoCut(
            id=self.id if preserve_id else str(uuid4()),
            start=new_start,
            duration=new_duration,
            channel=self.channel,
            supervisions=sorted(supervisions, key=lambda s: s.start),
            features=self.features,
            recording=self.recording
        )

    def pad(
            self,
            duration: Seconds = None,
            num_frames: int = None,
            num_samples: int = None,
            pad_feat_value: float = LOG_EPSILON,
            direction: str = 'right'
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
        :return: a padded MixedCut if duration is greater than this cut's duration, otherwise ``self``.
        """
        return pad(
            self,
            duration=duration,
            num_frames=num_frames,
            num_samples=num_samples,
            pad_feat_value=pad_feat_value,
            direction=direction
        )

    def resample(self, sampling_rate: int, affix_id: bool = False) -> 'MonoCut':
        """
        Return a new ``MonoCut`` that will lazily resample the audio while reading it.
        This operation will drop the feature manifest, if attached.
        It does not affect the supervision.

        :param sampling_rate: The new sampling rate.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the current ``MonoCut``.
        """
        assert self.has_recording, 'Cannot resample a MonoCut without Recording.'
        return fastcopy(
            self,
            id=f'{self.id}_rs{sampling_rate}' if affix_id else self.id,
            recording=self.recording.resample(sampling_rate),
            features=None,
        )

    def perturb_speed(self, factor: float, affix_id: bool = True) -> 'MonoCut':
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
        assert self.has_recording, 'Cannot perturb speed on a MonoCut without Recording.'
        if self.has_features:
            logging.warning(
                'Attempting to perturb speed on a MonoCut that references pre-computed features. '
                'The feature manifest will be detached, as we do not support feature-domain '
                'speed perturbation.'
            )
            self.features = None
        # Actual audio perturbation.
        recording_sp = self.recording.perturb_speed(factor=factor, affix_id=affix_id)
        # Match the supervision's start and duration to the perturbed audio.
        # Since SupervisionSegment "start" is relative to the MonoCut's, it's okay (and necessary)
        # to perturb it as well.
        supervisions_sp = [
            s.perturb_speed(factor=factor, sampling_rate=self.sampling_rate, affix_id=affix_id)
            for s in self.supervisions
        ]
        # New start and duration have to be computed through num_samples to be accurate
        start_samples = perturb_num_samples(compute_num_samples(self.start, self.sampling_rate), factor)
        new_start = start_samples / self.sampling_rate
        new_num_samples = perturb_num_samples(self.num_samples, factor)
        new_duration = new_num_samples / self.sampling_rate
        return fastcopy(
            self,
            id=f'{self.id}_sp{factor}' if affix_id else self.id,
            recording=recording_sp,
            supervisions=supervisions_sp,
            duration=new_duration,
            start=new_start
        )

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> 'MonoCut':
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
        assert self.has_recording, 'Cannot perturb speed on a MonoCut without Recording.'
        if self.has_features:
            logging.warning(
                'Attempting to perturb tempo on a MonoCut that references pre-computed features. '
                'The feature manifest will be detached, as we do not support feature-domain '
                'speed perturbation.'
            )
            self.features = None
        # Actual audio perturbation.
        recording_sp = self.recording.perturb_tempo(factor=factor, affix_id=affix_id)
        # Match the supervision's start and duration to the perturbed audio.
        # Since SupervisionSegment "start" is relative to the MonoCut's, it's okay (and necessary)
        # to perturb it as well.
        supervisions_sp = [
            s.perturb_tempo(factor=factor, sampling_rate=self.sampling_rate, affix_id=affix_id)
            for s in self.supervisions
        ]
        # New start and duration have to be computed through num_samples to be accurate
        start_samples = perturb_num_samples(compute_num_samples(self.start, self.sampling_rate), factor)
        new_start = start_samples / self.sampling_rate
        new_num_samples = perturb_num_samples(self.num_samples, factor)
        new_duration = new_num_samples / self.sampling_rate
        return fastcopy(
            self,
            id=f'{self.id}_tp{factor}' if affix_id else self.id,
            recording=recording_sp,
            supervisions=supervisions_sp,
            duration=new_duration,
            start=new_start
        )

    def perturb_volume(self, factor: float, affix_id: bool = True) -> 'MonoCut':
        """
        Return a new ``MonoCut`` that will lazily perturb the volume while loading audio.

        :param factor: The volume will be adjusted this many times (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``MonoCut.id`` field
            by affixing it with "_vp{factor}".
        :return: a modified copy of the current ``MonoCut``.
        """
        # Pre-conditions
        assert self.has_recording, 'Cannot perturb volume on a MonoCut without Recording.'
        if self.has_features:
            logging.warning(
                'Attempting to perturb volume on a MonoCut that references pre-computed features. '
                'The feature manifest will be detached, as we do not support feature-domain '
                'volume perturbation.'
            )
            self.features = None
        # Actual audio perturbation.
        recording_vp = self.recording.perturb_volume(factor=factor, affix_id=affix_id)
        # Match the supervision's id (and it's underlying recording id).
        supervisions_vp = [s.perturb_volume(factor=factor, affix_id=affix_id) for s in self.supervisions]

        return fastcopy(
            self,
            id=f'{self.id}_vp{factor}' if affix_id else self.id,
            recording=recording_vp,
            supervisions=supervisions_vp
        )

    def map_supervisions(self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]) -> Cut:
        """
        Modify the SupervisionSegments by `transform_fn` of this MonoCut.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a modified MonoCut.
        """
        new_cut = fastcopy(self, supervisions=[s.map(transform_fn) for s in self.supervisions])
        return new_cut

    def filter_supervisions(self, predicate: Callable[[SupervisionSegment], bool]) -> Cut:
        """
        Modify cut to store only supervisions accepted by `predicate`

        Example:
            >>> cut = cut.filter_supervisions(lambda s: s.id in supervision_ids)
            >>> cut = cut.filter_supervisions(lambda s: s.duration < 5.0)
            >>> cut = cut.filter_supervisions(lambda s: s.text is not None)

        :param predicate: A callable that accepts `SupervisionSegment` and returns bool
        :return: a modified MonoCut
        """
        new_cut = fastcopy(self, supervisions=[s for s in self.supervisions if predicate(s)])
        return new_cut

    @staticmethod
    def from_dict(data: dict) -> 'MonoCut':
        features = Features.from_dict(data.pop('features')) if 'features' in data else None
        recording = Recording.from_dict(data.pop('recording')) if 'recording' in data else None
        supervision_infos = data.pop('supervisions') if 'supervisions' in data else []
        return MonoCut(
            **data,
            features=features,
            recording=recording,
            supervisions=[SupervisionSegment.from_dict(s) for s in supervision_infos]
        )

    def with_features_path_prefix(self, path: Pathlike) -> 'MonoCut':
        if not self.has_features:
            return self
        return fastcopy(self, features=self.features.with_path_prefix(path))

    def with_recording_path_prefix(self, path: Pathlike) -> 'MonoCut':
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

    @property
    def start(self) -> Seconds:
        return 0

    @property
    def end(self) -> Seconds:
        return self.duration

    @property
    def supervisions(self):
        return []

    @property
    def has_features(self) -> bool:
        return self.num_frames is not None

    @property
    def has_recording(self) -> bool:
        return self.num_samples is not None

    # noinspection PyUnusedLocal
    def load_features(self, *args, **kwargs) -> Optional[np.ndarray]:
        if self.has_features:
            return np.ones((self.num_frames, self.num_features), np.float32) * self.feat_value
        return None

    # noinspection PyUnusedLocal
    def load_audio(self, *args, **kwargs) -> Optional[np.ndarray]:
        if self.has_recording:
            return np.zeros((1, compute_num_samples(self.duration, self.sampling_rate)), np.float32)
        return None

    # noinspection PyUnusedLocal
    def truncate(
            self,
            *,
            offset: Seconds = 0.0,
            duration: Optional[Seconds] = None,
            keep_excessive_supervisions: bool = True,
            preserve_id: bool = False,
            **kwargs
    ) -> 'PaddingCut':
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
                sampling_rate=self.sampling_rate
            ) if self.num_frames is not None else None,
            num_samples=compute_num_samples(
                duration=new_duration,
                sampling_rate=self.sampling_rate
            ) if self.num_samples is not None else None,
        )

    def pad(
            self,
            duration: Seconds = None,
            num_frames: int = None,
            num_samples: int = None,
            pad_feat_value: float = LOG_EPSILON,
            direction: str = 'right'
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
        :return: a padded MixedCut if duration is greater than this cut's duration, otherwise ``self``.
        """
        return pad(
            self,
            duration=duration,
            num_frames=num_frames,
            num_samples=num_samples,
            pad_feat_value=pad_feat_value,
            direction=direction
        )

    def resample(self, sampling_rate: int, affix_id: bool = False) -> 'PaddingCut':
        """
        Return a new ``MonoCut`` that will lazily resample the audio while reading it.
        This operation will drop the feature manifest, if attached.
        It does not affect the supervision.

        :param sampling_rate: The new sampling rate.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the current ``MonoCut``.
        """
        assert self.has_recording, 'Cannot resample a MonoCut without Recording.'
        return fastcopy(
            self,
            id=f'{self.id}_rs{sampling_rate}' if affix_id else self.id,
            sampling_rate=sampling_rate,
            num_samples=compute_num_samples(self.duration, sampling_rate),
            num_frames=None,
            num_features=None,
            frame_shift=None
        )

    def perturb_speed(self, factor: float, affix_id: bool = True) -> 'PaddingCut':
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
                'Attempting to perturb speed on a MonoCut that references pre-computed features. '
                'The feature manifest will be detached, as we do not support feature-domain '
                'speed perturbation.'
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
            id=f'{self.id}_sp{factor}' if affix_id else self.id,
            num_samples=new_num_samples,
            duration=new_duration,
            num_frames=new_num_frames,
            num_features=new_num_features,
            frame_shift=new_frame_shift
        )

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> 'PaddingCut':
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
                'Attempting to perturb tempo on a MonoCut that references pre-computed features. '
                'The feature manifest will be detached, as we do not support feature-domain '
                'tempo perturbation.'
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
            id=f'{self.id}_tp{factor}' if affix_id else self.id,
            num_samples=new_num_samples,
            duration=new_duration,
            num_frames=new_num_frames,
            num_features=new_num_features,
            frame_shift=new_frame_shift
        )

    def perturb_volume(self, factor: float, affix_id: bool = True) -> 'PaddingCut':
        """
        Return a new ``PaddingCut`` that will "mimic" the effect of volume perturbation
        on amplitude of samples.

        :param factor: The volume will be adjusted this many times (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``PaddingCut.id`` field
            by affixing it with "_vp{factor}".
        :return: a modified copy of the current ``PaddingCut``.
        """

        return fastcopy(self, id=f'{self.id}_vp{factor}' if affix_id else self.id)

    def drop_features(self) -> 'PaddingCut':
        """Return a copy of the current :class:`.PaddingCut`, detached from ``features``."""
        assert self.has_recording, f"Cannot detach features from a MonoCut with no Recording (cut ID = {self.id})."
        return fastcopy(self, num_frames=None, num_features=None, frame_shift=None)

    def drop_recording(self) -> 'PaddingCut':
        """Return a copy of the current :class:`.PaddingCut`, detached from ``recording``."""
        assert self.has_features, f"Cannot detach recording from a PaddingCut with no Features (cut ID = {self.id})."
        return fastcopy(self, num_samples=None)

    def drop_supervisions(self) -> 'PaddingCut':
        """Return a copy of the current :class:`.PaddingCut`, detached from ``supervisions``."""
        return self

    def compute_and_store_features(self, extractor: FeatureExtractor, *args, **kwargs) -> Cut:
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
                sampling_rate=self.sampling_rate
            ),
            frame_shift=extractor.frame_shift
        )

    def map_supervisions(self, transform_fn: Callable[[Any], Any]) -> Cut:
        """
        Just for consistency with `MonoCut` and `MixedCut`.

        :param transform_fn: a dummy function that would be never called actually.
        :return: the PaddingCut itself.
        """
        return self

    def filter_supervisions(self, predicate: Callable[[SupervisionSegment], bool]) -> Cut:
        """
        Just for consistency with `MonoCut` and `MixedCut`.

        :param predicate: A callable that accepts `SupervisionSegment` and returns bool
        :return: a modified MonoCut
        """
        return self

    @staticmethod
    def from_dict(data: dict) -> 'PaddingCut':
        return PaddingCut(**data)

    def with_features_path_prefix(self, path: Pathlike) -> 'PaddingCut':
        return self

    def with_recording_path_prefix(self, path: Pathlike) -> 'PaddingCut':
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
        raw_cut = data.pop('cut')
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
    def end(self) -> Seconds:
        return self.duration

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
                sampling_rate=self.sampling_rate
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

    def truncate(
            self,
            *,
            offset: Seconds = 0.0,
            duration: Optional[Seconds] = None,
            keep_excessive_supervisions: bool = True,
            preserve_id: bool = False,
            _supervisions_index: Optional[Dict[str, IntervalTree]] = None
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

        new_tracks = []
        old_duration = self.duration
        new_mix_end = old_duration - offset if duration is None else offset + duration

        for track in sorted(self.tracks, key=lambda t: t.offset):
            # First, determine how much of the beginning of the current track we're going to truncate:
            # when the track offset is larger than the truncation offset, we are not truncating the cut;
            # just decreasing the track offset.

            # 'cut_offset' determines how much we're going to truncate the Cut for the current track.
            cut_offset = max(offset - track.offset, 0)
            # 'track_offset' determines the new track's offset after truncation.
            track_offset = max(track.offset - offset, 0)
            # 'track_end' is expressed relative to the beginning of the mix
            # (not to be confused with the 'start' of the underlying MonoCut)
            track_end = track.offset + track.cut.duration

            if track_end < offset:
                # Omit a MonoCut that ends before the truncation offset.
                continue

            cut_duration_decrease = 0
            if track_end > new_mix_end:
                if duration is not None:
                    cut_duration_decrease = track_end - new_mix_end
                else:
                    cut_duration_decrease = track_end - old_duration

            # Compute the new MonoCut's duration after trimming the start and the end.
            new_duration = track.cut.duration - cut_offset - cut_duration_decrease
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
                        _supervisions_index=_supervisions_index
                    ),
                    offset=track_offset,
                    snr=track.snr
                )
            )
        if len(new_tracks) == 1:
            # The truncation resulted in just a single cut - simply return it.
            return new_tracks[0].cut
        return MixedCut(id=str(uuid4()), tracks=new_tracks)

    def pad(
            self,
            duration: Seconds = None,
            num_frames: int = None,
            num_samples: int = None,
            pad_feat_value: float = LOG_EPSILON,
            direction: str = 'right'
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
        :return: a padded MixedCut if duration is greater than this cut's duration, otherwise ``self``.
        """
        return pad(
            self,
            duration=duration,
            num_frames=num_frames,
            num_samples=num_samples,
            pad_feat_value=pad_feat_value,
            direction=direction
        )

    def resample(self, sampling_rate: int, affix_id: bool = False) -> 'MixedCut':
        """
        Return a new ``MixedCut`` that will lazily resample the audio while reading it.
        This operation will drop the feature manifest, if attached.
        It does not affect the supervision.

        :param sampling_rate: The new sampling rate.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the current ``MixedCut``.
        """
        assert self.has_recording, 'Cannot resample a MixedCut without Recording.'
        return MixedCut(
            id=f'{self.id}_rs{sampling_rate}' if affix_id else self.id,
            tracks=[
                fastcopy(t, cut=t.cut.resample(sampling_rate))
                for t in self.tracks
            ]
        )

    def perturb_speed(self, factor: float, affix_id: bool = True) -> 'MixedCut':
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
        assert self.has_recording, 'Cannot perturb speed on a MonoCut without Recording.'
        if self.has_features:
            logging.warning(
                'Attempting to perturb speed on a MixedCut that references pre-computed features. '
                'The feature manifest(s) will be detached, as we do not support feature-domain '
                'speed perturbation.'
            )
        return MixedCut(
            id=f'{self.id}_sp{factor}' if affix_id else self.id,
            tracks=[
                MixTrack(
                    cut=track.cut.perturb_speed(factor=factor, affix_id=affix_id),
                    offset=round(
                        perturb_num_samples(
                            num_samples=compute_num_samples(track.offset, self.sampling_rate),
                            factor=factor
                        ) / self.sampling_rate,
                        ndigits=8
                    ),
                    snr=track.snr
                )
                for track in self.tracks
            ]
        )

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> 'MixedCut':
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
        assert self.has_recording, 'Cannot perturb tempo on a MonoCut without Recording.'
        if self.has_features:
            logging.warning(
                'Attempting to perturb tempo on a MixedCut that references pre-computed features. '
                'The feature manifest(s) will be detached, as we do not support feature-domain '
                'tempo perturbation.'
            )
        return MixedCut(
            id=f'{self.id}_tp{factor}' if affix_id else self.id,
            tracks=[
                MixTrack(
                    cut=track.cut.perturb_tempo(factor=factor, affix_id=affix_id),
                    offset=round(
                        perturb_num_samples(
                            num_samples=compute_num_samples(track.offset, self.sampling_rate),
                            factor=factor
                        ) / self.sampling_rate,
                        ndigits=8
                    ),
                    snr=track.snr
                )
                for track in self.tracks
            ]
        )

    def perturb_volume(self, factor: float, affix_id: bool = True) -> 'MixedCut':
        """
        Return a new ``MixedCut`` that will lazily perturb the volume while loading audio.
        Recordings of the underlying Cuts are updated to reflect volume change.

        :param factor: The volume will be adjusted this many times (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``MixedCut.id`` field
            by affixing it with "_vp{factor}".
        :return: a modified copy of the current ``MixedCut``.
        """
        # Pre-conditions
        assert self.has_recording, 'Cannot perturb volume on a MonoCut without Recording.'
        if self.has_features:
            logging.warning(
                'Attempting to perturb volume on a MixedCut that references pre-computed features. '
                'The feature manifest(s) will be detached, as we do not support feature-domain '
                'volume perturbation.'
            )
        return MixedCut(
            id=f'{self.id}_vp{factor}' if affix_id else self.id,
            tracks=[
                fastcopy(track, cut=track.cut.perturb_volume(factor=factor, affix_id=affix_id))
                for track in self.tracks
            ]
        )

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
            feats[:first_cut.num_frames, :] = first_cut.load_features()
            return feats
        # When there is more than one "regular" cut, we will perform an actual mix.
        mixer = FeatureMixer(
            feature_extractor=create_default_feature_extractor(self._first_non_padding_cut.features.type),
            base_feats=first_cut.load_features(),
            frame_shift=first_cut.frame_shift,
        )
        for track in self.tracks[1:]:
            try:
                mixer.add_to_mix(
                    feats=track.cut.load_features(),
                    snr=track.snr,
                    offset=track.offset,
                    sampling_rate=track.cut.sampling_rate
                )
            except NonPositiveEnergyError as e:
                logging.warning(str(e) + f' MonoCut with id "{track.cut.id}" will not be mixed in.')
        if mixed:
            feats = mixer.mixed_feats
            # Note: The slicing below is a work-around for an edge-case
            #  when two cuts have durations that ended with 0.005 (e.g. 10.125 and 5.715)
            #  - then, the feature extractor "squeezed in" a last extra frame and the simple
            #  relationship between num_frames and duration we strived for is not true;
            #  i.e. the duration is 10.125 + 5.715 = 15.84, but the number of frames is
            #  1013 + 572 = 1585. If the frame_shift is 0.01, we have gained an extra 0.01s...
            if feats.shape[0] - self.num_frames == 1:
                feats = feats[:self.num_frames, :]
            # TODO(pzelasko): This can sometimes happen in a MixedCut with >= 5 different Cuts,
            #   with a regular MonoCut at the end, when the mix offsets are floats with a lot of decimals.
            #   For now we're duplicating the last frame to match the declared "num_frames" of this cut.
            if feats.shape[0] - self.num_frames == -1:
                feats = np.concatenate((feats, feats[-1:, :]), axis=0)
            assert feats.shape[0] == self.num_frames, "Inconsistent number of frames in a MixedCut: please report " \
                                                      "this issue at https://github.com/lhotse-speech/lhotse/issues " \
                                                      "showing the output of print(cut) or str(cut) on which" \
                                                      "load_features() was called."
            return feats
        else:
            return mixer.unmixed_feats

    def load_audio(self, mixed: bool = True) -> Optional[np.ndarray]:
        """
        Loads the audios of the source cuts and mix them on-the-fly.

        :param mixed: When True (default), returns a mono mix of the underlying tracks.
            Otherwise returns a numpy array with the number of channels equal to the number of tracks.
        :return: A numpy ndarray with audio samples and with shape ``(num_channels, num_samples)``
        """
        if not self.has_recording:
            return None
        mixer = AudioMixer(self.tracks[0].cut.load_audio(), sampling_rate=self.tracks[0].cut.sampling_rate)
        for track in self.tracks[1:]:
            try:
                mixer.add_to_mix(
                    audio=track.cut.load_audio(),
                    snr=track.snr,
                    offset=track.offset,
                )
            except NonPositiveEnergyError as e:
                logging.warning(str(e) + f' MonoCut with id "{track.cut.id}" will not be mixed in.')
        if mixed:
            # Off-by-one errors can happen during mixing due to imperfect float arithmetic and rounding;
            # we will fix them on-the-fly so that the manifest does not lie about the num_samples.
            audio = mixer.mixed_audio
            if audio.shape[1] - self.num_samples == 1:
                audio = audio[:, :self.num_samples]
            if audio.shape[1] - self.num_samples == -1:
                audio = np.concatenate((audio, audio[:, -1:]), axis=1)
            assert audio.shape[1] == self.num_samples, f"Inconsistent number of samples in a MixedCut: please report " \
                                                       f"this issue at https://github.com/lhotse-speech/lhotse/issues " \
                                                       f"showing the cut below. MixedCut:\n{self}"
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
                ax.axvspan(track.offset + supervision.start, track.offset + supervision.end, color='green', alpha=0.1)
        return axes

    def drop_features(self) -> 'MixedCut':
        """Return a copy of the current :class:`MixedCut`, detached from ``features``."""
        assert self.has_recording, f"Cannot detach features from a MixedCut with no Recording (cut ID = {self.id})."
        return fastcopy(self, tracks=[fastcopy(t, cut=t.cut.drop_features()) for t in self.tracks])

    def drop_recording(self) -> 'MixedCut':
        """Return a copy of the current :class:`.MixedCut`, detached from ``recording``."""
        assert self.has_features, f"Cannot detach recording from a MixedCut with no Features (cut ID = {self.id})."
        return fastcopy(self, tracks=[fastcopy(t, cut=t.cut.drop_recording()) for t in self.tracks])

    def drop_supervisions(self) -> 'MixedCut':
        """Return a copy of the current :class:`.MixedCut`, detached from ``supervisions``."""
        return fastcopy(self, tracks=[fastcopy(t, cut=t.cut.drop_supervisions()) for t in self.tracks])

    def compute_and_store_features(
            self,
            extractor: FeatureExtractor,
            storage: FeaturesWriter,
            augment_fn: Optional[AugmentFn] = None,
            mix_eagerly: bool = True
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
            return MonoCut(
                id=self.id,
                start=0,
                duration=self.duration,
                channel=0,
                supervisions=self.supervisions,
                features=features_info,
                recording=None
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
                    snr=track.snr
                )
                for track in self.tracks
            ]
            return MixedCut(id=self.id, tracks=new_tracks)

    def map_supervisions(self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]) -> Cut:
        """
        Modify the SupervisionSegments by `transform_fn` of this MixedCut.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a modified MixedCut.
        """
        new_mixed_cut = fastcopy(self)
        for track in new_mixed_cut.tracks:
            track.cut.supervisions = [segment.map(transform_fn) for segment in track.cut.supervisions]
        return new_mixed_cut

    def filter_supervisions(self, predicate: Callable[[SupervisionSegment], bool]) -> Cut:
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
            ]
        )
        return new_mixed_cut

    @staticmethod
    def from_dict(data: dict) -> 'MixedCut':
        return MixedCut(id=data['id'], tracks=[MixTrack.from_dict(track) for track in data['tracks']])

    def with_features_path_prefix(self, path: Pathlike) -> 'MixedCut':
        if not self.has_features:
            return self
        return MixedCut(
            id=self.id,
            tracks=[fastcopy(t, cut=t.cut.with_features_path_prefix(path)) for t in self.tracks]
        )

    def with_recording_path_prefix(self, path: Pathlike) -> 'MixedCut':
        if not self.has_recording:
            return self
        return MixedCut(
            id=self.id,
            tracks=[fastcopy(t, cut=t.cut.with_recording_path_prefix(path)) for t in self.tracks]
        )

    @property
    def _first_non_padding_cut(self) -> MonoCut:
        return [t.cut for t in self.tracks if not isinstance(t.cut, PaddingCut)][0]


class CutSet(Serializable, Sequence[Cut]):
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

    .. caution::
        If the :class:`.CutSet` contained :class:`~lhotse.features.base.Features` manifests, they will be
        detached after performing audio augmentations such as :meth:`.CutSet.perturb_speed` or :meth:`.CutSet.resample` or :meth:`.CutSet.perturb_volume`.

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

    def __eq__(self, other: 'CutSet') -> bool:
        return self.cuts == other.cuts

    @property
    def is_lazy(self) -> bool:
        """
        Indicates whether this manifest was opened in lazy (read-on-the-fly) mode or not.
        """
        from lhotse.serialization import LazyJsonlIterator
        return isinstance(self.cuts, LazyJsonlIterator)

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
        return frozenset(supervision.speaker for cut in self for supervision in cut.supervisions)

    @staticmethod
    def from_cuts(cuts: Iterable[Cut]) -> 'CutSet':
        return CutSet(cuts=index_by_id_and_check(cuts))

    @staticmethod
    def from_manifests(
            recordings: Optional[RecordingSet] = None,
            supervisions: Optional[SupervisionSet] = None,
            features: Optional[FeatureSet] = None,
            random_ids: bool = False,
    ) -> 'CutSet':
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
        :param random_ids: boolean, should the cut IDs be randomized. By default, use the recording ID
            with a loop index and a channel idx, i.e. "{recording_id}-{idx}-{channel}")
        :return: a new :class:`.CutSet` instance.
        """
        assert features is not None or recordings is not None, \
            "At least one of 'features' or 'recordings' has to be provided."
        sup_ok, feat_ok, rec_ok = supervisions is not None, features is not None, recordings is not None
        if feat_ok:
            # Case I: Features are provided.
            # Use features to determine the cut boundaries and attach recordings and supervisions as available.
            return CutSet.from_cuts(
                MonoCut(
                    id=str(uuid4()) if random_ids else f'{feats.recording_id}-{idx}-{feats.channels}',
                    start=feats.start,
                    duration=feats.duration,
                    channel=feats.channels,
                    features=feats,
                    recording=recordings[feats.recording_id] if rec_ok else None,
                    # The supervisions' start times are adjusted if the features object starts at time other than 0s.
                    supervisions=list(supervisions.find(
                        recording_id=feats.recording_id,
                        channel=feats.channels,
                        start_after=feats.start,
                        end_before=feats.end,
                        adjust_offset=True
                    )) if sup_ok else []
                )
                for idx, feats in enumerate(features)
            )
        # Case II: Recordings are provided (and features are not).
        # Use recordings to determine the cut boundaries.
        return CutSet.from_cuts(
            MonoCut(
                id=str(uuid4()) if random_ids else f'{recording.id}-{ridx}-{cidx}',
                start=0,
                duration=recording.duration,
                channel=channel,
                recording=recording,
                supervisions=list(supervisions.find(
                    recording_id=recording.id,
                    channel=channel
                )) if sup_ok else []
            )
            for ridx, recording in enumerate(recordings)
            # A single cut always represents a single channel. When a recording has multiple channels,
            # we create a new cut for each channel separately.
            for cidx, channel in enumerate(recording.channel_ids)
        )

    @staticmethod
    def from_dicts(data: Iterable[dict]) -> 'CutSet':
        def deserialize_one(raw_cut: dict) -> Cut:
            cut_type = raw_cut.pop('type')
            if cut_type == 'MonoCut':
                return MonoCut.from_dict(raw_cut)
            if cut_type == 'Cut':
                warnings.warn(
                    'Your manifest was created with Lhotse version earlier than v0.8, when MonoCut was called Cut. '
                    'Please re-generate it with Lhotse v0.8 as it might stop working in a future version '
                    '(using manifest.from_file() and then manifest.to_file() should be sufficient).')
                return MonoCut.from_dict(raw_cut)
            if cut_type == 'MixedCut':
                return MixedCut.from_dict(raw_cut)
            raise ValueError(f"Unexpected cut type during deserialization: '{cut_type}'")

        return CutSet.from_cuts(deserialize_one(cut) for cut in data)

    def to_dicts(self) -> Iterable[dict]:
        return (cut.to_dict() for cut in self)

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
            max     5415.0
            dtype: float64
        """
        import pandas as pd
        durations = pd.Series([c.duration for c in self])
        speech_durations = pd.Series([s.trim(c.duration).duration for c in self for s in c.supervisions])
        total_sum = durations.sum()
        speech_sum = speech_durations.sum()
        print('Cuts count:', len(self))
        print(f'Total duration (hours): {total_sum / 3600:.1f}')
        print(f'Speech duration (hours): {speech_sum / 3600:.1f} ({speech_sum / total_sum:.1%})')
        print('***')
        print('Duration statistics (seconds):')
        with pd.option_context('precision', 1):
            print(durations.describe().drop('count'))

    def shuffle(self, rng: Optional[random.Random] = None) -> 'CutSet':
        """
        Shuffle the cut IDs in the current :class:`.CutSet` and return a shuffled copy of self.

        :param rng: an optional instance of ``random.Random`` for precise control of randomness.
        :return: a shuffled copy of self.
        """
        if rng is None:
            rng = random
        ids = list(self.ids)
        rng.shuffle(ids)
        return CutSet(cuts={cid: self[cid] for cid in ids})

    def split(self, num_splits: int, shuffle: bool = False, drop_last: bool = False) -> List['CutSet']:
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
            CutSet.from_cuts(subset) for subset in
            split_sequence(self, num_splits=num_splits, shuffle=shuffle, drop_last=drop_last)
        ]

    def subset(
            self,
            *,  # only keyword arguments allowed
            supervision_ids: Optional[Iterable[str]] = None,
            cut_ids: Optional[Iterable[str]] = None,
            first: Optional[int] = None,
            last: Optional[int] = None
    ) -> 'CutSet':
        """
        Return a new ``CutSet`` according to the selected subset criterion.
        Only a single argument to ``subset`` is supported at this time.

        Example:
            >>> cuts = CutSet.from_yaml('path/to/cuts')
            >>> train_set = cuts.subset(supervision_ids=train_ids)
            >>> test_set = cuts.subset(supervision_ids=test_ids)

        :param supervision_ids: List of supervision IDs to keep.
        :param cut_ids: List of cut IDs to keep.
        :param first: int, the number of first cuts to keep.
        :param last: int, the number of last cuts to keep.
        :return: a new ``CutSet`` with the subset results.
        """
        assert exactly_one_not_null(supervision_ids, cut_ids, first, last), "subset() can handle only one non-None arg."

        if first is not None:
            assert first > 0
            if first > len(self):
                logging.warning(f'CutSet has only {len(self)} items but first {first} required; not doing anything.')
                return self
            return CutSet.from_cuts(islice(self, first))

        if last is not None:
            assert last > 0
            if last > len(self):
                logging.warning(f'CutSet has only {len(self)} items but last {last} required; not doing anything.')
                return self
            cut_ids = list(self.ids)[-last:]
            return CutSet.from_cuts(self[cid] for cid in cut_ids)

        if supervision_ids is not None:
            # Remove cuts without supervisions
            supervision_ids = set(supervision_ids)
            return CutSet.from_cuts(
                cut.filter_supervisions(lambda s: s.id in supervision_ids) for cut in self
                if any(s.id in supervision_ids for s in cut.supervisions)
            )

        if cut_ids is not None:
            return CutSet.from_cuts(self[cid] for cid in cut_ids)

    def filter_supervisions(self, predicate: Callable[[SupervisionSegment], bool]) -> 'CutSet':
        """
        Return a new CutSet with Cuts containing only `SupervisionSegments` satisfying `predicate`

        Cuts without supervisions are preserved

        Example:
            >>> cuts = CutSet.from_yaml('path/to/cuts')
            >>> at_least_five_second_supervisions = cuts.filter_supervisions(lambda s: s.duration >= 5)

        :param predicate: A callable that accepts `SupervisionSegment` and returns bool
        :return: a CutSet with filtered supervisions
        """
        return CutSet.from_cuts(
            cut.filter_supervisions(predicate) for cut in self
        )

    def filter(self, predicate: Callable[[Cut], bool]) -> 'CutSet':
        """
        Return a new CutSet with the Cuts that satisfy the `predicate`.

        :param predicate: a function that takes a cut as an argument and returns bool.
        :return: a filtered CutSet.
        """
        return CutSet.from_cuts(cut for cut in self if predicate(cut))

    def trim_to_supervisions(
            self,
            keep_overlapping: bool = True,
            min_duration: Optional[Seconds] = None,
            context_direction: Literal['center', 'left', 'right', 'random'] = 'center',
            num_jobs: int = 1
    ) -> 'CutSet':
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
                        context_direction=context_direction
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
            context_direction=context_direction
        )
        return result

    def trim_to_unsupervised_segments(self) -> 'CutSet':
        """
        Return a new CutSet with Cuts created from segments that have no supervisions (likely
        silence or noise).

        :return: a ``CutSet``.
        """
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

    def mix_same_recording_channels(self) -> 'CutSet':
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
            raise ValueError("This operation is not applicable to CutSet's containing MixedCut's.")
        groups = groupby(lambda cut: (cut.recording.id, cut.start, cut.end), self)
        return CutSet.from_cuts(mix_cuts(cuts) for cuts in groups.values())

    def sort_by_duration(self, ascending: bool = False) -> 'CutSet':
        """
        Sort the CutSet according to cuts duration and return the result. Descending by default.
        """
        return CutSet.from_cuts(sorted(self, key=(lambda cut: cut.duration), reverse=not ascending))

    def sort_like(self, other: 'CutSet') -> 'CutSet':
        """
        Sort the CutSet according to the order of cut IDs in ``other`` and return the result.
        """
        assert set(self.ids) == set(other.ids), "sort_like() expects both CutSet's to have identical cut IDs."
        return CutSet.from_cuts(self[cid] for cid in other.ids)

    def index_supervisions(
            self,
            index_mixed_tracks: bool = False,
            keep_ids: Optional[Set[str]] = None
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
            indexed.update(cut.index_supervisions(index_mixed_tracks=index_mixed_tracks, keep_ids=keep_ids))
        return indexed

    def pad(
            self,
            duration: Seconds = None,
            num_frames: int = None,
            num_samples: int = None,
            pad_feat_value: float = LOG_EPSILON,
            direction: str = 'right'
    ) -> 'CutSet':
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
                direction=direction
            ) for cut in self
        )

    def truncate(
            self,
            max_duration: Seconds,
            offset_type: str,
            keep_excessive_supervisions: bool = True,
            preserve_id: bool = False
    ) -> 'CutSet':
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
        :return: a new CutSet instance with truncated cuts.
        """
        truncated_cuts = []
        for cut in self:
            if cut.duration <= max_duration:
                truncated_cuts.append(cut)
                continue

            def compute_offset():
                if offset_type == 'start':
                    return 0.0
                last_offset = cut.duration - max_duration
                if offset_type == 'end':
                    return last_offset
                if offset_type == 'random':
                    return random.uniform(0.0, last_offset)
                raise ValueError(f"Unknown 'offset_type' option: {offset_type}")

            truncated_cuts.append(cut.truncate(
                offset=compute_offset(),
                duration=max_duration,
                keep_excessive_supervisions=keep_excessive_supervisions,
                preserve_id=preserve_id
            ))
        return CutSet.from_cuts(truncated_cuts)

    def cut_into_windows(
            self,
            duration: Seconds,
            keep_excessive_supervisions: bool = True,
            num_jobs: int = 1,
    ) -> 'CutSet':
        """
        Return a new ``CutSet``, made by traversing each ``MonoCut`` in windows of ``duration`` seconds and
        creating new ``MonoCut`` out of them.

        The last window might have a shorter duration if there was not enough audio, so you might want to
        use either ``.filter()`` or ``.pad()`` afterwards to obtain a uniform duration ``CutSet``.

        :param duration: Desired duration of the new cuts in seconds.
        :param keep_excessive_supervisions: bool. When a cut is truncated in the middle of a supervision segment,
            should the supervision be kept.
        :param num_jobs: The number of parallel workers.
        :return: a new CutSet with cuts made from shorter duration windows.
        """
        if num_jobs == 1:
            new_cuts = []
            for cut in self:
                n_windows = ceil(cut.duration / duration)
                for i in range(n_windows):
                    new_cuts.append(
                        cut.truncate(
                            offset=duration * i,
                            duration=duration,
                            keep_excessive_supervisions=keep_excessive_supervisions
                        )
                    )
            return CutSet(cuts={c.id: c for c in new_cuts})

        from lhotse.manipulation import split_parallelize_combine
        result = split_parallelize_combine(
            num_jobs,
            self,
            CutSet.cut_into_windows,
            duration=duration,
            keep_excessive_supervisions=keep_excessive_supervisions,
        )
        return result

    def sample(self, n_cuts: int = 1) -> Union[Cut, 'CutSet']:
        """
        Randomly sample this ``CutSet`` and return ``n_cuts`` cuts.
        When ``n_cuts`` is 1, will return a single cut instance; otherwise will return a ``CutSet``.
        """
        assert n_cuts > 0
        # TODO: We might want to make this more efficient in the future
        #  by holding a cached list of cut ids as a member of CutSet...
        cut_indices = [random.randint(0, len(self) - 1) for _ in range(n_cuts)]
        cuts = [self[idx] for idx in cut_indices]
        if n_cuts == 1:
            return cuts[0]
        return CutSet.from_cuts(cuts)

    def resample(self, sampling_rate: int, affix_id: bool = False) -> 'CutSet':
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

    def perturb_speed(self, factor: float, affix_id: bool = True) -> 'CutSet':
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

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> 'CutSet':
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

    def perturb_volume(self, factor: float, affix_id: bool = True) -> 'CutSet':
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
        return self.map(lambda cut: cut.perturb_volume(factor=factor, affix_id=affix_id))

    def mix(
            self,
            cuts: 'CutSet',
            duration: Optional[Seconds] = None,
            snr: Optional[Union[Decibels, Sequence[Decibels]]] = 20,
            mix_prob: float = 1.0
    ) -> 'CutSet':
        """
        Mix cuts in this ``CutSet`` with randomly sampled cuts from another ``CutSet``.
        A typical application would be data augmentation with noise, music, babble, etc.

        :param cuts: a ``CutSet`` containing cuts to be mixed into this ``CutSet``.
        :param duration: an optional float in seconds.
            When ``None``, we will preserve the duration of the cuts in ``self``
            (i.e. we'll truncate the mix if it exceeded the original duration).
            Otherwise, we will keep sampling cuts to mix in until we reach the specified
            ``duration`` (and truncate to that value, should it be exceeded).
        :param snr: an optional float, or pair (range) of floats, in decibels.
            When it's a single float, we will mix all cuts with this SNR level
            (where cuts in ``self`` are treated as signals, and cuts in ``cuts`` are treated as noise).
            When it's a pair of floats, we will uniformly sample SNR values from that range.
            When ``None``, we will mix the cuts without any level adjustment
            (could be too noisy for data augmentation).
        :param mix_prob: an optional float in range [0, 1].
            Specifies the probability of performing a mix.
            Values lower than 1.0 mean that some cuts in the output will be unchanged.
        :return: a new ``CutSet`` with mixed cuts.
        """
        assert 0.0 <= mix_prob <= 1.0
        assert duration is None or duration > 0
        if isinstance(snr, (tuple, list)):
            assert len(snr) == 2, f"SNR range must be a list or tuple with exactly two values (got: {snr})"
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
            mixed = cut.mix(other=to_mix, snr=snr)
            # Did the user specify a duration?
            # If yes, we will ensure that shorter cuts have more noise mixed in
            # to "pad" them with at the end.
            if duration is not None:
                mixed_in_duration = to_mix.duration
                # Keep sampling until we mixed in a "duration" amount of noise.
                while mixed_in_duration < duration:
                    to_mix = cuts.sample()
                    # Keep the SNR constant for each cut from "self".
                    mixed = mixed.mix(other=to_mix, snr=snr, offset_other_by=mixed_in_duration)
                    # Since we're adding floats, we can be off by an epsilon and trigger
                    # some assertions for exceeding duration; do precautionary rounding here.
                    mixed_in_duration = round(mixed_in_duration + to_mix.duration, ndigits=8)
            # We truncate the mixed to either the original duration or the requested duration.
            mixed = mixed.truncate(duration=cut.duration if duration is None else duration)
            mixed_cuts.append(mixed)
        return CutSet.from_cuts(mixed_cuts)

    def drop_features(self) -> 'CutSet':
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its extracted features.
        """
        return CutSet.from_cuts(c.drop_features() for c in self)

    def drop_recordings(self) -> 'CutSet':
        """
        Return a new :class:`.CutSet`, where each :class:`.Cut` is copied and detached from its recordings.
        """
        return CutSet.from_cuts(c.drop_recording() for c in self)

    def drop_supervisions(self) -> 'CutSet':
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
            storage_type: Type[FW] = LilcomHdf5Writer,
            executor: Optional[Executor] = None,
            mix_eagerly: bool = True,
            progress_bar: bool = True,
    ) -> 'CutSet':
        """
        Extract features for all cuts, possibly in parallel,
        and store them using the specified storage object.

        Examples:

            Extract fbank features on one machine using 8 processes,
            store arrays partitioned in 8 HDF5 files with lilcom compression:

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
            store arrays partitioned in 80 HDF5 files with lilcom compression:

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
        progress = identity  # does nothing, unless we overwrite it with an actual prog bar
        if num_jobs is None:
            num_jobs = 1
        if num_jobs == 1 and executor is not None:
            logging.warning('Executor argument was passed but num_jobs set to 1: '
                            'we will ignore the executor and use non-parallel execution.')
            executor = None

        # Non-parallel execution
        if executor is None and num_jobs == 1:
            if progress_bar:
                progress = partial(
                    tqdm, desc='Extracting and storing features', total=len(self)
                )
            with storage_type(storage_path) as storage:
                return CutSet.from_cuts(
                    progress(
                        cut.compute_and_store_features(
                            extractor=extractor,
                            storage=storage,
                            augment_fn=augment_fn,
                            mix_eagerly=mix_eagerly
                        ) for cut in self
                    )
                )

        # HACK: support URL storage for writing
        if '://' in str(storage_path):
            # "storage_path" is actually an URL
            def sub_storage_path(idx: int) -> str:
                return f'{storage_path}/feats-{idx}'
        else:
            # We are now sure that "storage_path" will be the root for
            # multiple feature storages - we can create it as a directory
            storage_path = Path(storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)

            def sub_storage_path(idx: int) -> str:
                return storage_path / f'feats-{idx}'

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
                progress_bar=False
            )
            for i, cs in enumerate(cut_sets)
        ]

        if progress_bar:
            progress = partial(
                tqdm, desc='Extracting and storing features (chunks progress)', total=len(futures)
            )

        cuts_with_feats = combine(progress(f.result() for f in futures))
        return cuts_with_feats

    def compute_and_store_recordings(
            self,
            storage_path: Pathlike,
            num_jobs: Optional[int] = None,
            executor: Optional[Executor] = None,
            augment_fn: Optional[AugmentFn] = None,
            progress_bar: bool = True
    ) -> 'CutSet':
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
        from lhotse.manipulation import combine
        from cytoolz import identity

        # Pre-conditions and args setup
        progress = identity  # does nothing, unless we overwrite it with an actual prog bar
        if num_jobs is None:
            num_jobs = 1
        if num_jobs == 1 and executor is not None:
            logging.warning('Executor argument was passed but num_jobs set to 1: '
                            'we will ignore the executor and use non-parallel execution.')
            executor = None

        def file_storage_path(cut: Cut, storage_path: Pathlike) -> Path:
            # Introduce a sub-directory that starts with the first 3 characters of the cut's ID.
            # This allows to avoid filesystem performance problems related to storing
            # too many files in a single directory.
            subdir = Path(storage_path) / cut.id[:3]
            subdir.mkdir(exist_ok=True, parents=True)
            return (subdir / cut.id).with_suffix('.flac')

        # Non-parallel execution
        if executor is None and num_jobs == 1:
            if progress_bar:
                progress = partial(
                    tqdm, desc='Storing audio recordings', total=len(self)
                )
            return CutSet.from_cuts(
                progress(
                    cut.compute_and_store_recording(
                        storage_path=file_storage_path(cut, storage_path),
                        augment_fn=augment_fn
                    ) for cut in self
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
                CutSet.compute_and_store_recordings,
                cs,
                storage_path=storage_path,
                augment_fn=augment_fn,
                # Disable individual workers progress bars for readability
                progress_bar=False
            )
            for i, cs in enumerate(cut_sets)
        ]

        if progress_bar:
            progress = partial(
                tqdm, desc='Storing audio recordings (chunks progress)', total=len(futures)
            )

        cuts = combine(progress(f.result() for f in futures))
        return cuts

    def compute_global_feature_stats(
            self,
            storage_path: Optional[Pathlike] = None,
            max_cuts: Optional[int] = None
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
            raise ValueError("Could not find any features in this CutSet; did you forget to extract them?")
        if not all(have_features):
            logging.warning(
                f'Computing global stats: only {sum(have_features)}/{len(have_features)} cuts have features.'
            )
        return compute_global_stats(
            # islice(X, 50) is like X[:50] except it works with lazy iterables
            feature_manifests=islice(
                (cut.features for cut in self if cut.has_features),
                max_cuts if max_cuts is not None else len(self)
            ),
            storage_path=storage_path
        )

    def with_features_path_prefix(self, path: Pathlike) -> 'CutSet':
        return CutSet.from_cuts(c.with_features_path_prefix(path) for c in self)

    def with_recording_path_prefix(self, path: Pathlike) -> 'CutSet':
        return CutSet.from_cuts(c.with_recording_path_prefix(path) for c in self)

    def map(self, transform_fn: Callable[[Cut], Cut]) -> 'CutSet':
        """
        Apply `transform_fn` to the cuts in this :class:`.CutSet` and return a new :class:`.CutSet`.

        :param transform_fn: A callable (function) that accepts a single cut instance
            and returns a single cut instance.
        :return: a new ``CutSet`` with transformed cuts.
        """

        def verified(mapped: Any) -> Cut:
            assert isinstance(mapped, (MonoCut, MixedCut, PaddingCut)), \
                "The callable passed to CutSet.map() must return a Cut class instance."
            return mapped

        return CutSet.from_cuts(verified(transform_fn(c)) for c in self)

    def modify_ids(self, transform_fn: Callable[[str], str]) -> 'CutSet':
        """
        Modify the IDs of cuts in this ``CutSet``.
        Useful when combining multiple ``CutSet``s that were created from a single source,
        but contain features with different data augmentations techniques.

        :param transform_fn: A callable (function) that accepts a string (cut ID) and returns
        a new string (new cut ID).
        :return: a new ``CutSet`` with cuts with modified IDs.
        """
        return CutSet.from_cuts(c.with_id(transform_fn(c.id)) for c in self)

    def map_supervisions(self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]) -> 'CutSet':
        """
        Modify the SupervisionSegments by `transform_fn` in this CutSet.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a new, modified CutSet.
        """
        return CutSet.from_cuts(cut.map_supervisions(transform_fn) for cut in self)

    def transform_text(self, transform_fn: Callable[[str], str]) -> 'CutSet':
        """
        Return a copy of this ``CutSet`` with all ``SupervisionSegments`` text transformed with ``transform_fn``.
        Useful for text normalization, phonetic transcription, etc.

        :param transform_fn: a function that accepts a string and returns a string.
        :return: a new, modified CutSet.
        """
        return self.map_supervisions(lambda s: s.transform_text(transform_fn))

    def __repr__(self) -> str:
        return f'CutSet(len={len(self)})'

    def __contains__(self, item: Union[str, Cut]) -> bool:
        if isinstance(item, str):
            return item in self.cuts
        else:
            return item.id in self.cuts

    def __getitem__(self, cut_id_or_index: Union[int, str]) -> 'Cut':
        if isinstance(cut_id_or_index, str):
            return self.cuts[cut_id_or_index]
        # ~100x faster than list(dict.values())[index] for 100k elements
        return next(val for idx, val in enumerate(self.cuts.values()) if idx == cut_id_or_index)

    def __len__(self) -> int:
        return len(self.cuts)

    def __iter__(self) -> Iterable[Cut]:
        return iter(self.cuts.values())

    def __add__(self, other: 'CutSet') -> 'CutSet':
        merged_cuts = {**self.cuts, **other.cuts}
        assert len(merged_cuts) == len(self.cuts) + len(other.cuts), \
            f"Conflicting IDs when concatenating CutSets! " \
            f"Failed check: {len(merged_cuts)} == {len(self.cuts)} + {len(other.cuts)}"
        return CutSet(cuts={**self.cuts, **other.cuts})


def make_windowed_cuts_from_features(
        feature_set: FeatureSet,
        cut_duration: Seconds,
        cut_shift: Optional[Seconds] = None,
        keep_shorter_windows: bool = False
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
        if (n_cuts - 1) * cut_shift + cut_duration > features.duration and not keep_shorter_windows:
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
                    supervisions=[]
                )
            )
    return CutSet.from_cuts(cuts)


def mix(
        reference_cut: Cut,
        mixed_in_cut: Cut,
        offset: Seconds = 0,
        snr: Optional[Decibels] = None
) -> MixedCut:
    """
    Overlay, or mix, two cuts. Optionally the `mixed_in_cut` may be shifted by `offset` seconds
    and scaled down (positive SNR) or scaled up (negative SNR).
    Returns a MixedCut, which contains both cuts and the mix information.
    The actual feature mixing is performed during the call to ``MixedCut.load_features()``.

    :param reference_cut: The reference cut for the mix - offset and snr are specified w.r.t this cut.
    :param mixed_in_cut: The mixed-in cut - it will be offset and rescaled to match the offset and snr parameters.
    :param offset: How many seconds to shift the ``mixed_in_cut`` w.r.t. the ``reference_cut``.
    :param snr: Desired SNR of the `right_cut` w.r.t. the `left_cut` in the mix.
    :return: A MixedCut instance.
    """
    if any(isinstance(cut, PaddingCut) for cut in (reference_cut, mixed_in_cut)) and snr is not None:
        warnings.warn('You are mixing cuts to a padding cut with a specified SNR - '
                      'the resulting energies would be extremely low or high. '
                      'We are setting snr to None, so that the original signal energies will be retained instead.')
        snr = None

    if reference_cut.num_features is not None:
        assert reference_cut.num_features == mixed_in_cut.num_features, "Cannot mix cuts with different feature " \
                                                                        "dimensions. "
    assert offset <= reference_cut.duration, f"Cannot mix cut '{mixed_in_cut.id}' with offset {offset}," \
                                             f" which is greater than cuts {reference_cut.id} duration" \
                                             f" of {reference_cut.duration}"
    assert reference_cut.sampling_rate == mixed_in_cut.sampling_rate, \
        f'Cannot mix cuts with different sampling rates ' \
        f'({reference_cut.sampling_rate} vs. ' \
        f'{mixed_in_cut.sampling_rate}). ' \
        f'Please resample the recordings first.'
    # When the left_cut is a MixedCut, take its existing tracks, otherwise create a new track.
    old_tracks = (
        reference_cut.tracks
        if isinstance(reference_cut, MixedCut)
        else [MixTrack(cut=reference_cut)]
    )
    # When the right_cut is a MixedCut, adapt its existing tracks with the new offset and snr,
    # otherwise create a new track.
    new_tracks = (
        [
            MixTrack(
                cut=track.cut,
                offset=round(track.offset + offset, ndigits=8),
                snr=(
                    # When no new SNR is specified, retain whatever was there in the first place.
                    track.snr if snr is None
                    # When new SNR is specified but none was specified before, assign the new SNR value.
                    else snr if track.snr is None
                    # When both new and previous SNR were specified, assign their sum,
                    # as the SNR for each track is defined with regard to the first track energy.
                    else track.snr + snr if snr is not None and track is not None
                    # When no SNR was specified whatsoever, use none.
                    else None
                )
            ) for track in mixed_in_cut.tracks
        ]
        if isinstance(mixed_in_cut, MixedCut)
        else [MixTrack(cut=mixed_in_cut, offset=offset, snr=snr)]
    )
    return MixedCut(
        id=str(uuid4()),
        tracks=old_tracks + new_tracks
    )


def pad(
        cut: Cut,
        duration: Seconds = None,
        num_frames: int = None,
        num_samples: int = None,
        pad_feat_value: float = LOG_EPSILON,
        direction: str = 'right'
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
    :return: a padded MixedCut if duration is greater than this cut's duration, otherwise ``self``.
    """
    assert exactly_one_not_null(duration, num_frames, num_samples), \
        f"Expected only one of (duration, num_frames, num_samples) to be set: " \
        f"got ({duration}, {num_frames}, {num_samples})"

    if duration is not None:
        if duration <= cut.duration:
            return cut
        total_num_frames = compute_num_frames(
            duration=duration,
            frame_shift=cut.frame_shift,
            sampling_rate=cut.sampling_rate
        ) if cut.has_features else None
        total_num_samples = compute_num_samples(
            duration=duration,
            sampling_rate=cut.sampling_rate
        ) if cut.has_recording else None

    if num_frames is not None:
        assert cut.has_features, 'Cannot pad a cut using num_frames when it is missing pre-computed features ' \
                                 '(did you run cut.compute_and_store_features(...)?).'
        total_num_frames = num_frames
        duration = total_num_frames * cut.frame_shift
        total_num_samples = compute_num_samples(
            duration=duration,
            sampling_rate=cut.sampling_rate
        ) if cut.has_recording else None
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
        assert cut.has_recording, 'Cannot pad a cut using num_samples when it is missing a Recording object ' \
                                  '(did you attach recording/recording set when creating the cut/cut set?)'
        if num_samples <= cut.num_samples:
            return cut
        total_num_samples = num_samples
        duration = total_num_samples / cut.sampling_rate
        total_num_frames = compute_num_frames(
            duration=duration,
            frame_shift=cut.frame_shift,
            sampling_rate=cut.sampling_rate
        ) if cut.has_features else None

    padding_cut = PaddingCut(
        id=str(uuid4()),
        duration=round(duration - cut.duration, ndigits=8),
        feat_value=pad_feat_value,
        num_features=cut.num_features,
        # The num_frames and sampling_rate fields are tricky, because it is possible to create a MixedCut
        # from Cuts that have different sampling rates and frame shifts. In that case, we are assuming
        # that we should use the values from the reference cut, i.e. the first one in the mix.
        num_frames=(
            total_num_frames - cut.num_frames
            if cut.has_features
            else None
        ),
        num_samples=(
            total_num_samples - cut.num_samples
            if cut.has_recording
            else None
        ),
        frame_shift=cut.frame_shift,
        sampling_rate=cut.sampling_rate,
    )

    if direction == 'right':
        padded = cut.append(padding_cut)
    elif direction == 'left':
        padded = padding_cut.append(cut)
    elif direction == 'both':
        padded = (
            padding_cut.truncate(duration=padding_cut.duration / 2)
                .append(cut)
                .append(padding_cut.truncate(duration=padding_cut.duration / 2))
        )
    else:
        raise ValueError(f"Unknown type of padding: {direction}")

    return padded


def append(
        left_cut: Cut,
        right_cut: Cut,
        snr: Optional[Decibels] = None
) -> MixedCut:
    """Helper method for functional-style appending of Cuts."""
    return left_cut.append(right_cut, snr=snr)


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
        use_alignment_if_exists: Optional[str] = None
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
    assert cut.has_features or frame_shift is not None, f"No features available. " \
                                                        f"Either pre-compute features or provide frame_shift."
    if cut.has_features:
        frame_shift = cut.frame_shift
        num_frames = cut.num_frames
    else:
        num_frames = compute_num_frames(
            duration=cut.duration,
            frame_shift=frame_shift,
            sampling_rate=cut.sampling_rate
        )
    mask = np.zeros(num_frames, dtype=np.float32)
    for supervision in cut.supervisions:
        if use_alignment_if_exists and supervision.alignment and use_alignment_if_exists in supervision.alignment:
            for ali in supervision.alignment[use_alignment_if_exists]:
                st = round(ali.start / frame_shift) if ali.start > 0 else 0
                et = round(ali.end / frame_shift) if ali.end < cut.duration else num_frames
                mask[st:et] = 1.0
        else:
            st = round(supervision.start / frame_shift) if supervision.start > 0 else 0
            et = round(supervision.end / frame_shift) if supervision.end < cut.duration else num_frames
            mask[st:et] = 1.0
    return mask
