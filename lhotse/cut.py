import logging
import numpy as np
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, Executor
from cytoolz import sliding_window
from cytoolz.itertoolz import groupby
from dataclasses import dataclass, field
from functools import partial, reduce
from intervaltree import Interval, IntervalTree
from itertools import islice
from math import ceil, floor
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from lhotse.audio import AudioMixer, Recording, RecordingSet
from lhotse.augmentation import AugmentFn
from lhotse.features import FeatureExtractor, FeatureMixer, FeatureSet, Features, create_default_feature_extractor
from lhotse.features.base import compute_global_stats
from lhotse.features.io import FeaturesWriter, LilcomHdf5Writer, LilcomFilesWriter
from lhotse.features.mixer import NonPositiveEnergyError
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (Decibels, EPSILON, JsonMixin, Pathlike, Seconds, TimeSpan, YamlMixin, asdict_nonull,
                          compute_num_frames, fastcopy,
                          index_by_id_and_check, overlaps, overspans, perturb_num_samples, split_sequence, uuid4)

# One of the design principles for Cuts is a maximally "lazy" implementation, e.g. when mixing Cuts,
# we'd rather sum the feature matrices only after somebody actually calls "load_features". It helps to avoid
# an excessive storage size for data augmented in various ways.


# Helper "typedef" to artbitrary Cut type as they do not share a common base class.
# The class names are strings here so that the Python interpreter resolves them after parsing the whole file.
AnyCut = Union['Cut', 'MixedCut', 'PaddingCut']

FW = TypeVar('FW', bound=FeaturesWriter)


# noinspection PyTypeChecker,PyUnresolvedReferences
class CutUtilsMixin:
    """
    A mixin class for cuts which contains all the methods that share common implementations.

    Note: Ideally, this would've been an abstract base class specifying the common interface,
    but ABC's do not mix well with dataclasses in Python. It is possible we'll ditch the dataclass
    for cuts in the future and make this an ABC instead.
    """

    @property
    def trimmed_supervisions(self) -> List[SupervisionSegment]:
        """
        Return the supervisions in this Cut that have modified time boundaries so as not to exceed
        the Cut's start or end.

        Note that when ``cut.supervisions`` is called, the supervisions may have negative ``start``
        values that indicate the supervision actually begins before the cut, or ``end`` values
        that exceed the Cut's duration (it means the supervision continued in the original recording
        after the Cut's ending).
        """
        return [s.trim(self.duration) for s in self.supervisions]

    def mix(self, other: AnyCut, offset_other_by: Seconds = 0.0, snr: Optional[Decibels] = None) -> 'MixedCut':
        """Refer to mix() documentation."""
        return mix(self, other, offset=offset_other_by, snr=snr)

    def append(self, other: AnyCut, snr: Optional[Decibels] = None) -> 'MixedCut':
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

    def speakers_feature_mask(
            self,
            min_speaker_dim: Optional[int] = None,
            speaker_to_idx_map: Optional[Dict[str, int]] = None
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
            st = round(supervision.start / self.frame_shift) if supervision.start > 0 else 0
            et = round(supervision.end / self.frame_shift) if supervision.end < self.duration else self.num_frames
            mask[speaker_idx, st:et] = 1
        return mask

    def speakers_audio_mask(
            self,
            min_speaker_dim: Optional[int] = None,
            speaker_to_idx_map: Optional[Dict[str, int]] = None
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
            st = round(supervision.start * self.sampling_rate) if supervision.start > 0 else 0
            et = (
                round(supervision.end * self.sampling_rate)
                if supervision.end < self.duration
                else self.duration * self.sampling_rate
            )
            mask[speaker_idx, st:et] = 1
        return mask

    def supervisions_feature_mask(self) -> np.ndarray:
        """
        Return a 1D numpy array with value 1 for **frames** covered by at least one supervision,
        and 0 for **frames** not covered by any supervision.
        """
        assert self.has_features, f"No features available. " \
                                  f"Can't compute supervisions feature mask for cut with ID: {self.id}."
        mask = np.zeros(self.num_frames, dtype=np.float32)
        for supervision in self.supervisions:
            st = round(supervision.start / self.frame_shift) if supervision.start > 0 else 0
            et = round(supervision.end / self.frame_shift) if supervision.end < self.duration else self.num_frames
            mask[st:et] = 1.0
        return mask

    def supervisions_audio_mask(self) -> np.ndarray:
        """
        Return a 1D numpy array with value 1 for **samples** covered by at least one supervision,
        and 0 for **samples** not covered by any supervision.
        """
        assert self.has_recording, f"No recording available. " \
                                   f"Can't compute supervisions audio mask for cut with ID: {self.id}."
        mask = np.zeros(self.num_samples, dtype=np.float32)
        for supervision in self.supervisions:
            st = round(supervision.start * self.sampling_rate) if supervision.start > 0 else 0
            et = (
                round(supervision.end * self.sampling_rate)
                if supervision.end < self.duration
                else self.duration * self.sampling_rate
            )
            mask[st:et] = 1.0
        return mask

    def with_id(self, id_: str) -> AnyCut:
        """Return a copy of the Cut with a new ID."""
        return fastcopy(self, id=id_)


@dataclass
class Cut(CutUtilsMixin):
    """
    A Cut is a single "segment" that we'll train on. It contains the features corresponding to
    a piece of a recording, with zero or more SupervisionSegments.

    The SupervisionSegments indicate which time spans of the Cut contain some kind of supervision information:
    e.g. transcript, speaker, language, etc. The regions without a corresponding SupervisionSegment may
    contain anything - usually we assume it's either silence or some kind of noise.

    Note: The SupervisionSegment time boundaries are relative to the beginning of the cut.
    E.g. if the underlying Recording starts at 0s (always true), the Cut starts at 100s,
    and the SupervisionSegment starts at 3s, it means that in the Recording the supervision actually started at 103s.
    In some cases, the supervision might have a negative start, or a duration exceeding the duration of the Cut;
    this means that the supervision in the recording extends beyond the Cut.
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
        return compute_num_frames(duration=self.duration, frame_shift=self.frame_shift) if self.has_features else None

    @property
    def num_samples(self) -> Optional[int]:
        return round(self.duration * self.sampling_rate) if self.has_recording else None

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
        [begin, duration] region of the current Cut.
        """
        if self.has_features:
            return self.features.load(start=self.start, duration=self.duration)
        return None

    def load_audio(self) -> Optional[np.ndarray]:
        """
        Load the audio by locating the appropriate recording in the supplied RecordingSet.
        The audio is trimmed to the [begin, end] range specified by the Cut.

        :return: a numpy ndarray with audio samples, with shape (1 <channel>, N <samples>)
        """
        if self.has_recording:
            return self.recording.load_audio(
                channels=self.channel,
                offset_seconds=self.start,
                duration_seconds=self.duration,
            )
        return None

    def compute_and_store_features(
            self,
            extractor: FeatureExtractor,
            storage: FeaturesWriter,
            augment_fn: Optional[AugmentFn] = None,
            *args,
            **kwargs
    ) -> AnyCut:
        """
        Compute the features from this cut, store them on disk, and attach a feature manifest to this cut.
        This cut has to be able to load audio.

        :param extractor: a ``FeatureExtractor`` instance used to compute the features.
        :param output_dir: the directory where the computed features will be stored.
        :param augment_fn: an optional callable used for audio augmentation.
        :return: a new ``Cut`` instance with a ``Features`` manifest attached to it.
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
    ) -> 'Cut':
        """
        Returns a new Cut that is a sub-region of the current Cut.

        Note that no operation is done on the actual features - it's only during the call to load_features()
        when the actual changes happen (a subset of features is loaded).

        :param offset: float (seconds), controls the start of the new cut relative to the current Cut's start.
            E.g., if the current Cut starts at 10.0, and offset is 2.0, the new start is 12.0.
        :param duration: optional float (seconds), controls the duration of the resulting Cut.
            By default, the duration is (end of the cut before truncation) - (offset).
        :param keep_excessive_supervisions: bool. Since trimming may happen inside a SupervisionSegment,
            the caller has an option to either keep or discard such supervisions.
        :param preserve_id: bool. Should the truncated cut keep the same ID or get a new, random one.
        :param _supervisions_index: when passed, allows to speed up processing of Cuts with a very
            large number of supervisions. Intended as an internal parameter.
        :return: a new Cut instance. If the current Cut is shorter than the duration, return None.
        """
        new_start = self.start + offset
        until = offset + (duration if duration is not None else self.duration)
        new_duration = self.duration - new_start if duration is None else until - offset
        assert new_duration > 0.0
        duration_past_end = (new_start + new_duration) - (self.start + self.duration)
        if duration_past_end > 0:
            # When the end of the Cut has been exceeded, trim the new duration to not exceed the old Cut's end.
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
            match_supervisions = tree.overlap if keep_excessive_supervisions else tree.envelop
            supervisions = [
                interval.data.with_offset(-offset)
                for interval in match_supervisions(begin=offset, end=offset + new_duration)
            ]

        return Cut(
            id=self.id if preserve_id else str(uuid4()),
            start=new_start,
            duration=new_duration,
            channel=self.channel,
            supervisions=sorted(supervisions, key=lambda s: s.start),
            features=self.features,
            recording=self.recording
        )

    def pad(self, duration: Seconds) -> AnyCut:
        """
        Return a new MixedCut, padded to ``duration`` seconds with zeros in the recording,
        and low-energy values in each feature bin.

        :param duration: The cut's minimal duration after padding.
        :return: a padded MixedCut if ``duration`` is greater than this cut's duration, otherwise ``self``.
        """
        if duration <= self.duration:
            return self
        total_num_frames = compute_num_frames(duration=duration, frame_shift=self.frame_shift) \
            if self.has_features else None
        total_num_samples = round(duration * self.sampling_rate) if self.has_recording else None
        padding_duration = round(duration - self.duration, ndigits=8)
        padded = self.append(PaddingCut(
            id=str(uuid4()),
            duration=padding_duration,
            num_features=self.num_features if self.features is not None else None,
            num_frames=total_num_frames - self.num_frames if self.features is not None else None,
            num_samples=total_num_samples - self.num_samples if self.recording is not None else None,
            sampling_rate=self.features.sampling_rate if self.features is not None else self.recording.sampling_rate,
            use_log_energy=self.features.type in ('fbank', 'mfcc') if self.features is not None else False
        ))
        return padded

    def perturb_speed(self, factor: float, affix_id: bool = True) -> 'Cut':
        """
        Return a new ``Cut`` that will lazily perturb the speed while loading audio.
        The ``num_samples``, ``start`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.
        We are also updating the time markers of the underlying ``Recording`` and the supervisions.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``Cut.id`` field
            by affixing it with "_sp{factor}".
        :return: a modified copy of the current ``Cut``.
        """
        # Pre-conditions
        assert self.has_recording, 'Cannot perturb speed on a Cut without Recording.'
        if self.has_features:
            logging.warning(
                'Attempting to perturb speed on a Cut that references pre-computed features. '
                'The feature manifest will be detached, as we do not support feature-domain '
                'speed perturbation.'
            )
            self.features = None
        # Actual audio perturbation.
        recording_sp = self.recording.perturb_speed(factor=factor, affix_id=affix_id)
        # Match the supervision's start and duration to the perturbed audio.
        # Since SupervisionSegment "start" is relative to the Cut's, it's okay (and necessary)
        # to perturb it as well.
        supervisions_sp = [
            s.perturb_speed(factor=factor, sampling_rate=self.sampling_rate, affix_id=affix_id)
            for s in self.supervisions
        ]
        # New start and duration have to be computed through num_samples to be accurate
        start_samples = perturb_num_samples(round(self.start * self.sampling_rate), factor)
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

    def map_supervisions(self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]) -> AnyCut:
        """
        Modify the SupervisionSegments by `transform_fn` of this Cut.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a modified Cut.
        """
        new_cut = fastcopy(self, supervisions=[s.map(transform_fn) for s in self.supervisions])
        return new_cut

    def filter_supervisions(self, predicate: Callable[[SupervisionSegment], bool]) -> AnyCut:
        """
        Modify cut to store only supervisions accepted by `predicate`

        Example:
            >>> cut = cut.filter_supervisions(lambda s: s.id in supervision_ids)
            >>> cut = cut.filter_supervisions(lambda s: s.duration < 5.0)
            >>> cut = cut.filter_supervisions(lambda s: s.text is not None)

        :param predicate: A callable that accepts `SupervisionSegment` and returns bool
        :return: a modified Cut
        """
        new_cut = fastcopy(self, supervisions=[s for s in self.supervisions if predicate(s)])
        return new_cut

    @staticmethod
    def from_dict(data: dict) -> 'Cut':
        features = Features.from_dict(data.pop('features')) if 'features' in data else None
        recording = Recording.from_dict(data.pop('recording')) if 'recording' in data else None
        supervision_infos = data.pop('supervisions') if 'supervisions' in data else []
        return Cut(
            **data,
            features=features,
            recording=recording,
            supervisions=[SupervisionSegment.from_dict(s) for s in supervision_infos]
        )

    def with_features_path_prefix(self, path: Pathlike) -> 'Cut':
        if not self.has_features:
            return self
        return fastcopy(self, features=self.features.with_path_prefix(path))

    def with_recording_path_prefix(self, path: Pathlike) -> 'Cut':
        if not self.has_recording:
            return self
        return fastcopy(self, recording=self.recording.with_path_prefix(path))


@dataclass
class PaddingCut(CutUtilsMixin):
    """
    This represents a cut filled with zeroes in the time domain, or low energy/log-energy values in the
    frequency domain. It's used to make training samples evenly sized (same duration/number of frames).
    """
    id: str
    duration: Seconds

    sampling_rate: int
    use_log_energy: bool

    # For frequency domain
    num_frames: Optional[int] = None
    num_features: Optional[int] = None

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

    @property
    def frame_shift(self):
        return round(self.duration / self.num_frames, ndigits=3) if self.has_features else None

    # noinspection PyUnusedLocal
    def load_features(self, *args, **kwargs) -> Optional[np.ndarray]:
        if self.has_features:
            value = np.log(EPSILON) if self.use_log_energy else EPSILON
            return np.ones((self.num_frames, self.num_features), np.float32) * value
        return None

    # noinspection PyUnusedLocal
    def load_audio(self, *args, **kwargs) -> Optional[np.ndarray]:
        if self.has_recording:
            return np.zeros((1, round(self.duration * self.sampling_rate)), np.float32)
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
        return PaddingCut(
            id=self.id if preserve_id else str(uuid4()),
            duration=new_duration,
            num_frames=round(new_duration / self.frame_shift) if self.num_frames is not None else None,
            num_features=self.num_features,
            num_samples=round(new_duration * self.sampling_rate) if self.num_samples is not None else None,
            use_log_energy=self.use_log_energy,
            sampling_rate=self.sampling_rate
        )

    def pad(self, duration: Seconds) -> 'PaddingCut':
        """
        Create a new PaddingCut with ``duration`` when its longer than this Cuts duration.
        Helper function used in batch cut padding.

        :param duration: The cuts minimal duration after padding.
        :return: ``self`` or a new PaddingCut, depending on ``duration``.
        """
        if duration <= self.duration:
            return self
        return PaddingCut(
            id=str(uuid4()),
            duration=duration,
            num_features=self.num_features,
            num_frames=round(duration / self.frame_shift),
            sampling_rate=self.sampling_rate,
            use_log_energy=self.use_log_energy
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
                'Attempting to perturb speed on a Cut that references pre-computed features. '
                'The feature manifest will be detached, as we do not support feature-domain '
                'speed perturbation.'
            )
            self.num_frames = None
            self.num_features = None
        new_num_samples = perturb_num_samples(self.num_samples, factor)
        new_duration = new_num_samples / self.sampling_rate
        return fastcopy(
            self,
            id=f'{self.id}_sp{factor}' if affix_id else self.id,
            num_samples=new_num_samples,
            duration=new_duration
        )

    def compute_and_store_features(self, extractor: FeatureExtractor, *args, **kwargs) -> AnyCut:
        """
        Returns a new PaddingCut with updates information about the feature dimension and number of
        feature frames, depending on the ``extractor`` properties.
        """
        return fastcopy(
            self,
            num_features=extractor.feature_dim(self.sampling_rate),
            num_frames=round(self.duration / extractor.frame_shift)
        )

    def map_supervisions(self, transform_fn: Callable[[Any], Any]) -> AnyCut:
        """
        Just for consistency with `Cut` and `MixedCut`.

        :param transform_fn: a dummy function that would be never called actually.
        :return: the PaddingCut itself.
        """
        return self

    def filter_supervisions(self, predicate: Callable[[SupervisionSegment], bool]) -> AnyCut:
        """
        Just for consistency with `Cut` and `MixedCut`.

        :param predicate: A callable that accepts `SupervisionSegment` and returns bool
        :return: a modified Cut
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
    Represents a single track in a mix of Cuts. Points to a specific Cut and holds information on
    how to mix it with other Cuts, relative to the first track in a mix.
    """
    cut: Union[Cut, PaddingCut]
    offset: Seconds = 0.0
    snr: Optional[Decibels] = None

    @staticmethod
    def from_dict(data: dict):
        raw_cut = data.pop('cut')
        try:
            cut = Cut.from_dict(raw_cut)
        except TypeError:
            cut = PaddingCut.from_dict(raw_cut)
        return MixTrack(cut, **data)


@dataclass
class MixedCut(CutUtilsMixin):
    """
    Represents a Cut that's created from other Cuts via mix or append operations.
    The actual mixing operations are performed upon loading the features into memory.
    In order to load the features, it needs to access the CutSet object that holds the "ingredient" cuts,
    as it only holds their IDs ("pointers").
    The SNR and offset of all the tracks are specified relative to the first track.
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
            return compute_num_frames(duration=self.duration, frame_shift=self._first_non_padding_cut.frame_shift)
        return None

    @property
    def frame_shift(self) -> Optional[Seconds]:
        return self._first_non_padding_cut.frame_shift

    @property
    def sampling_rate(self) -> Optional[int]:
        return self._first_non_padding_cut.sampling_rate

    @property
    def num_samples(self) -> Optional[int]:
        return round(self.duration * self.sampling_rate)

    @property
    def num_features(self) -> Optional[int]:
        return self._first_non_padding_cut.num_features

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
    ) -> 'MixedCut':
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
            # (not to be confused with the 'start' of the underlying Cut)
            track_end = track.offset + track.cut.duration

            if track_end < offset:
                # Omit a Cut that ends before the truncation offset.
                continue

            cut_duration_decrease = 0
            if track_end > new_mix_end:
                if duration is not None:
                    cut_duration_decrease = track_end - new_mix_end
                else:
                    cut_duration_decrease = track_end - old_duration

            # Compute the new Cut's duration after trimming the start and the end.
            new_duration = track.cut.duration - cut_offset - cut_duration_decrease
            if new_duration <= 0:
                # Omit a Cut that is completely outside the time span of the new truncated MixedCut.
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
        return MixedCut(id=str(uuid4()), tracks=new_tracks)

    def pad(self, duration: Seconds) -> AnyCut:
        """
        Return a new MixedCut, padded to ``duration`` seconds with zeros in the recording,
        and low-energy values in each feature bin.

        :param duration: The cut's minimal duration after padding.
        :return: a padded MixedCut if duration is greater than this cut's duration, otherwise ``self``.
        """
        if duration <= self.duration:
            return self
        if self.has_features:
            total_num_frames = compute_num_frames(duration=duration, frame_shift=self.frame_shift)
        if self.has_recording:
            total_num_samples = round(duration * self.sampling_rate)
        padding_duration = round(duration - self.duration, ndigits=8)
        return self.append(PaddingCut(
            id=str(uuid4()),
            duration=padding_duration,
            num_features=self.num_features,
            # The num_frames and sampling_rate fields are tricky, because it is possible to create a MixedCut
            # from Cuts that have different sampling rates and frame shifts. In that case, we are assuming
            # that we should use the values from the reference cut, i.e. the first one in the mix.
            num_frames=(
                total_num_frames - self.num_frames
                if self.has_features
                else None
            ),
            num_samples=(
                total_num_samples - self.num_samples
                if self.has_recording
                else None
            ),
            sampling_rate=self.tracks[0].cut.sampling_rate,
            use_log_energy=self.features_type in ('fbank', 'mfcc')
        ))

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
        assert self.has_recording, 'Cannot perturb speed on a Cut without Recording.'
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
                            num_samples=round(track.offset * self.sampling_rate),
                            factor=factor
                        ) / self.sampling_rate,
                        ndigits=8
                    ),
                    snr=track.snr
                )
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
                    offset=track.offset
                )
            except NonPositiveEnergyError as e:
                logging.warning(str(e) + f' Cut with id "{track.cut.id}" will not be mixed in.')
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
            #   with a regular Cut at the end, when the mix offsets are floats with a lot of decimals.
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
            mixer.add_to_mix(
                audio=track.cut.load_audio(),
                snr=track.snr,
                offset=track.offset,
            )
        if mixed:
            # Off-by-one errors can happen during mixing due to imperfect float arithmetic and rounding;
            # we will fix them on-the-fly so that the manifest does not lie about the num_samples.
            audio = mixer.mixed_audio
            if audio.shape[1] - self.num_samples == 1:
                audio = audio[:, :self.num_samples]
            if audio.shape[1] - self.num_samples == -1:
                audio = np.concatenate((audio, audio[:, -1:]), axis=1)
            assert audio.shape[1] == self.num_samples, "Inconsistent number of samples in a MixedCut: please report " \
                                                       "this issue at https://github.com/lhotse-speech/lhotse/issues " \
                                                       "showing the output of print(cut) or str(cut) on which" \
                                                       "load_audio() was called."
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

    def compute_and_store_features(
            self,
            extractor: FeatureExtractor,
            storage: FeaturesWriter,
            augment_fn: Optional[AugmentFn] = None,
            mix_eagerly: bool = True
    ) -> AnyCut:
        """
        Compute the features from this cut, store them on disk, and create a new `Cut` object with the
        feature manifest attached. This cut has to be able to load audio.

        :param extractor: a ``FeatureExtractor`` instance used to compute the features.
        :param storage: a ``FeaturesWriter`` instance used to store the features.
        :param augment_fn: an optional callable used for audio augmentation.
        :param mix_eagerly: when False, extract and store the features for each track separately,
            and mix them dynamically when loading the features.
            When True, mix the audio first and store the mixed features, returning a new ``Cut`` instance
            with the same ID. The returned ``Cut`` will not have a ``Recording`` attached.
        :return: a new ``Cut`` instance if ``mix_eagerly`` is True, or returns ``self``
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
            return Cut(
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

    def map_supervisions(self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]) -> AnyCut:
        """
        Modify the SupervisionSegments by `transform_fn` of this MixedCut.

        :param transform_fn: a function that modifies a supervision as an argument.
        :return: a modified MixedCut.
        """
        new_mixed_cut = fastcopy(self)
        for track in new_mixed_cut.tracks:
            track.cut.supervisions = [segment.map(transform_fn) for segment in track.cut.supervisions]
        return new_mixed_cut

    def filter_supervisions(self, predicate: Callable[[SupervisionSegment], bool]) -> AnyCut:
        """
        Modify cut to store only supervisions accepted by `predicate`

        Example:
            >>> cut = cut.filter_supervisions(lambda s: s.id in supervision_ids)
            >>> cut = cut.filter_supervisions(lambda s: s.duration < 5.0)
            >>> cut = cut.filter_supervisions(lambda s: s.text is not None)

        :param predicate: A callable that accepts `SupervisionSegment` and returns bool
        :return: a modified Cut
        """
        new_mixed_cut = fastcopy(self)
        for track in new_mixed_cut.tracks:
            track.cut = track.cut.filter_supervisions(predicate)
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
    def _first_non_padding_cut(self) -> Cut:
        return [t.cut for t in self.tracks if not isinstance(t.cut, PaddingCut)][0]


@dataclass
class CutSet(JsonMixin, YamlMixin, Sequence[AnyCut]):
    """
    CutSet combines features with their corresponding supervisions.
    It may have wider span than the actual supervisions, provided the features for the whole span exist.
    It is the basic building block of PyTorch-style Datasets for speech/audio processing tasks.
    """
    cuts: Dict[str, AnyCut]

    @property
    def mixed_cuts(self) -> Dict[str, MixedCut]:
        return {id_: cut for id_, cut in self.cuts.items() if isinstance(cut, MixedCut)}

    @property
    def simple_cuts(self) -> Dict[str, Cut]:
        return {id_: cut for id_, cut in self.cuts.items() if isinstance(cut, Cut)}

    @property
    def ids(self) -> Iterable[str]:
        return self.cuts.keys()

    @property
    def speakers(self) -> FrozenSet[str]:
        return frozenset(supervision.speaker for cut in self for supervision in cut.supervisions)

    @staticmethod
    def from_cuts(cuts: Iterable[AnyCut]) -> 'CutSet':
        return CutSet(cuts=index_by_id_and_check(cuts))

    @staticmethod
    def from_manifests(
            recordings: Optional[RecordingSet] = None,
            supervisions: Optional[SupervisionSet] = None,
            features: Optional[FeatureSet] = None
    ) -> 'CutSet':
        """
        Create a CutSet from any combination of supervision, feature and recording manifests.
        At least one of ``recording_set`` or ``feature_set`` is required.
        The Cut boundaries correspond to those found in the ``feature_set``, when available,
        otherwise to those found in the ``recording_set``
        When a ``supervision_set`` is provided, we'll attach to the Cut all supervisions that
        have a matching recording ID and are fully contained in the Cut's boundaries.
        """
        assert features is not None or recordings is not None, \
            "At least one of feature_set and recording_set has to be provided."
        sup_ok, feat_ok, rec_ok = supervisions is not None, features is not None, recordings is not None
        if feat_ok:
            # Case I: Features are provided.
            # Use features to determine the cut boundaries and attach recordings and supervisions as available.
            return CutSet.from_cuts(
                Cut(
                    id=str(uuid4()),
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
                for feats in features
            )
        # Case II: Recordings are provided (and features are not).
        # Use recordings to determine the cut boundaries.
        return CutSet.from_cuts(
            Cut(
                id=str(uuid4()),
                start=0,
                duration=recording.duration,
                channel=channel,
                recording=recording,
                supervisions=list(supervisions.find(
                    recording_id=recording.id,
                    channel=channel
                )) if sup_ok else []
            )
            for recording in recordings
            # A single cut always represents a single channel. When a recording has multiple channels,
            # we create a new cut for each channel separately.
            for channel in recording.channel_ids
        )

    @staticmethod
    def from_dicts(data: Iterable[dict]) -> 'CutSet':
        def deserialize_one(raw_cut: dict) -> AnyCut:
            cut_type = raw_cut.pop('type')
            if cut_type == 'Cut':
                return Cut.from_dict(raw_cut)
            if cut_type == 'MixedCut':
                return MixedCut.from_dict(raw_cut)
            raise ValueError(f"Unexpected cut type during deserialization: '{cut_type}'")

        return CutSet.from_cuts(deserialize_one(cut) for cut in data)

    def to_dicts(self) -> List[dict]:
        return [{**asdict_nonull(cut), 'type': type(cut).__name__} for cut in self]

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

    def split(self, num_splits: int, shuffle: bool = False) -> List['CutSet']:
        """
        Split the ``CutSet`` into ``num_splits`` pieces of equal size.

        :param num_splits: Requested number of splits.
        :param shuffle: Optionally shuffle the cuts order first.
        :return: A list of ``CutSet`` pieces.
        """
        return [
            CutSet.from_cuts(subset) for subset in
            split_sequence(self, num_splits=num_splits, shuffle=shuffle)
        ]

    def subset(self, supervision_ids: Optional[Iterable[str]] = None) -> 'CutSet':
        """
        Return a new CutSet with Cuts containing only `supervision_ids`.

        Cuts without supervisions are removed.

        Example:
            >>> cuts = CutSet.from_yaml('path/to/cuts')
            >>> train_set = cuts.subset(supervision_ids=train_ids)
            >>> test_set = cuts.subset(supervision_ids=test_ids)

        :param supervision_ids: List of `supervision_ids` to keep
        :return: a CutSet with filtered supervisions
        """

        # remove cuts without supervisions
        supervision_ids = set(supervision_ids)
        return CutSet.from_cuts(
            cut.filter_supervisions(lambda s: s.id in supervision_ids) for cut in self
            if any(s.id in supervision_ids for s in cut.supervisions)
        )

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

    def filter(self, predicate: Callable[[AnyCut], bool]) -> 'CutSet':
        """
        Return a new CutSet with the Cuts that satisfy the `predicate`.

        :param predicate: a function that takes a cut as an argument and returns bool.
        :return: a filtered CutSet.
        """
        return CutSet.from_cuts(cut for cut in self if predicate(cut))

    def trim_to_supervisions(self) -> 'CutSet':
        """
        Return a new CutSet with Cuts that have identical spans as their supervisions.

        :return: a ``CutSet``.
        """
        supervisions_index = self.index_supervisions(index_mixed_tracks=True)
        return CutSet.from_cuts(
            cut.truncate(offset=segment.start, duration=segment.duration,
                         _supervisions_index=supervisions_index)
            for cut in self
            for segment in cut.supervisions
        )

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
        """Sort the CutSet according to cuts duration. Descending by default."""
        return CutSet.from_cuts(sorted(self, key=(lambda cut: cut.duration), reverse=not ascending))

    def index_supervisions(self, index_mixed_tracks: bool = False) -> Dict[str, IntervalTree]:
        """
        Create a two-level index of supervision segments. It is a mapping from a Cut's ID to an
        interval tree that contains the supervisions of that Cut.

        The interval tree can be efficiently queried for overlapping and/or enveloping segments.
        It helps speed up some operations on Cuts of very long recordings (1h+) that contain many
        supervisions.

        :param index_mixed_tracks: Should the tracks of MixedCut's be indexed as additional, separate entries.
        :return: a mapping from Cut ID to an interval tree of SupervisionSegments.
        """
        indexed = {
            cut.id: IntervalTree(Interval(s.start, s.end, s) for s in cut.supervisions)
            for cut in self
        }
        if index_mixed_tracks:
            for cut in self:
                if isinstance(cut, MixedCut):
                    for track in cut.tracks:
                        indexed[track.cut.id] = IntervalTree(
                            Interval(s.start, s.end, s) for s in track.cut.supervisions)
        return indexed

    def pad(
            self,
            duration: Seconds = None,
    ) -> 'CutSet':
        """
        Return a new CutSet with Cuts padded to ``duration`` in seconds.
        Cuts longer than ``duration`` will not be affected.
        Cuts will be padded to the right (i.e. after the signal).
        :param duration: The cuts minimal duration after padding.
        When not specified, we'll choose the duration of the longest cut in the CutSet.
        :return: A padded CutSet.
        """
        if duration is None:
            duration = max(cut.duration for cut in self)
        return CutSet.from_cuts(cut.pad(duration=duration) for cut in self)

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

    def cut_into_windows(self, duration: Seconds, keep_excessive_supervisions: bool = True) -> 'CutSet':
        """
        Return a new ``CutSet``, made by traversing each ``Cut`` in windows of ``duration`` seconds and
        creating new ``Cut`` out of them.

        The last window might have a shorter duration if there was not enough audio, so you might want to
        use either ``.filter()`` or ``.pad()`` afterwards to obtain a uniform duration ``CutSet``.

        :param duration: Desired duration of the new cuts in seconds.
        :param keep_excessive_supervisions: bool. When a cut is truncated in the middle of a supervision segment,
            should the supervision be kept.
        :return: a new CutSet with cuts made from shorter duration windows.
        """
        new_cuts = []
        for cut in self:
            n_windows = ceil(cut.duration / duration)
            for i in range(n_windows):
                new_cuts.append(cut.truncate(
                    offset=duration * i,
                    duration=duration,
                    keep_excessive_supervisions=keep_excessive_supervisions
                ))
        return CutSet.from_cuts(new_cuts)

    def sample(self, n_cuts: int = 1) -> Union[AnyCut, 'CutSet']:
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

    def perturb_speed(self, factor: float, affix_id: bool = True) -> 'CutSet':
        return self.map(lambda cut: cut.perturb_speed(factor=factor, affix_id=affix_id))

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
                while mixed.duration < duration:
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

    def compute_and_store_features(
            self,
            extractor: FeatureExtractor,
            storage_path: Pathlike,
            num_jobs: Optional[int] = None,
            augment_fn: Optional[AugmentFn] = None,
            storage_type: Type[FW] = LilcomFilesWriter,
            executor: Optional[Executor] = None,
            mix_eagerly: bool = True,
            progress_bar: bool = True,
    ) -> 'CutSet':
        """
        Extract features for all cuts, possibly in parallel,
        and store them using the specified storage object.

        Examples:

            Extract fbank features on one machine using 8 processes,
            store each array in a separate file with lilcom compression:

            >>> cuts = CutSet(...)
            ... cuts.compute_and_store_features(
            ...     extractor=Fbank(),
            ...     storage_path='feats',
            ...     num_jobs=8
            ... )

            Extract fbank features on one machine using 8 processes,
            store arrays partitioned in 8 HDF5 files with lilcom compression:

            >>> cuts = CutSet(...)
            ... cuts.compute_and_store_features(
            ...     extractor=Fbank(),
            ...     storage_path='feats',
            ...     num_jobs=8,
            ...     storage_type=LilcomHdf5Writer
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
            ...     storage_type=LilcomHdf5Writer,
            ...     executor=Client(...)
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
            It determines how the featurs are stored to disk,
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
            returning a new ``Cut`` instance with the same ID.
            The returned ``Cut`` will not have a ``Recording`` attached.
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

        # We are now sure that "storage_path" will be the root for
        # multiple feature storages - we can create it as a directory
        storage_path = Path(storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

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
                storage_path=storage_path / f'feats-{i}',
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

    def map(self, transform_fn: Callable[[AnyCut], AnyCut]) -> 'CutSet':
        """
        Modify the cuts in this ``CutSet`` and return a new ``CutSet``.

        :param transform_fn: A callable (function) that accepts a single cut instance
            and returns a single cut instance.
        :return: a new ``CutSet`` with modified cuts.
        """

        def verified(mapped: Any) -> AnyCut:
            assert isinstance(mapped, (Cut, MixedCut, PaddingCut)), \
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

    def __contains__(self, item: Union[str, Cut, MixedCut]) -> bool:
        if isinstance(item, str):
            return item in self.cuts
        else:
            return item.id in self.cuts

    def __getitem__(self, cut_id_or_index: Union[int, str]) -> 'AnyCut':
        if isinstance(cut_id_or_index, str):
            return self.cuts[cut_id_or_index]
        # ~100x faster than list(dict.values())[index] for 100k elements
        return next(val for idx, val in enumerate(self.cuts.values()) if idx == cut_id_or_index)

    def __len__(self) -> int:
        return len(self.cuts)

    def __iter__(self) -> Iterable[AnyCut]:
        return iter(self.cuts.values())

    def __add__(self, other: 'CutSet') -> 'CutSet':
        assert not set(self.cuts.keys()).intersection(other.cuts.keys()), "Conflicting IDs when concatenating CutSets!"
        return CutSet(cuts={**self.cuts, **other.cuts})


def make_windowed_cuts_from_features(
        feature_set: FeatureSet,
        cut_duration: Seconds,
        cut_shift: Optional[Seconds] = None,
        keep_shorter_windows: bool = False
) -> CutSet:
    """
    Converts a FeatureSet to a CutSet by traversing each Features object in - possibly overlapping - windows, and
    creating a Cut out of that area. By default, the last window in traversal will be discarded if it cannot satisfy
    the `cut_duration` requirement.

    :param feature_set: a FeatureSet object.
    :param cut_duration: float, duration of created Cuts in seconds.
    :param cut_shift: optional float, specifies how many seconds are in between the starts of consecutive windows.
        Equals `cut_duration` by default.
    :param keep_shorter_windows: bool, when True, the last window will be used to create a Cut even if
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
                Cut(
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
        reference_cut: AnyCut,
        mixed_in_cut: AnyCut,
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


def append(
        left_cut: AnyCut,
        right_cut: AnyCut,
        snr: Optional[Decibels] = None
) -> MixedCut:
    """Helper method for functional-style appending of Cuts."""
    return left_cut.append(right_cut, snr=snr)


def mix_cuts(cuts: Iterable[AnyCut]) -> MixedCut:
    """Return a MixedCut that consists of the input Cuts mixed with each other as-is."""
    # The following is a fold (accumulate/aggregate) operation; it starts with cuts[0], and mixes it with cuts[1];
    #  then takes their mix and mixes it with cuts[2]; and so on.
    return reduce(mix, cuts)


def append_cuts(cuts: Iterable[AnyCut]) -> AnyCut:
    """Return a MixedCut that consists of the input Cuts appended to each other as-is."""
    # The following is a fold (accumulate/aggregate) operation; it starts with cuts[0], and appends cuts[1] to it;
    #  then takes their it concatenation and appends cuts[2] to it; and so on.
    return reduce(append, cuts)
