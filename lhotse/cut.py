import random
import warnings
from dataclasses import dataclass, field
from functools import reduce
from math import ceil, floor, log, isclose
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np

from lhotse.audio import AudioMixer, Recording, RecordingSet
from lhotse.features import Features, FeatureSet, FeatureMixer, create_default_feature_extractor
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import (
    EPSILON,
    Decibels,
    Pathlike,
    Seconds,
    TimeSpan,
    asdict_nonull,
    load_yaml,
    overlaps,
    overspans,
    save_to_yaml,
    uuid4
)

# One of the design principles for Cuts is a maximally "lazy" implementation, e.g. when overlaying/mixing Cuts,
# we'd rather sum the feature matrices only after somebody actually calls "load_features". It helps to avoid
# an excessive storage size for data augmented in various ways.


# Helper "typedef" to artbitrary Cut type as they do not share a common base class.
# The class names are strings here so that the Python interpreter resolves them after parsing the whole file.
AnyCut = Union['Cut', 'MixedCut', 'PaddingCut']


@dataclass
class Cut:
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
        return self.features.recording_id if self.has_features else self.recording.id

    @property
    def end(self) -> Seconds:
        return round(self.start + self.duration, ndigits=3)

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
        return round(self.duration / self.frame_shift) if self.has_features else None

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

    def load_features(self, root_dir: Optional[Pathlike] = None) -> Optional[np.ndarray]:
        """
        Load the features from the underlying storage and cut them to the relevant
        [begin, duration] region of the current Cut.
        Optionally specify a `root_dir` prefix to prefix the features path with.
        """
        if self.has_features:
            return self.features.load(root_dir=root_dir, start=self.start, duration=self.duration)
        return None

    def load_audio(self, root_dir: Optional[Pathlike] = None) -> Optional[np.ndarray]:
        """
        Load the audio by locating the appropriate recording in the supplied RecordingSet.
        The audio is trimmed to the [begin, end] range specified by the Cut.
        Optionally specify a `root_dir` prefix to prefix the features path with.

        :param root_dir: optional Path prefix to find the recording in the filesystem.
        :return: a numpy ndarray with audio samples, with shape (1 <channel>, N <samples>)
        """
        if self.has_recording:
            return self.recording.load_audio(
                channels=self.channel,
                offset_seconds=self.start,
                duration_seconds=self.duration,
                root_dir=root_dir
            )
        return None

    def truncate(
            self,
            *,
            offset: Seconds = 0.0,
            duration: Optional[Seconds] = None,
            keep_excessive_supervisions: bool = True,
            preserve_id: bool = False
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
        :return: a new Cut instance.
        """
        new_start = self.start + offset
        until = offset + (duration if duration is not None else self.duration)
        new_duration = self.duration - new_start if duration is None else until - offset
        if duration is not None and isclose(new_duration, duration):
            # Because duration is a float, the subtraction is imprecise and can result in durations
            # such as 0.3999999996 instead of 0.4, which could lead to RecordingSet loading 1 audio sample too little,
            # causing trouble for users expecting matching audio tensor shapes for cuts truncated to a single duration.
            new_duration = duration
        assert new_duration > 0.0
        if new_start + new_duration > self.start + self.duration:
            new_duration = self.start + self.duration - new_start
        new_time_span = TimeSpan(start=0, end=new_duration)
        criterion = overlaps if keep_excessive_supervisions else overspans
        new_supervisions = (segment.with_offset(-offset) for segment in self.supervisions)
        return Cut(
            id=self.id if preserve_id else str(uuid4()),
            start=new_start,
            duration=new_duration,
            channel=self.channel,
            supervisions=[
                segment for segment in new_supervisions if criterion(new_time_span, segment)
            ],
            features=self.features,
            recording=self.recording
        )

    def overlay(self, other: AnyCut, offset_other_by: Seconds = 0.0, snr: Optional[Decibels] = None) -> 'MixedCut':
        """Refer to mix() documentation."""
        return mix(self, other, offset=offset_other_by, snr=snr)

    def append(self, other: AnyCut, snr: Optional[Decibels] = None) -> 'MixedCut':
        """
        Append the `other` Cut after the current Cut. Conceptually the same as `overlay` but with an offset
        matching the current cuts length. Optionally scale down (positive SNR) or scale up (negative SNR)
        the `other` cut.
        Returns a MixedCut, which only keeps the information about the mix; actual mixing is performed
        during the call to `load_features`.
        """
        return self.overlay(other=other, offset_other_by=self.duration, snr=snr)

    def pad(self, desired_duration: Seconds) -> AnyCut:
        f"""
        Return a new MixedCut, padded to `desired_seconds` duration with low-energy values in each bin.
        We use {EPSILON} for energies, or {log(EPSILON)} for log-energies.

        :param desired_duration: The cut's minimal duration after padding.
        :return: a padded MixedCut if desired_duration is greater than this cut's duration, otherwise self.
        """
        if desired_duration <= self.duration:
            return self
        padding_duration = desired_duration - self.duration
        return self.append(PaddingCut(
            id=str(uuid4()),
            duration=padding_duration,
            num_features=self.num_features if self.features is not None else None,
            num_frames=round(padding_duration / self.frame_shift) if self.features is not None else None,
            num_samples=round(padding_duration * self.sampling_rate) if self.recording is not None else None,
            sampling_rate=self.features.sampling_rate if self.features is not None else self.recording.sampling_rate,
            use_log_energy=self.features.type in ('fbank', 'mfcc') if self.features is not None else False
        ))

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


@dataclass
class PaddingCut:
    f"""
    This represents a cut filled with zeroes in the time domain, or low energy/log-energy values in the
    frequency domain. It's used to make training samples evenly sized (same duration/number of frames).

    We use {EPSILON} for energies and {log(EPSILON)} for log-energies.
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
            return np.ones((self.num_frames, self.num_features)) * value
        return None

    # noinspection PyUnusedLocal
    def load_audio(self, *args, **kwargs) -> Optional[np.ndarray]:
        if self.has_recording:
            return np.zeros((1, round(self.duration * self.sampling_rate)))
        return None

    # noinspection PyUnusedLocal
    def truncate(
            self,
            *,
            offset: Seconds = 0.0,
            duration: Optional[Seconds] = None,
            keep_excessive_supervisions: bool = True,
            preserve_id: bool = False,
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

    def overlay(
            self,
            other: AnyCut,
            offset_other_by: Seconds = 0.0,
            snr: Optional[Decibels] = None
    ) -> 'MixedCut':
        """Refer to mix() documentation."""
        return mix(self, other, offset=offset_other_by, snr=snr)

    def append(self, other: AnyCut, snr: Optional[Decibels] = None) -> 'MixedCut':
        return self.overlay(other=other, snr=snr)

    def pad(self, desired_duration: Seconds) -> 'PaddingCut':
        """
        Create a new PaddingCut with `desired_duration` when its longer than this Cuts duration.
        Helper function used in batch cut padding.

        :param desired_duration: The cuts minimal duration after padding.
        :return: self or a new PaddingCut, depending on `desired_duration`.
        """
        if desired_duration <= self.duration:
            return self
        return PaddingCut(
            id=str(uuid4()),
            duration=desired_duration,
            num_features=self.num_features,
            num_frames=round(desired_duration / self.frame_shift),
            sampling_rate=self.sampling_rate,
            use_log_energy=self.use_log_energy
        )

    @staticmethod
    def from_dict(data: dict) -> 'PaddingCut':
        return PaddingCut(**data)


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
class MixedCut:
    """
    Represents a Cut that's created from other Cuts via overlay or append operations.
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
    def duration(self) -> Seconds:
        track_durations = (track.offset + track.cut.duration for track in self.tracks)
        return max(track_durations)

    @property
    def has_features(self) -> bool:
        return self._first_non_padding_cut.has_features

    @property
    def has_recording(self) -> bool:
        return self._first_non_padding_cut.has_recording

    @property
    def num_frames(self) -> Optional[int]:
        if self.has_features:
            return round(self.duration / self._first_non_padding_cut.frame_shift)
        return None

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
        return self._first_non_padding_cut.features.type

    def overlay(
            self,
            other: AnyCut,
            offset_other_by: Seconds = 0.0,
            snr: Optional[Decibels] = None
    ) -> 'MixedCut':
        """Refer to mix() documentation."""
        return mix(self, other, offset=offset_other_by, snr=snr)

    def append(self, other: AnyCut, snr: Optional[Decibels] = None) -> 'MixedCut':
        """
        Append the `other` Cut after the current Cut. Conceptually the same as `overlay` but with an offset
        matching the current cuts length. Optionally scale down (positive SNR) or scale up (negative SNR)
        the `other` cut.
        Returns a MixedCut, which only keeps the information about the mix; actual mixing is performed
        during the call to `load_features`.
        """
        return self.overlay(other=other, offset_other_by=self.duration, snr=snr)

    def truncate(
            self,
            *,
            offset: Seconds = 0.0,
            duration: Optional[Seconds] = None,
            keep_excessive_supervisions: bool = True,
            preserve_id: bool = False,
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
                        preserve_id=preserve_id
                    ),
                    offset=track_offset,
                    snr=track.snr
                )
            )
        return MixedCut(id=str(uuid4()), tracks=new_tracks)

    def pad(self, desired_duration: Seconds) -> AnyCut:
        f"""
        Return a new MixedCut, padded to `desired_seconds` duration with low-energy values in each bin.
        We use {EPSILON} for energies, or {log(EPSILON)} for log-energies.

        :param desired_duration: The cut's minimal duration after padding.
        :return: a padded MixedCut if desired_duration is greater than this cut's duration, otherwise self.
        """
        if desired_duration <= self.duration:
            return self
        padding_duration = desired_duration - self.duration
        return self.append(PaddingCut(
            id=str(uuid4()),
            duration=padding_duration,
            num_features=self.num_features,
            # The num_frames and sampling_rate fields are tricky, because it is possible to create a MixedCut
            # from Cuts that have different sampling rates and frame shifts. In that case, we are assuming
            # that we should use the values from the reference cut, i.e. the first one in the mix.
            num_frames=round(padding_duration / self.tracks[0].cut.frame_shift),
            sampling_rate=self.tracks[0].cut.sampling_rate,
            use_log_energy=self.features_type in ('fbank', 'mfcc')
        ))

    def load_features(self, root_dir: Optional[Pathlike] = None) -> Optional[np.ndarray]:
        """Loads the features of the source cuts and overlays them on-the-fly."""
        if not self.has_features:
            return None
        first_cut = self.tracks[0].cut
        mixer = FeatureMixer(
            feature_extractor=create_default_feature_extractor(self._first_non_padding_cut.features.type),
            base_feats=first_cut.load_features(root_dir=root_dir),
            frame_shift=first_cut.frame_shift,
        )
        for track in self.tracks[1:]:
            mixer.add_to_mix(
                feats=track.cut.load_features(root_dir=root_dir),
                snr=track.snr,
                offset=track.offset
            )
        return mixer.mixed_feats

    def load_audio(self, root_dir: Optional[Pathlike] = None) -> Optional[np.ndarray]:
        """
        Loads the audios of the source cuts and mix them on-the-fly.

        :return: the mixed audio samples in an ndarray, with the shape (1, sample_num)
        """
        if not self.has_recording:
            return None
        # cuts = [track.cut for track in self.tracks]
        # unmixed_audio = [cut.load_audio(root_dir) for cut in cuts]
        mixer = AudioMixer(self.tracks[0].cut.load_audio(root_dir=root_dir))
        for track in self.tracks[1:]:
            mixer.add_to_mix(
                audio=track.cut.load_audio(root_dir=root_dir),
                snr=track.snr,
                offset=track.offset,
                sampling_rate=track.cut.sampling_rate
            )
        return mixer.mixed_audio

    @staticmethod
    def from_dict(data: dict) -> 'MixedCut':
        return MixedCut(id=data['id'], tracks=[MixTrack.from_dict(track) for track in data['tracks']])

    @property
    def _first_non_padding_cut(self) -> Cut:
        return [t.cut for t in self.tracks if not isinstance(t.cut, PaddingCut)][0]


@dataclass
class CutSet:
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

    @staticmethod
    def from_cuts(cuts: Iterable[AnyCut]) -> 'CutSet':
        return CutSet({cut.id: cut for cut in cuts})

    @staticmethod
    def from_manifests(
            feature_set: Optional[FeatureSet] = None,
            recording_set: Optional[RecordingSet] = None,
            supervision_set: Optional[SupervisionSet] = None,
    ) -> 'CutSet':
        """
        Create a CutSet from any combination of supervision, feature and recording manifests.
        At least one of ``recording_set`` or ``feature_set`` is required.
        The Cut boundaries correspond to those found in the ``feature_set``, when available,
        otherwise to those found in the ``recording_set``
        When a ``supervision_set`` is provided, we'll attach to the Cut all supervisions that
        have a matching recording ID and are fully contained in the Cut's boundaries.
        """
        assert feature_set is not None or recording_set is not None, \
            "At least one of feature_set and recording_set has to be provided."
        sup_ok, feat_ok, rec_ok = supervision_set is not None, feature_set is not None, recording_set is not None
        if feat_ok:
            # Case I: Features are provided.
            # Use features to determine the cut boundaries and attach recordings and supervisions as available.
            return CutSet.from_cuts(
                Cut(
                    id=str(uuid4()),
                    start=features.start,
                    duration=features.duration,
                    channel=features.channel_id,
                    features=features,
                    recording=recording_set[features.recording_id] if rec_ok else None,
                    # The supervisions' start times are adjusted if the features object starts at time other than 0s.
                    supervisions=list(supervision_set.find(
                        recording_id=features.recording_id,
                        channel=features.channel_id,
                        start_after=features.start,
                        end_before=features.end,
                        adjust_offset=True
                    )) if sup_ok else []
                )
                for features in feature_set
            )
        # Case II: Recordings are provided (and features are not).
        # Use recordings to determine the cut boundaries.
        return CutSet.from_cuts(
            Cut(
                id=str(uuid4()),
                start=0,
                duration=recording.duration_seconds,
                channel=channel,
                recording=recording,
                supervisions=list(supervision_set.find(
                    recording_id=recording.id,
                    channel=channel
                )) if sup_ok else []
            )
            for recording in recording_set
            # A single cut always represents a single channel. When a recording has multiple channels,
            # we create a new cut for each channel separately.
            for channel in recording.channel_ids
        )

    @staticmethod
    def from_yaml(path: Pathlike) -> 'CutSet':
        raw_cuts = load_yaml(path)

        def deserialize_one(raw_cut: dict) -> AnyCut:
            cut_type = raw_cut.pop('type')
            if cut_type == 'Cut':
                return Cut.from_dict(raw_cut)
            if cut_type == 'MixedCut':
                return MixedCut.from_dict(raw_cut)
            raise ValueError(f"Unexpected cut type during deserialization: '{cut_type}'")

        return CutSet.from_cuts(deserialize_one(cut) for cut in raw_cuts)

    def to_yaml(self, path: Pathlike):
        data = [{**asdict_nonull(cut), 'type': type(cut).__name__} for cut in self]
        save_to_yaml(data, path)

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
        return CutSet.from_cuts(
            cut.truncate(offset=segment.start, duration=segment.duration)
            for cut in self
            for segment in cut.supervisions
        )

    def pad(
            self,
            desired_duration: Seconds = None,
    ) -> 'CutSet':
        """
        Return a new CutSet with Cuts padded to `desired_duration` in seconds.
        Cuts longer than `desired_duration` will not be affected.
        Cuts will be padded to the right (i.e. after the signal).
        :param desired_duration: The cuts minimal duration after padding.
        When not specified, we'll choose the duration of the longest cut in the CutSet.
        :return: A padded CutSet.
        """
        if desired_duration is None:
            desired_duration = max(cut.duration for cut in self)
        return CutSet.from_cuts(cut.pad(desired_duration=desired_duration) for cut in self)

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
                    channel=features.channel_id,
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
        assert reference_cut.num_features == mixed_in_cut.num_features, "Cannot overlay cuts with different feature " \
                                                                        "dimensions. "
    assert offset <= reference_cut.duration, f"Cannot overlay cut '{mixed_in_cut.id}' with offset {offset}," \
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
                offset=track.offset + offset,
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
    """Return a MixedCut that consists of the input Cuts overlayed on each other as-is."""
    # The following is a fold (accumulate/aggregate) operation; it starts with cuts[0], and overlays it with cuts[1];
    #  then takes their mix and overlays it with cuts[2]; and so on.
    return reduce(mix, cuts)


def append_cuts(cuts: Iterable[AnyCut]) -> AnyCut:
    """Return a MixedCut that consists of the input Cuts appended to each other as-is."""
    # The following is a fold (accumulate/aggregate) operation; it starts with cuts[0], and appends cuts[1] to it;
    #  then takes their it concatenation and appends cuts[2] to it; and so on.
    return reduce(append, cuts)
