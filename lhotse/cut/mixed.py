import logging
import warnings
from dataclasses import dataclass
from functools import partial, reduce
from io import BytesIO
from operator import add
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from intervaltree import IntervalTree

from lhotse.audio import Recording, VideoInfo, get_audio_duration_mismatch_tolerance
from lhotse.audio.backend import save_flac_file
from lhotse.audio.mixer import AudioMixer, VideoMixer, audio_energy
from lhotse.augmentation import (
    AudioTransform,
    AugmentFn,
    LoudnessNormalization,
    ReverbWithImpulseResponse,
)
from lhotse.cut.base import Cut
from lhotse.cut.data import DataCut
from lhotse.cut.padding import PaddingCut
from lhotse.features import (
    FeatureExtractor,
    FeatureMixer,
    create_default_feature_extractor,
)
from lhotse.features.io import FeaturesWriter
from lhotse.supervision import SupervisionSegment
from lhotse.utils import (
    DEFAULT_PADDING_VALUE,
    LOG_EPSILON,
    Decibels,
    Pathlike,
    Seconds,
    add_durations,
    compute_num_frames,
    compute_num_samples,
    fastcopy,
    hash_str_to_int,
    merge_items_with_delimiter,
    overlaps,
    perturb_num_samples,
    rich_exception_info,
    uuid4,
)


@dataclass
class MixTrack:
    """
    Represents a single track in a mix of Cuts. Points to a specific DataCut or PaddingCut and holds information on
    how to mix it with other Cuts, relative to the first track in a mix.
    """

    cut: Union[DataCut, PaddingCut]
    type: str = None
    offset: Seconds = 0.0
    snr: Optional[Decibels] = None

    def __post_init__(self):
        self.type = type(self.cut).__name__

    @staticmethod
    def from_dict(data: dict):
        from .set import deserialize_cut

        # Take out `type` from data dict and put it into the `cut` dict.
        cut_dict = data.pop("cut")
        cut_dict["type"] = data.pop("type")
        return MixTrack(deserialize_cut(cut_dict), **data)


@dataclass
class MixedCut(Cut):
    """
    :class:`~lhotse.cut.MixedCut` is a :class:`~lhotse.cut.Cut` that actually consists of multiple other cuts.
    Its primary purpose is to allow time-domain and feature-domain augmentation via mixing the training cuts
    with noise, music, and babble cuts. The actual mixing operations are performed on-the-fly.

    Internally, :class:`~lhotse.cut.MixedCut` holds other cuts in multiple tracks (:class:`~lhotse.cut.MixTrack`),
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

    .. note:: MixedCut is different from MultiCut, which is intended to represent multi-channel recordings
        that share the same supervisions.

    .. note:: Each track in a MixedCut can be either a MonoCut, MultiCut, or PaddingCut.

    .. note:: The ``transforms`` field is a list of dictionaries that describe the transformations
        that should be applied to the track after mixing.

    See also:

        - :class:`lhotse.cut.Cut`
        - :class:`lhotse.cut.MonoCut`
        - :class:`lhotse.cut.MultiCut`
        - :class:`lhotse.cut.CutSet`
    """

    id: str
    tracks: List[MixTrack]
    transforms: Optional[List[Dict]] = None

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
    def channel(self) -> Union[int, List[int]]:
        num_channels = self.num_channels
        return list(range(num_channels)) if num_channels > 1 else 0

    @property
    def has_features(self) -> bool:
        return self._first_non_padding_cut.has_features

    @property
    def has_recording(self) -> bool:
        return self._first_non_padding_cut.has_recording

    @property
    def has_video(self) -> bool:
        return self._first_non_padding_cut.has_video

    def has(self, field: str) -> bool:
        return self._first_non_padding_cut.has(field)

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
    def num_channels(self) -> Optional[int]:
        # PaddingCut and MonoCut have 1 channel each, MultiCut has N. We don't support MixedCut with MixedCut tracks.
        return max(track.cut.num_channels for track in self.tracks)

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
            ) = self._assert_one_data_cut_with_attr_and_return_it_with_track_index(name)
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
        ) = self._assert_one_data_cut_with_attr_and_return_it_with_track_index(name)

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

    def _assert_one_data_cut_with_attr_and_return_it_with_track_index(
        self,
        attr_name: str,
    ) -> Tuple[int, DataCut]:
        # TODO(pzelasko): consider relaxing this condition to
        #                 supporting mixed cuts that are not overlapping
        non_padding_cuts = [
            (idx, t.cut)
            for idx, t in enumerate(self.tracks)
            if isinstance(t.cut, DataCut)
        ]
        non_padding_cuts_with_custom_attr = [
            (idx, cut)
            for idx, cut in non_padding_cuts
            if cut.custom is not None and attr_name in cut.custom
        ]
        assert len(non_padding_cuts_with_custom_attr) == 1, (
            f"This MixedCut has {len(non_padding_cuts_with_custom_attr)} non-padding cuts "
            f"with a custom attribute '{attr_name}'. We currently don't support mixing custom attributes. "
            f"Consider dropping the attribute on all but one of DataCuts. Problematic cut:\n{self}"
        )
        non_padding_idx, mono_cut = non_padding_cuts_with_custom_attr[0]
        return non_padding_idx, mono_cut

    def move_to_memory(
        self,
        audio_format: str = "flac",
        load_audio: bool = True,
        load_features: bool = True,
        load_custom: bool = True,
    ) -> "MixedCut":
        """
        Load data (audio, features, or custom arrays) into memory and attach them
        to a copy of the manifest. This is useful when you want to store cuts together
        with the actual data in some binary format that enables sequential data reads.

        Audio is encoded with ``audio_format`` (compatible with ``torchaudio.save``),
        floating point features are encoded with lilcom, and other arrays are pickled.
        """
        return fastcopy(
            self,
            tracks=[
                fastcopy(
                    t,
                    cut=t.cut.move_to_memory(
                        audio_format=audio_format,
                        load_audio=load_audio,
                        load_features=load_features,
                        load_custom=load_custom,
                    ),
                )
                for t in self.tracks
            ],
        )

    def to_mono(
        self,
        encoding: str = "flac",
        bits_per_sample: Optional[int] = 16,
        **kwargs,
    ) -> "Cut":
        """
        Convert this MixedCut to a MonoCut by mixing all tracks and channels into a single one.
        The result audio array is stored in memory, and can be saved to disk by calling
        ``cut.save_audio(path, ...)`` on the result.

        .. hint:: the resulting MonoCut will have ``custom`` field populated with the
            ``custom`` value from the first track of the MixedCut.

        :param encoding: Audio encoding argument supported by ``torchaudio.save``. See
            https://pytorch.org/audio/stable/backend.html#save (sox_io backend) and
            https://pytorch.org/audio/stable/backend.html#id3 (soundfile backend) for more details.
        :param bits_per_sample: Audio bits_per_sample argument supported by ``torchaudio.save``. See
            https://pytorch.org/audio/stable/backend.html#save (sox_io backend) and
            https://pytorch.org/audio/stable/backend.html#id3 (soundfile backend) for more details.
        :return: a new MonoCut instance.
        """
        samples = self.load_audio(mono_downmix=True)
        stream = BytesIO()
        save_flac_file(
            stream,
            samples,
            self.sampling_rate,
            format=encoding,
            bits_per_sample=bits_per_sample,
        )
        recording = Recording.from_bytes(stream.getvalue(), recording_id=self.id)
        return fastcopy(
            recording.to_cut(),
            supervisions=[fastcopy(s, channel=0) for s in self.supervisions],
            custom=self.tracks[0].cut.custom,
        )

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
                # Omit a Cut that ends before the truncation offset.
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

            # Compute the new Cut's duration after trimming the start and the end.
            new_duration = add_durations(
                track.cut.duration,
                -cut_offset,
                -cut_duration_decrease,
                sampling_rate=self.sampling_rate,
            )
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
                        _supervisions_index=_supervisions_index,
                    ),
                    offset=track_offset,
                    snr=track.snr,
                )
            )

        # Edge case: no tracks with data left after truncation. This can happen if we truncated an offset region.
        # In this case, return a PaddingCut of the requested duration
        if len([t for t in new_tracks if not isinstance(t.cut, PaddingCut)]) == 0:
            return PaddingCut(
                id=self.id if preserve_id else str(uuid4()),
                duration=duration,
                sampling_rate=self.sampling_rate,
                feat_value=0.0,
                num_samples=compute_num_samples(duration, self.sampling_rate),
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
        pad_silence: bool = True,
    ) -> "MixedCut":
        """
        This raises a ValueError since extending a MixedCut is not defined.

        :param duration: float (seconds), duration (in seconds) to extend the MixedCut.
        :param direction: string, 'left', 'right' or 'both'. Determines whether to extend on the left,
            right, or both sides. If 'both', extend on both sides by the duration specified in `duration`.
        :param preserve_id: bool. Should the extended cut keep the same ID or get a new, random one.
        :param pad_silence: bool. See usage in `lhotse.cut.MonoCut.extend_by`.
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
        from .set import pad

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
        ), "Cannot perturb speed on a MixedCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb speed on a MixedCut that references pre-computed features. "
                "The feature manifest(s) will be detached, as we do not support feature-domain "
                "speed perturbation."
            )
        return MixedCut(
            id=f"{self.id}_sp{factor}" if affix_id else self.id,
            tracks=[
                fastcopy(
                    track,
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
        ), "Cannot perturb tempo on a MixedCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb tempo on a MixedCut that references pre-computed features. "
                "The feature manifest(s) will be detached, as we do not support feature-domain "
                "tempo perturbation."
            )
        return MixedCut(
            id=f"{self.id}_tp{factor}" if affix_id else self.id,
            tracks=[
                fastcopy(
                    track,
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
        ), "Cannot perturb volume on a MixedCut without Recording."
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

    def normalize_loudness(
        self, target: float, mix_first: bool = True, affix_id: bool = False
    ) -> "DataCut":
        """
        Return a new ``MixedCut`` that will lazily apply loudness normalization.

        :param target: The target loudness in dBFS.
        :param mix_first: If true, we will mix the underlying cuts before applying
            loudness normalization. If false, we cannot guarantee that the resulting
            cut will have the target loudness.
        :param affix_id: When true, we will modify the ``DataCut.id`` field
            by affixing it with "_ln{target}".
        :return: a modified copy of the current ``DataCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot apply loudness normalization on a MixedCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to normalize loudness on a MixedCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "loudness normalization."
            )
            self.features = None

        if mix_first:
            transforms = self.transforms.copy() if self.transforms is not None else []
            transforms.append(LoudnessNormalization(target=target).to_dict())
            return fastcopy(
                self,
                id=f"{self.id}_ln{target}" if affix_id else self.id,
                transforms=transforms,
            )
        else:
            return MixedCut(
                id=f"{self.id}_ln{target}" if affix_id else self.id,
                tracks=[
                    fastcopy(
                        track,
                        cut=track.cut.normalize_loudness(
                            target=target, affix_id=affix_id
                        ),
                    )
                    for track in self.tracks
                ],
            )

    def reverb_rir(
        self,
        rir_recording: Optional["Recording"] = None,
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: List[int] = [0],
        room_rng_seed: Optional[int] = None,
        source_rng_seed: Optional[int] = None,
        mix_first: bool = True,
    ) -> "MixedCut":
        """
        Return a new ``MixedCut`` that will convolve the audio with the provided impulse response.
        If no ``rir_recording`` is provided, we will generate an impulse response using a fast random
        generator (https://arxiv.org/abs/2208.04101).

        :param rir_recording: The impulse response to use for convolving.
        :param normalize_output: When true, output will be normalized to have energy as input.
        :param early_only: When true, only the early reflections (first 50 ms) will be used.
        :param affix_id: When true, we will modify the ``MixedCut.id`` field
            by affixing it with "_rvb".
        :param rir_channels: The channels of the impulse response to use. By default, first channel is used.
            If only one channel is specified, all tracks will be convolved with this channel. If a list
            is provided, it must contain as many channels as there are tracks such that each track will
            be convolved with one of the specified channels.
        :param room_rng_seed: Seed for the room configuration.
        :param source_rng_seed: Seed for the source position.
        :param mix_first: When true, the mixing will be done first before convolving with the RIR.
            This effectively means that all tracks will be convolved with the same RIR. If you
            are simulating multi-speaker mixtures, you should set this to False.
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

        assert rir_recording is None or all(
            c < rir_recording.num_channels for c in rir_channels
        ), "Invalid channel index in `rir_channels`."

        assert len(rir_channels) == 1 or len(rir_channels) == len(
            self.tracks
        ), "Invalid number of channels in `rir_channels`, must be either 1 or equal to the number of tracks."

        # There are 2 ways to apply RIRs:
        # 1. Mix the tracks first, then apply RIRs. This is same as applying the same RIR
        #    to all tracks. It does not make sense if all tracks belong to different speakers,
        #    but it is useful for cases when we have a mixture of MonoCut and PaddingCut,
        #    and we want to apply the same RIR to all of them.
        # 2. Apply RIRs to each track separately. This is useful when we want to simulate
        #    different speakers in the same room.

        # First simulate the room config (will only be used if RIR is not provided)
        uuid4_str = str(uuid4())
        # The room RNG seed is based on the cut ID. This ensures that all tracks in the
        # mixed cut will have the same room configuration.
        if room_rng_seed is None:
            room_rng_seed = hash_str_to_int(uuid4_str + self.id)
        # The source RNG seed is based on the track ID. This ensures that each track
        # will have a different source position.
        source_rng_seeds = [source_rng_seed] * len(self.tracks)
        if source_rng_seed is None:
            source_rng_seeds = [
                hash_str_to_int(uuid4_str + track.cut.id) for track in self.tracks
            ]
            source_rng_seed = source_rng_seeds[0]

        # Apply same RIR to all tracks after mixing (default)
        if mix_first:
            if rir_recording is None:
                from lhotse.augmentation.utils import FastRandomRIRGenerator

                rir_generator = FastRandomRIRGenerator(
                    sr=self.sampling_rate,
                    room_seed=room_rng_seed,
                    source_seed=source_rng_seed,
                )
            else:
                rir_generator = None

            transforms = self.transforms.copy() if self.transforms is not None else []
            transforms.append(
                ReverbWithImpulseResponse(
                    rir=rir_recording,
                    normalize_output=normalize_output,
                    early_only=early_only,
                    rir_channels=rir_channels if rir_channels is not None else [0],
                    rir_generator=rir_generator,
                ).to_dict()
            )
            return fastcopy(
                self,
                id=f"{self.id}_rvb" if affix_id else self.id,
                transforms=transforms,
            )

        # Apply RIRs to each track separately. Note that we do not pass a `mix_first`
        # argument below since it is True by default.

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
                        room_rng_seed=room_rng_seed,
                        source_rng_seed=seed,
                    ),
                )
                for track, channel, seed in zip(
                    self.tracks, rir_channels, source_rng_seeds
                )
            ],
        )

    @rich_exception_info
    def load_features(self, mixed: bool = True) -> Optional[np.ndarray]:
        """
        Loads the features of the source cuts and mixes them on-the-fly.

        :param mixed: when True (default), the features are mixed together (as defined in
            the mixing function for the extractor). This could result in either a 2D or 3D
            array. For example, if all underlying tracks are single-channel, the output
            will be a 2D array of shape (num_frames, num_features). If any of the tracks
            are multi-channel, the output may be a 3D array of shape (num_frames, num_features,
            num_channels).
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
            first_cut_feats = first_cut.load_features()
            if first_cut_feats.ndim == 2:
                # 2D features
                feats = np.ones((self.num_frames, self.num_features)) * padding_val
            else:
                # 3D features
                feats = (
                    np.ones(
                        (self.num_frames, self.num_features, first_cut_feats.shape[-1])
                    )
                    * padding_val
                )
            feats[: first_cut.num_frames, ...] = first_cut_feats
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
    def load_audio(
        self, mixed: bool = True, mono_downmix: bool = False
    ) -> Optional[np.ndarray]:
        """
        Loads the audios of the source cuts and mix them on-the-fly.

        :param mixed: When True (default), returns a mix of the underlying tracks. This will
            return a numpy array with shape ``(num_channels, num_samples)``, where ``num_channels``
            is determined by the ``num_channels`` property of the MixedCut. Otherwise returns a
            numpy array with the number of channels equal to the total number of channels
            across all tracks in the MixedCut. For example, if it contains a MultiCut with 2
            channels and a MonoCut with 1 channel, the returned array will have shape
            ``(3, num_samples)``.
        :param mono_downmix: If the MixedCut contains > 1 channels (for e.g. when one of its tracks
            is a MultiCut), this parameter controls whether the returned array will be down-mixed
            to a single channel. This down-mixing is done by summing the channels together.
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

        mixer = AudioMixer(
            self.tracks[0].cut.load_audio(),
            sampling_rate=self.tracks[0].cut.sampling_rate,
            reference_energy=reference_energy,
            base_offset=self.tracks[0].offset,
        )

        for pos, track in enumerate(self.tracks[1:], start=1):
            if pos == reference_pos and reference_audio is not None:
                audio = reference_audio  # manual caching to avoid duplicated I/O
            else:
                audio = track.cut.load_audio()
            mixer.add_to_mix(
                audio=audio,
                snr=track.snr,
                offset=track.offset,
            )

        # Flattening a MixedCut without MultiCut tracks has no effect
        mono_downmix = mono_downmix and any(
            track.type == "MultiCut" for track in self.tracks
        )

        # Flattening a MixedCut without mixed=True has no effect
        mono_downmix = mono_downmix and mixed

        if mixed:
            # Off-by-one errors can happen during mixing due to imperfect float arithmetic and rounding;
            # we will fix them on-the-fly so that the manifest does not lie about the num_samples.
            audio = mixer.mixed_mono_audio if mono_downmix else mixer.mixed_audio
            tol_samples = compute_num_samples(
                get_audio_duration_mismatch_tolerance(),
                sampling_rate=self.sampling_rate,
            )
            num_samples_diff = audio.shape[1] - self.num_samples
            if 0 < num_samples_diff < tol_samples:
                audio = audio[:, : self.num_samples]
            if -tol_samples < num_samples_diff < 0:
                audio = np.pad(audio, [(0, 0), (0, -num_samples_diff)], mode="reflect")
            assert audio.shape[1] == self.num_samples, (
                f"Inconsistent number of samples in a MixedCut. Expected {self.num_samples} "
                f"but the output of mix has {audio.shape[1]}. Please report "
                f"this issue at https://github.com/lhotse-speech/lhotse/issues "
                f"showing the cut below. MixedCut:\n{self}"
            )

            # We'll apply the transforms now (if any).
            transforms = [
                AudioTransform.from_dict(params) for params in self.transforms or []
            ]
            for tfn in transforms:
                audio = tfn(audio, self.sampling_rate)
        else:
            audio = mixer.unmixed_audio

        return audio

    @property
    def video(self) -> Optional[VideoInfo]:
        if self.has_video:
            v = self._first_non_padding_cut.video
            return v.copy_with(num_frames=compute_num_samples(self.duration, v.fps))
        return None

    @rich_exception_info
    def load_video(
        self,
        with_audio: bool = True,
        mixed: bool = True,
        mono_downmix: bool = False,
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if not self.has_video:
            return None

        mixer = VideoMixer(
            self.tracks[0].cut.load_video(with_audio=False)[0],
            fps=self.video.fps,
            base_offset=self.tracks[0].offset,
        )
        for pos, track in enumerate(self.tracks[1:], start=1):
            mixer.add_to_mix(
                video=track.cut.load_video(with_audio=False)[0],
                offset=track.offset,
            )
        video = mixer.mixed_video

        if with_audio:
            # For now load audio separately to re-use the complex logic of load_audio
            # This means the same video file is potentially opened twice, but given the
            # cost of video decoding, the extra file open cost could be negligible.
            audio = self.load_audio(mixed=mixed, mono_downmix=mono_downmix)
        return video, torch.from_numpy(audio)

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
            samples = audio[idx].squeeze(0)
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

    def drop_alignments(self) -> "MixedCut":
        """Return a copy of the current :class:`.MixedCut`, detached from ``supervisions``."""
        return fastcopy(
            self,
            tracks=[fastcopy(t, cut=t.cut.drop_alignments()) for t in self.tracks],
        )

    def compute_and_store_features(
        self,
        extractor: FeatureExtractor,
        storage: FeaturesWriter,
        augment_fn: Optional[AugmentFn] = None,
        mix_eagerly: bool = True,
    ) -> DataCut:
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
            from .mono import MonoCut

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
                idx for idx, t in enumerate(self.tracks) if isinstance(t.cut, DataCut)
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
            if isinstance(track.cut, PaddingCut):
                continue
            track.cut.supervisions = [
                segment.map(transform_fn) for segment in track.cut.supervisions
            ]
        return new_mixed_cut

    def merge_supervisions(
        self,
        merge_policy: str = "delimiter",
        custom_merge_fn: Optional[Callable[[str, Iterable[Any]], Any]] = None,
    ) -> "MixedCut":
        """
        Return a copy of the cut that has all of its supervisions merged into
        a single segment.

        The new start is the start of the earliest superivion, and the new duration
        is a minimum spanning duration for all the supervisions. The text fields are
        concatenated with a whitespace.

        .. note:: If you're using individual tracks of a mixed cut, note that this transform
             drops all the supervisions in individual tracks and assigns the merged supervision
             in the first :class:`.DataCut` found in ``self.tracks``.

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
        merge_func_ = partial(
            merge_items_with_delimiter,
            delimiter="#",
            return_first=(merge_policy == "keep_first"),
        )

        # "m" stands for merged in variable names below

        if custom_merge_fn is not None:
            # Merge custom fields with the user-provided function.
            merge_custom = custom_merge_fn
        else:
            # Merge the string representations of custom fields.
            merge_custom = lambda k, vs: merge_func_(map(str, vs))

        sups = sorted(self.supervisions, key=lambda s: s.start)

        if len(sups) <= 1:
            return self

        # the sampling rate is arbitrary, ensures there are no float precision errors
        mstart = sups[0].start
        mend = sups[-1].end
        mduration = add_durations(mend, -mstart, sampling_rate=self.sampling_rate)

        custom_keys = set(
            k for s in sups if s.custom is not None for k in s.custom.keys()
        )
        alignment_keys = set(
            k for s in sups if s.alignment is not None for k in s.alignment.keys()
        )

        if any(overlaps(s1, s2) for s1, s2 in zip(sups, sups[1:])) and any(
            s.text is not None for s in sups
        ):
            warnings.warn(
                "You are merging overlapping supervisions that have text transcripts. "
                "The result is likely to be unusable if you are going to train speech "
                f"recognition models (cut id: {self.id})."
            )

        msup = SupervisionSegment(
            id=merge_func_(s.id for s in sups),
            # Make merged recording_id is a mix of recording_ids.
            recording_id=merge_func_(s.recording_id for s in sups),
            start=mstart,
            duration=mduration,
            # Hardcode -1 to indicate no specific channel, as the supervisions might have
            # come from different channels in their original recordings.
            channel=-1,
            text=" ".join(s.text for s in sups if s.text),
            speaker=merge_func_(s.speaker for s in sups if s.speaker),
            language=merge_func_(s.language for s in sups if s.language),
            gender=merge_func_(s.gender for s in sups if s.gender),
            custom={
                k: merge_custom(
                    k,
                    (
                        s.custom[k]
                        for s in sups
                        if s.custom is not None and k in s.custom
                    ),
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

        new_cut = self.drop_supervisions()
        new_cut._first_non_padding_cut.supervisions = [msup]
        return new_cut

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
        :return: a modified MixedCut
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
    def _first_non_padding_cut(self) -> DataCut:
        return self._first_non_padding_track.cut

    @property
    def _first_non_padding_track(self) -> MixTrack:
        return [t for t in self.tracks if not isinstance(t.cut, PaddingCut)][0]
