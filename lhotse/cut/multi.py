import logging
from dataclasses import dataclass, field
from functools import partial
from math import isclose
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from intervaltree import IntervalTree

from lhotse.audio import Recording
from lhotse.augmentation import AugmentFn
from lhotse.cut.base import Cut
from lhotse.cut.mono import MonoCut
from lhotse.features import FeatureExtractor, Features
from lhotse.features.io import FeaturesWriter
from lhotse.supervision import SupervisionSegment
from lhotse.utils import (
    LOG_EPSILON,
    Pathlike,
    Seconds,
    TimeSpan,
    add_durations,
    compute_num_frames,
    compute_num_samples,
    fastcopy,
    ifnone,
    measure_overlap,
    overlaps,
    overspans,
    perturb_num_samples,
    rich_exception_info,
    uuid4,
)


@dataclass
class MultiCut(MonoCut):
    """
    :class:`~lhotse.cut.MultiCut` is a :class:`~lhotse.cut.Cut` that is analogous to the MonoCut.
    While MonoCut represents a single channel of a recording, MultiCut represents multiple channels
    that share the same supervision. It is intended to be used to store, for example, segments
    of a microphone array recording.

    By definition, a MultiCut has the same attributes as a MonoCut. The key difference is that
    the Recording object has multiple channels, and the Supervision object is shared across
    multiple channels. The channels that the MultiCut pertains to is defined by the channels
    in the Supervision object, and these are the same as the ``channel`` attribute of the
    MultiCut.

    See also:

        - :class:`lhotse.cut.Cut`
        - :class:`lhotse.cut.MonoCut`
        - :class:`lhotse.cut.CutSet`
        - :class:`lhotse.cut.MixedCut`
    """

    id: str

    # Begin and duration are needed to specify which chunk of features/recording to load.
    start: Seconds
    duration: Seconds
    channel: List[int]

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

    def drop_supervisions(self, reset_channels: bool = False) -> "MultiCut":
        """Return a copy of the current :class:`.MultiCut`, detached from ``supervisions``.
        If reset_channels is True, the channel attribute will be set to use all the available
        channels in the recording.
        """
        return fastcopy(
            self,
            supervisions=[],
            channel=self.channel
            if not reset_channels
            else list(self.recording.channel_ids),
        )

    def extend_by(
        self,
        *,
        duration: Seconds,
        direction: str = "both",
        preserve_id: bool = False,
        pad_silence: bool = True,
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
            If the recording is shorter, additional silence will be padded to achieve the desired duration
            by default. This behavior can be changed by setting ``pad_silence=False``.
            Also see :meth:`MonoCut.pad` which pads a cut "to" a specified length.
            To "truncate" a cut, use :meth:`MonoCut.truncate`.

        .. hint::

            If `pad_silence` is set to False, then the cut will be extended only as much as allowed
            within the recording's boundary.

        .. hint::

            If `direction` is "both", the resulting cut will be extended by the specified duration in
            both directions. This is different from the usage in :meth:`MonoCut.pad` where a padding
            equal to 0.5*duration is added to both sides.

        :param duration: float (seconds), specifies the duration by which the cut should be extended.
        :param direction: string, 'left', 'right' or 'both'. Determines whether to extend on the left,
            right, or both sides. If 'both', extend on both sides by the duration specified in `duration`.
        :param preserve_id: bool. Should the extended cut keep the same ID or get a new, random one.
        :param pad_silence: bool. Should the cut be padded with silence if the recording is shorter than
            the desired duration. If False, the cut will be extended only as much as allowed within the
            recording's boundary.
        :return: a new MonoCut instance.
        """
        from lhotse.array import TemporalArray

        assert duration >= 0, f"Duration must be non-negative (provided {duration})."

        new_start, new_end = self.start, self.end
        pad_left, pad_right = 0, 0
        if direction == "left" or direction == "both":
            if self.start - duration < 0 and pad_silence:
                pad_left = duration - self.start
            new_start = max(self.start - duration, 0)
        if direction == "right" or direction == "both":
            if self.end + duration > self.recording.duration and pad_silence:
                pad_right = duration - (self.recording.duration - self.end)
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

        cut = fastcopy(
            self,
            id=self.id if preserve_id else str(uuid4()),
            start=new_start,
            duration=new_duration,
            supervisions=sorted(new_supervisions, key=lambda s: s.start),
            **feature_kwargs,
            custom=custom_kwargs,
        )

        # Now pad the cut on either side if needed
        if pad_left > 0:
            cut = cut.pad(
                duration=cut.duration + pad_left,
                direction="left",
                preserve_id=preserve_id,
            )
        if pad_right > 0:
            cut = cut.pad(
                duration=cut.duration + pad_right,
                direction="right",
                preserve_id=preserve_id,
            )
        return cut

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
        Return a new MultiCut, padded with zeros in the recording, and ``pad_feat_value`` in each feature bin.

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
        custom = self.custom
        if isinstance(custom, dict) and any(
            isinstance(v, Recording) for v in custom.values()
        ):
            custom = {
                k: v.resample(sampling_rate) if isinstance(v, Recording) else v
                for k, v in custom.items()
            }

        return fastcopy(
            self,
            id=f"{self.id}_rs{sampling_rate}" if affix_id else self.id,
            recording=self.recording.resample(sampling_rate),
            features=None,
            custom=custom,
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
        rir_recording: Optional["Recording"] = None,
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: List[int] = [0],
    ) -> Union["MonoCut", "MixedCut"]:
        """
        Return a new ``MonoCut`` that will convolve the audio with the provided impulse response.
        If the `rir_recording` is multi-channel, the `rir_channels` argument determines which channels
        will be used. By default, we use the first channel and return a MonoCut.

        If no ``rir_recording`` is provided, we will generate an impulse response using a fast random
        generator (https://arxiv.org/abs/2208.04101).

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

        assert rir_recording is None or all(
            c < rir_recording.num_channels for c in rir_channels
        ), "Invalid channel index in `rir_channels`."
        if len(rir_channels) == 1 or (
            rir_recording is not None and rir_recording.num_channels == 1
        ):
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
            from .mixed import MixedCut, MixTrack

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
        from .set import merge_supervisions

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
