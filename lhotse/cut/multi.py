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
class MultiCut(Cut):
    """
    :class:`~lhotse.cut.MultiCut` is a :class:`~lhotse.cut.Cut` that is analogous to the MonoCut.
    While MonoCut represents a single channel of a recording, MultiCut represents multi-channel
    recordings where supervisions may or may not be shared across channels.
    It is intended to be used to store, for example, segments of a microphone array recording.
    The following diagrams illustrate some examples for MultiCut usage:

    >>> 2-channel telephone recording with 2 supervisions, one for each channel (e.g., Switchboard):


                  ╔══════════════════════════════  MultiCut  ═════════════════╗
                  ║ ┌──────────────────────────┐                              ║
     Channel 1  ──╬─│   Hello this is John.    │──────────────────────────────╬────────
                  ║ └──────────────────────────┘                              ║
                  ║                               ┌──────────────────────────┐║
     Channel 2  ──╬───────────────────────────────│ Hey, John. How are you?  │╠────────
                  ║                               └──────────────────────────┘║
                  ╚═══════════════════════════════════════════════════════════╝

    >>> Multi-array multi-microphone recording with shared supervisions (e.g., CHiME-6),
    along with close-talk microphones (A and B are distant arrays, C is close-talk):

               ╔═══════════════════════════════════════════════════════════════════════════╗
               ║ ┌───────────────────┐                         ┌───────────────────┐       ║
       A-1   ──╬─┤                   ├─────────────────────────┤                   ├───────╬─
               ║ │ What did you do?  │                         │I cleaned my room. │       ║
       A-2   ──╬─┤                   ├─────────────────────────┤                   ├───────╬─
               ║ └───────────────────┘  ┌───────────────────┐  └───────────────────┘       ║
       B-1   ──╬────────────────────────┤Yeah, we were going├──────────────────────────────╬─
               ║                        │   to the mall.    │                              ║
       B-2   ──╬────────────────────────┤                   ├──────────────────────────────╬─
               ║                        └───────────────────┘        ┌───────────────────┐ ║
        C    ──╬─────────────────────────────────────────────────────┤      Right.       ├─╬─
               ║                                                     └───────────────────┘ ║
               ╚════════════════════════════════  MultiCut  ═══════════════════════════════╝

    By definition, a MultiCut has the same attributes as a MonoCut. The key difference is that
    the Recording object has multiple channels, and the Supervision objects may correspond to
    any of these channels. The channels that the MultiCut can be a subset of the Recording
    channels, but must be a superset of the Supervision channels.

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

            >>> cut = MultiCut('cut1', start=0, duration=4, channel=0)
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

    def has(self, field: str) -> bool:
        if field == "recording":
            return self.has_recording
        elif field == "features":
            return self.has_features
        else:
            return self.custom is not None and field in self.custom

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
    def num_channels(self) -> Optional[int]:
        return len(self.channel) if self.has_recording else None

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
    def load_features(self, channel: int = 0) -> Optional[np.ndarray]:
        """
        Load the features from the underlying storage and cut them to the relevant
        [begin, duration] region of the current MonoCut. We can also specify which
        channel to load (by default, we load the first channel).
        """
        if self.has_features:
            feats = self.features.load(
                start=self.start, duration=self.duration, channel_id=channel
            )
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
    def load_audio(
        self, channel: Optional[Union[int, List[int]]] = None
    ) -> Optional[np.ndarray]:
        """
        Load the audio by locating the appropriate recording in the supplied RecordingSet.
        The audio is trimmed to the [begin, end] range specified by the MultiCut.

        :return: a numpy ndarray with audio samples, with shape (C <channel>, N <samples>)
        """
        if self.has_recording:
            return self.recording.load_audio(
                channels=self.channel if channel is None else channel,
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
    ) -> "MultiCut":
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

    def attach_tensor(
        self,
        name: str,
        data: Union[np.ndarray, torch.Tensor],
        frame_shift: Optional[Seconds] = None,
        temporal_dim: Optional[int] = None,
        compressed: bool = False,
    ) -> "MultiCut":
        """
        Attach a tensor to this MultiCut, described with an :class:`~lhotse.array.Array` manifest.
        The attached data is stored in-memory for later use, and can be accessed by
        calling ``cut.load_<name>()`` or :meth:`cut.load_custom`.

        This is useful if you want actions such as truncate/pad to propagate to the tensor, e.g.::

            >>> cut = MonoCut(id="c1", start=2, duration=8, ...)
            >>> cut = cut.attach_tensor(
            ...     "alignment",
            ...     torch.tensor([0, 0, 0, ...]),
            ...     frame_shift=0.1,
            ...     temporal_dim=0,
            ... )
            >>> half_alignment = cut.truncate(duration=4.0).load_alignment()

        .. note:: This object can't be stored in JSON/JSONL manifests anymore.

        :param name: attribute under which the data can be found.
        :param data: PyTorch tensor or numpy array.
        :param frame_shift: Optional float, when the array has a temporal dimension
            it indicates how much time has passed between the starts of consecutive frames
            (expressed in seconds).
        :param temporal_dim: Optional int, when the array has a temporal dimension,
            it indicates which dim to interpret as temporal.
        :param compressed: When True, we will apply lilcom compression to the array.
            Only applicable to arrays of floats.
        :return:
        """
        from lhotse.features.io import MemoryLilcomWriter, MemoryRawWriter

        cpy = fastcopy(
            self, custom=self.custom.copy() if self.custom is not None else {}
        )
        writer = MemoryLilcomWriter() if compressed else MemoryRawWriter()
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        with writer:
            setattr(
                cpy,
                name,
                writer.store_array(
                    key=cpy.id,
                    value=data,
                    frame_shift=frame_shift,
                    temporal_dim=temporal_dim,
                    start=cpy.start,
                ),
            )
        return cpy

    def drop_features(self) -> "MultiCut":
        """Return a copy of the current :class:`.MultiCut`, detached from ``features``."""
        assert (
            self.has_recording
        ), f"Cannot detach features from a MultiCut with no Recording (cut ID = {self.id})."
        return fastcopy(self, features=None)

    def drop_recording(self) -> "MultiCut":
        """Return a copy of the current :class:`.MonoCut`, detached from ``recording``."""
        assert (
            self.has_features
        ), f"Cannot detach recording from a MultiCut with no Features (cut ID = {self.id})."
        return fastcopy(self, recording=None)

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

    def fill_supervision(
        self, add_empty: bool = True, shrink_ok: bool = False
    ) -> "MultiCut":
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
            if (old_sup.start < 0 or old_sup.end > self.end) and not shrink_ok:
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
    ) -> "MultiCut":
        """
        Returns a new MultiCut that is a sub-region of the current MultiCut.

        Note that no operation is done on the actual features or recording -
        it's only during the call to :meth:`MultiCut.load_features` / :meth:`MultiCut.load_audio`
        when the actual changes happen (a subset of features/audio is loaded).

        .. hint::

            To extend a cut by a fixed duration, use the :meth:`MultiCut.extend_by` method.

        :param offset: float (seconds), controls the start of the new cut relative to the current MultiCut's start.
            E.g., if the current MultiCut starts at 10.0, and offset is 2.0, the new start is 12.0.
        :param duration: optional float (seconds), controls the duration of the resulting MultiCut.
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
            # When the end of the MultiCut has been exceeded, trim the new duration to not exceed the old MultiCut's end.
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
        pad_silence: bool = True,
    ) -> "Cut":
        """
        Returns a new MultiCut/MixedCut that is an extended region of the current MultiCut by extending
        the cut by a fixed duration in the specified direction.

        Note that no operation is done on the actual features or recording -
        it's only during the call to :meth:`MultiCut.load_features` / :meth:`MultiCut.load_audio`
        when the actual changes happen (an extended version of features/audio is loaded).

        .. hint::

            This method extends a cut by a given duration, either to the left or to the right (or both), using
            the "real" content of the recording that the cut is part of. For example, a MonoCut spanning
            the region from 2s to 5s in a recording, when extended by 2s to the right, will now span
            the region from 2s to 7s in the same recording (provided the recording length exceeds 7s).
            If the recording is shorter, additional silence will be padded to achieve the desired duration
            by default. This behavior can be changed by setting ``pad_silence=False``.
            Also see :meth:`MultiCut.pad` which pads a cut "to" a specified length.
            To "truncate" a cut, use :meth:`MonoCut.truncate`.

        .. hint::

            If `pad_silence` is set to False, then the cut will be extended only as much as allowed
            within the recording's boundary.

        .. hint::

            If `direction` is "both", the resulting cut will be extended by the specified duration in
            both directions. This is different from the usage in :meth:`MultiCut.pad` where a padding
            equal to 0.5*duration is added to both sides.

        :param duration: float (seconds), specifies the duration by which the cut should be extended.
        :param direction: string, 'left', 'right' or 'both'. Determines whether to extend on the left,
            right, or both sides. If 'both', extend on both sides by the duration specified in `duration`.
        :param preserve_id: bool. Should the extended cut keep the same ID or get a new, random one.
        :param pad_silence: bool. Should the cut be padded with silence if the recording is shorter than
            the desired duration. If False, the cut will be extended only as much as allowed within the
            recording's boundary.
        :return: a new MultiCut instance.
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

    def resample(self, sampling_rate: int, affix_id: bool = False) -> "MultiCut":
        """
        Return a new ``MultiCut`` that will lazily resample the audio while reading it.
        This operation will drop the feature manifest, if attached.
        It does not affect the supervision.

        :param sampling_rate: The new sampling rate.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the current ``MultiCut``.
        """
        assert self.has_recording, "Cannot resample a MultiCut without Recording."
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

    def perturb_speed(self, factor: float, affix_id: bool = True) -> "MultiCut":
        """
        Return a new ``MultiCut`` that will lazily perturb the speed while loading audio.
        The ``num_samples``, ``start`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.
        We are also updating the time markers of the underlying ``Recording`` and the supervisions.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``MultiCut.id`` field
            by affixing it with "_sp{factor}".
        :return: a modified copy of the current ``MultiCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb speed on a MultiCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb speed on a MultiCut that references pre-computed features. "
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

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> "MultiCut":
        """
        Return a new ``MultiCut`` that will lazily perturb the tempo while loading audio.

        Compared to speed perturbation, tempo preserves pitch.
        The ``num_samples``, ``start`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.
        We are also updating the time markers of the underlying ``Recording`` and the supervisions.

        :param factor: The tempo will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``MultiCut.id`` field
            by affixing it with "_tp{factor}".
        :return: a modified copy of the current ``MultiCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb speed on a MultiCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb tempo on a MultiCut that references pre-computed features. "
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

    def perturb_volume(self, factor: float, affix_id: bool = True) -> "MultiCut":
        """
        Return a new ``MonoCut`` that will lazily perturb the volume while loading audio.

        :param factor: The volume will be adjusted this many times (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``MultiCut.id`` field
            by affixing it with "_vp{factor}".
        :return: a modified copy of the current ``MultiCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb volume on a MultiCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb volume on a MultiCut that references pre-computed features. "
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
    ) -> "MultiCut":
        """
        Return a new ``MultiCut`` that will convolve the audio with the provided impulse response.
        If the `rir_recording` is multi-channel, the `rir_channels` argument determines which channels
        will be used. This list must be of the same length as the number of channels in the `MultiCut`.

        If no ``rir_recording`` is provided, we will generate an impulse response using a fast random
        generator (https://arxiv.org/abs/2208.04101), only if the MultiCut has exactly one channel.
        At the moment we do not support simulation of multi-channel impulse responses.

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
        ), "Cannot apply reverberation on a MultiCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to reverberate a MultiCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "reverberation."
            )
            self.features = None

        if rir_recording is None:
            assert self.num_channels == 1, (
                "We do not support reverberation simulation for multi-channel recordings. "
                "Please provide an impulse response."
            )
            rir_channels = [0]
        else:
            assert all(
                c < rir_recording.num_channels for c in rir_channels
            ), "Invalid channel index in `rir_channels`."

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

    def map_supervisions(
        self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]
    ) -> "MultiCut":
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
    ) -> "MultiCut":
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
    ) -> "MultiCut":
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
    def from_mono(cuts: Union[Cut, Iterable[Cut]]) -> "MultiCut":
        """
        Convert one or more MonoCut to a MultiCut. If multiple cuts are provided, they
        must match in all fields except the channel. We will not perform the check here,
        instead we will just take the first cut as a reference and copy the fields from it.

        :param cut: the input cut.
        :return: a MultiCut with a single track.
        """
        if isinstance(cuts, Cut):
            data = cuts.to_dict()
            data["channel"] = [data["channel"]]  # convert channel to list
            return MultiCut.from_dict(data)
        else:
            channels = sorted(set(c.channel for c in cuts))
            data = cuts[0].to_dict()
            data["channel"] = channels
            return MultiCut.from_dict(data)

    @staticmethod
    def from_dict(data: dict) -> "MultiCut":
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

        return MultiCut(
            **data,
            features=features,
            recording=recording,
            supervisions=[SupervisionSegment.from_dict(s) for s in supervision_infos],
        )

    def with_features_path_prefix(self, path: Pathlike) -> "MultiCut":
        if not self.has_features:
            return self
        return fastcopy(self, features=self.features.with_path_prefix(path))

    def with_recording_path_prefix(self, path: Pathlike) -> "MultiCut":
        if not self.has_recording:
            return self
        return fastcopy(self, recording=self.recording.with_path_prefix(path))
