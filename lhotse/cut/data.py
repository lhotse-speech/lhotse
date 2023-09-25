import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from decimal import ROUND_DOWN
from functools import partial
from math import isclose
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from intervaltree import IntervalTree

from lhotse.audio import Recording, VideoInfo
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
class DataCut(Cut, metaclass=ABCMeta):
    """
    :class:`~lhotse.cut.DataCut` is a base class for cuts that point to actual audio data.
    It can be either a :class:`~lhotse.cut.MonoCut` or a :class:`~lhotse.cut.MultiCut`.
    This is as opposed to :class:`~lhotse.cut.MixedCut`, which is simply an operation on
    a collection of cuts.

    See also:

        - :class:`lhotse.cut.MonoCut`
        - :class:`lhotse.cut.MultiCut`
    """

    id: str

    # Begin and duration are needed to specify which chunk of features/recording to load.
    start: Seconds
    duration: Seconds
    channel: Union[int, List[int]]

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

    def __setattr__(self, key: str, value: Any) -> None:
        """
        This magic function is called when the user tries to set an attribute.
        We use it as syntactic sugar to store custom attributes in ``self.custom``
        field, so that they can be (de)serialized later.
        Setting a ``None`` value will remove the attribute from ``custom``.
        """
        if key in self.__dataclass_fields__:
            super().__setattr__(key, value)
        else:
            custom = ifnone(self.custom, {})
            if value is None:
                custom.pop(key, None)
            else:
                custom[key] = value
            if custom:
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

    def __delattr__(self, key: str) -> None:
        """Used to support ``del cut.custom_attr`` syntax."""
        if key in self.__dataclass_fields__:
            super().__delattr__(key)
        if self.custom is None or key not in self.custom:
            raise AttributeError(f"No such member: '{key}'")
        del self.custom[key]

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

    def has_custom(self, name: str) -> bool:
        """
        Check if the Cut has a custom attribute with name ``name``.

        :param name: name of the custom attribute.
        :return: a boolean.
        """
        if self.custom is None:
            return False
        return name in self.custom

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
    def has_video(self) -> bool:
        return self.has_recording and self.recording.has_video

    @property
    def video(self) -> Optional[VideoInfo]:
        if self.has_recording:
            v = self.recording.video
            return v.copy_with(
                num_frames=compute_num_samples(
                    self.duration, v.fps, rounding=ROUND_DOWN
                )
            )
        return None

    def has(self, field: str) -> bool:
        if field == "recording":
            return self.has_recording
        elif field == "features":
            return self.has_features
        elif field == "video":
            return self.has_video
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
    @abstractmethod
    def num_channels(self) -> Optional[int]:
        ...

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
    @abstractmethod
    def load_features(self, **kwargs) -> Optional[np.ndarray]:
        ...

    @rich_exception_info
    @abstractmethod
    def load_audio(self, **kwargs) -> Optional[np.ndarray]:
        ...

    @rich_exception_info
    @abstractmethod
    def load_video(
        self, **kwargs
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        ...

    def move_to_memory(
        self,
        audio_format: str = "flac",
        load_audio: bool = True,
        load_features: bool = True,
        load_custom: bool = True,
    ) -> "Cut":
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
    ) -> "Cut":
        """
        Attach a tensor to this MonoCut, described with an :class:`~lhotse.array.Array` manifest.
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

    def drop_features(self) -> "DataCut":
        """Return a copy of the current :class:`.DataCut`, detached from ``features``."""
        assert (
            self.has_recording
        ), f"Cannot detach features from a DataCut with no Recording (cut ID = {self.id})."
        return fastcopy(self, features=None)

    def drop_recording(self) -> "DataCut":
        """Return a copy of the current :class:`.DataCut`, detached from ``recording``."""
        assert (
            self.has_features
        ), f"Cannot detach recording from a DataCut with no Features (cut ID = {self.id})."
        return fastcopy(self, recording=None)

    def drop_supervisions(self) -> "DataCut":
        """Return a copy of the current :class:`.DataCut`, detached from ``supervisions``."""
        return fastcopy(self, supervisions=[])

    def drop_alignments(self) -> "DataCut":
        """Return a copy of the current :class:`.DataCut`, detached from ``alignments``."""
        return fastcopy(
            self, supervisions=[fastcopy(s, alignment={}) for s in self.supervisions]
        )

    def fill_supervision(
        self, add_empty: bool = True, shrink_ok: bool = False
    ) -> "DataCut":
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
    ) -> "DataCut":
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
    ) -> "DataCut":
        """
        Returns a new MonoCut that is a sub-region of the current DataCut.

        Note that no operation is done on the actual features or recording -
        it's only during the call to :meth:`DataCut.load_features` / :meth:`DataCut.load_audio`
        when the actual changes happen (a subset of features/audio is loaded).

        .. hint::

            To extend a cut by a fixed duration, use the :meth:`DataCut.extend_by` method.

        :param offset: float (seconds), controls the start of the new cut relative to the current DataCut's start.
            E.g., if the current DataCut starts at 10.0, and offset is 2.0, the new start is 12.0.
        :param duration: optional float (seconds), controls the duration of the resulting DataCut.
            By default, the duration is (end of the cut before truncation) - (offset).
        :param keep_excessive_supervisions: bool. Since trimming may happen inside a SupervisionSegment,
            the caller has an option to either keep or discard such supervisions.
        :param preserve_id: bool. Should the truncated cut keep the same ID or get a new, random one.
        :param _supervisions_index: an IntervalTree; when passed, allows to speed up processing of Cuts with a very
            large number of supervisions. Intended as an internal parameter.
        :return: a new MonoCut instance. If the current DataCut is shorter than the duration, return None.
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
        pad_silence: bool = True,
    ) -> Cut:
        """
        Returns a new Cut (DataCut or MixedCut) that is an extended region of the current DataCut by extending
        the cut by a fixed duration in the specified direction.

        Note that no operation is done on the actual features or recording -
        it's only during the call to :meth:`DataCut.load_features` / :meth:`DataCut.load_audio`
        when the actual changes happen (an extended version of features/audio is loaded).

        .. hint::

            This method extends a cut by a given duration, either to the left or to the right (or both), using
            the "real" content of the recording that the cut is part of. For example, a DataCut spanning
            the region from 2s to 5s in a recording, when extended by 2s to the right, will now span
            the region from 2s to 7s in the same recording (provided the recording length exceeds 7s).
            If the recording is shorter, additional silence will be padded to achieve the desired duration
            by default. This behavior can be changed by setting ``pad_silence=False``.
            Also see :meth:`DataCut.pad` which pads a cut "to" a specified length.
            To "truncate" a cut, use :meth:`DataCut.truncate`.

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

    def resample(self, sampling_rate: int, affix_id: bool = False) -> "DataCut":
        """
        Return a new ``DataCut`` that will lazily resample the audio while reading it.
        This operation will drop the feature manifest, if attached.
        It does not affect the supervision.

        :param sampling_rate: The new sampling rate.
        :param affix_id: Should we modify the ID (useful if both versions of the same
            cut are going to be present in a single manifest).
        :return: a modified copy of the current ``DataCut``.
        """
        assert self.has_recording, "Cannot resample a DataCut without Recording."
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

    def perturb_speed(self, factor: float, affix_id: bool = True) -> "DataCut":
        """
        Return a new ``DataCut`` that will lazily perturb the speed while loading audio.
        The ``num_samples``, ``start`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.
        We are also updating the time markers of the underlying ``Recording`` and the supervisions.

        :param factor: The speed will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``MonoCut.id`` field
            by affixing it with "_sp{factor}".
        :return: a modified copy of the current ``DataCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb speed on a DataCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb speed on a DataCut that references pre-computed features. "
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

    def perturb_tempo(self, factor: float, affix_id: bool = True) -> "DataCut":
        """
        Return a new ``DataCut`` that will lazily perturb the tempo while loading audio.

        Compared to speed perturbation, tempo preserves pitch.
        The ``num_samples``, ``start`` and ``duration`` fields are updated to reflect the
        shrinking/extending effect of speed.
        We are also updating the time markers of the underlying ``Recording`` and the supervisions.

        :param factor: The tempo will be adjusted this many times (e.g. factor=1.1 means 1.1x faster).
        :param affix_id: When true, we will modify the ``MonoCut.id`` field
            by affixing it with "_tp{factor}".
        :return: a modified copy of the current ``DataCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb speed on a DataCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb tempo on a DataCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "speed perturbation."
            )
            self.features = None
        # Actual audio perturbation.
        recording_sp = self.recording.perturb_tempo(factor=factor, affix_id=affix_id)
        # Match the supervision's start and duration to the perturbed audio.
        # Since SupervisionSegment "start" is relative to the DataCut's, it's okay (and necessary)
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

    def perturb_volume(self, factor: float, affix_id: bool = True) -> "DataCut":
        """
        Return a new ``DataCut`` that will lazily perturb the volume while loading audio.

        :param factor: The volume will be adjusted this many times (e.g. factor=1.1 means 1.1x louder).
        :param affix_id: When true, we will modify the ``DataCut.id`` field
            by affixing it with "_vp{factor}".
        :return: a modified copy of the current ``DataCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot perturb volume on a DataCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to perturb volume on a DataCut that references pre-computed features. "
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

    def normalize_loudness(self, target: float, affix_id: bool = False) -> "DataCut":
        """
        Return a new ``DataCut`` that will lazily apply loudness normalization.

        :param target: The target loudness in dBFS.
        :param affix_id: When true, we will modify the ``DataCut.id`` field
            by affixing it with "_ln{target}".
        :return: a modified copy of the current ``DataCut``.
        """
        # Pre-conditions
        assert (
            self.has_recording
        ), "Cannot apply loudness normalization on a DataCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to normalize loudness on a DataCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "loudness normalization."
            )
            self.features = None

        # Add loudness normalization to the recording.
        recording_ln = self.recording.normalize_loudness(
            target=target, affix_id=affix_id
        )
        # Match the supervision's id (and it's underlying recording id).
        supervisions_ln = [
            fastcopy(
                s,
                id=f"{s.id}_ln{target}" if affix_id else s.id,
                recording_id=f"{s.recording_id}_ln{target}"
                if affix_id
                else s.recording_id,
            )
            for s in self.supervisions
        ]
        return fastcopy(
            self,
            id=f"{self.id}_ln{target}" if affix_id else self.id,
            recording=recording_ln,
            supervisions=supervisions_ln,
        )

    def dereverb_wpe(self, affix_id: bool = True) -> "DataCut":
        """
        Return a new ``DataCut`` that will lazily apply WPE dereverberation.

        :param affix_id: When true, we will modify the ``DataCut.id`` field
            by affixing it with "_wpe".
        :return: a modified copy of the current ``DataCut``.
        """
        # Pre-conditions
        assert self.has_recording, "Cannot apply WPE on a DataCut without Recording."
        if self.has_features:
            logging.warning(
                "Attempting to de-reverberate a DataCut that references pre-computed features. "
                "The feature manifest will be detached, as we do not support feature-domain "
                "de-reverberation."
            )
            self.features = None

        # Add WPE to the recording.
        recording_wpe = self.recording.dereverb_wpe(affix_id=affix_id)
        # Match the supervision's id (and it's underlying recording id).
        supervisions_wpe = [
            fastcopy(
                s,
                id=f"{s.id}_wpe" if affix_id else s.id,
                recording_id=f"{s.recording_id}_wpe" if affix_id else s.recording_id,
            )
            for s in self.supervisions
        ]
        return fastcopy(
            self,
            id=f"{self.id}_wpe" if affix_id else self.id,
            recording=recording_wpe,
            supervisions=supervisions_wpe,
        )

    @abstractmethod
    def reverb_rir(
        self,
        rir_recording: Optional["Recording"] = None,
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: List[int] = [0],
        room_rng_seed: Optional[int] = None,
        source_rng_seed: Optional[int] = None,
    ) -> "DataCut":
        ...

    def map_supervisions(
        self, transform_fn: Callable[[SupervisionSegment], SupervisionSegment]
    ) -> "DataCut":
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
    ) -> "DataCut":
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

    @abstractmethod
    def merge_supervisions(
        self,
        merge_policy: str = "delimiter",
        custom_merge_fn: Optional[Callable[[str, Iterable[Any]], Any]] = None,
        **kwargs,
    ) -> "DataCut":
        ...

    @staticmethod
    @abstractmethod
    def from_dict(data: dict) -> "DataCut":
        ...

    def with_features_path_prefix(self, path: Pathlike) -> "DataCut":
        if not self.has_features:
            return self
        return fastcopy(self, features=self.features.with_path_prefix(path))

    def with_recording_path_prefix(self, path: Pathlike) -> "DataCut":
        if not self.has_recording:
            return self
        return fastcopy(self, recording=self.recording.with_path_prefix(path))
