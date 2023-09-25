import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from lhotse.audio import Recording
from lhotse.audio.utils import VideoInfo
from lhotse.cut.base import Cut
from lhotse.features import FeatureExtractor
from lhotse.supervision import SupervisionSegment
from lhotse.utils import (
    LOG_EPSILON,
    Pathlike,
    Seconds,
    compute_num_frames,
    compute_num_samples,
    fastcopy,
    perturb_num_samples,
    uuid4,
)


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
    video: Optional[VideoInfo] = None

    # Dict for storing padding values for custom array attributes
    custom: Optional[dict] = None

    @property
    def start(self) -> Seconds:
        return 0

    @property
    def supervisions(self):
        return []

    @property
    def channel(self) -> int:
        return 0

    @property
    def has_features(self) -> bool:
        return self.num_frames is not None

    @property
    def has_recording(self) -> bool:
        return self.num_samples is not None

    @property
    def has_video(self) -> bool:
        return self.has_recording and self.video is not None

    @property
    def num_channels(self) -> int:
        return 1

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

    def load_video(
        self,
        with_audio: bool = True,
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        if self.has_video:
            audio = None
            if with_audio:
                audio = torch.zeros(
                    1,
                    compute_num_samples(self.duration, self.sampling_rate),
                    dtype=torch.float32,
                )
            return (
                torch.zeros(
                    self.video.num_frames,
                    3,
                    self.video.height,
                    self.video.width,
                    dtype=torch.uint8,
                ),
                audio,
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
        pad_silence: bool = True,
    ) -> "PaddingCut":
        """
        Return a new PaddingCut with region extended by the specified duration.

        :param duration: The duration by which to extend the cut.
        :param direction: string, 'left', 'right' or 'both'. Determines whether the cut should
            be extended to the left, right or both sides. By default, the cut is extended by
            the specified duration on both sides.
        :param preserve_id: When ``True``, preserves the cut ID from before padding.
            Otherwise, generates a new random ID (default).
        :param pad_silence: See usage in :func:`lhotse.cut.MonoCut.extend_by`. It is ignored here.
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
        rir_recording: Optional["Recording"] = None,
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: List[int] = [0],
        room_rng_seed: Optional[int] = None,
        source_rng_seed: Optional[int] = None,
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

    def drop_alignments(self) -> "PaddingCut":
        """Return a copy of the current :class:`.PaddingCut`, detached from ``alignments``."""
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

    def move_to_memory(self, *args, **kwargs) -> "PaddingCut":
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
