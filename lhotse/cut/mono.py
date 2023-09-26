import logging
import warnings
from dataclasses import dataclass
from functools import partial, reduce
from operator import add
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch

from lhotse.audio import Recording
from lhotse.cut.data import DataCut
from lhotse.features import Features
from lhotse.supervision import SupervisionSegment
from lhotse.utils import (
    add_durations,
    fastcopy,
    hash_str_to_int,
    merge_items_with_delimiter,
    overlaps,
    rich_exception_info,
    uuid4,
)


@dataclass
class MonoCut(DataCut):
    """
    :class:`~lhotse.cut.MonoCut` is a :class:`~lhotse.cut.Cut` of a single channel of
    a :class:`~lhotse.audio.Recording`. In addition to Cut, it has a specified channel attribute. This is the most commonly used type of cut.

    Please refer to the documentation of :class:`~lhotse.cut.Cut` to learn more about using cuts.

    See also:

        - :class:`lhotse.cut.Cut`
        - :class:`lhotse.cut.MixedCut`
        - :class:`lhotse.cut.CutSet`
    """

    channel: int

    @property
    def num_channels(self) -> int:
        # MonoCut is always single-channel.
        return 1

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

    @rich_exception_info
    def load_video(
        self,
        with_audio: bool = True,
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Load the subset of video (and audio) from attached recording.
        The data is trimmed to the [begin, end] range specified by the MonoCut.

        :param with_audio: bool, whether to load and return audio alongside video. True by default.
        :return: a tuple of video tensor and optionally audio tensor (or ``None``),
            or ``None`` if this cut has no video.
        """
        if self.has_video:
            return self.recording.load_video(
                channels=self.channel,
                offset=self.start,
                duration=self.duration,
                with_audio=with_audio,
            )
        return None

    def reverb_rir(
        self,
        rir_recording: Optional["Recording"] = None,
        normalize_output: bool = True,
        early_only: bool = False,
        affix_id: bool = True,
        rir_channels: List[int] = [0],
        room_rng_seed: Optional[int] = None,
        source_rng_seed: Optional[int] = None,
    ) -> DataCut:
        """
        Return a new ``DataCut`` that will convolve the audio with the provided impulse response.
        If the `rir_recording` is multi-channel, the `rir_channels` argument determines which channels
        will be used. By default, we use the first channel and return a MonoCut. If we reverberate
        with a multi-channel RIR, we return a MultiCut.

        If no ``rir_recording`` is provided, we will generate an impulse response using a fast random
        generator (https://arxiv.org/abs/2208.04101). Note that the generator only supports simulating
        reverberation with a single microphone, so we will return a MonoCut in this case.

        :param rir_recording: The impulse response to use for convolving.
        :param normalize_output: When true, output will be normalized to have energy as input.
        :param early_only: When true, only the early reflections (first 50 ms) will be used.
        :param affix_id: When true, we will modify the ``MonoCut.id`` field
            by affixing it with "_rvb".
        :param rir_channels: The channels of the impulse response to use. First channel is used by default.
            If multiple channels are specified, this will produce a MultiCut instead of a MonoCut.
        :param room_rng_seed: The seed for the room configuration.
        :param source_rng_seed: The seed for the source position.
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

        if rir_recording is None:
            # Set rir_channels to 0 since we can only generate a single-channel RIR.
            rir_channels = [0]

            if room_rng_seed is None:
                room_rng_seed = hash_str_to_int(str(uuid4()) + self.id)

            if source_rng_seed is None:
                source_rng_seed = room_rng_seed

        if len(rir_channels) == 1:
            # reverberation will return a MonoCut
            recording_rvb = self.recording.reverb_rir(
                rir_recording=rir_recording,
                normalize_output=normalize_output,
                early_only=early_only,
                affix_id=affix_id,
                rir_channels=rir_channels,
                room_rng_seed=room_rng_seed,
                source_rng_seed=source_rng_seed,
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
            from .multi import MultiCut

            channels = list(range(len(rir_channels)))
            # we will return a MultiCut where each channel represents the MonoCut convolved
            # with a single channel of the RIR
            recording_rvb = self.recording.reverb_rir(
                rir_recording=rir_recording,
                normalize_output=normalize_output,
                early_only=early_only,
                affix_id=affix_id,
                rir_channels=rir_channels,
                room_rng_seed=room_rng_seed,
                source_rng_seed=source_rng_seed,
            )
            # Match the supervision's id (and it's underlying recording id).
            supervisions_rvb = [
                s.reverb_rir(
                    affix_id=affix_id,
                    channel=channels,
                )
                for s in self.supervisions
            ]

            return fastcopy(
                MultiCut.from_mono(self),
                recording=recording_rvb,
                supervisions=supervisions_rvb,
                channel=channels,
            )

    def merge_supervisions(
        self,
        merge_policy: str = "delimiter",
        custom_merge_fn: Optional[Callable[[str, Iterable[Any]], Any]] = None,
    ) -> "MonoCut":
        """
        Return a copy of the cut that has all of its supervisions merged into
        a single segment.

        The new start is the start of the earliest superivion, and the new duration
        is a minimum spanning duration for all the supervisions. The text fields of
        all segments are concatenated with a whitespace.

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
            recording_id=sups[0].recording_id,
            start=mstart,
            duration=mduration,
            channel=sups[0].channel,
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

        return fastcopy(self, supervisions=[msup])

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
