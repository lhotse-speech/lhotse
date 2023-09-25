import logging
import warnings
from dataclasses import dataclass
from functools import partial, reduce
from itertools import groupby
from operator import add
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

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
    is_equal_or_contains,
    merge_items_with_delimiter,
    overlaps,
    rich_exception_info,
    to_list,
    uuid4,
)


@dataclass
class MultiCut(DataCut):
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

    channel: List[int]

    @property
    def num_channels(self) -> int:
        return len(to_list(self.channel))

    @rich_exception_info
    def load_features(
        self, channel: Optional[Union[int, List[int]]] = None
    ) -> Optional[np.ndarray]:
        """
        Load the features from the underlying storage and cut them to the relevant
        [begin, duration] region of the current MultiCut.

        :param channel: The channel to load the features for. If None, all channels will be loaded.
            This is useful for the case when we have features extracted for each channel of
            the multi-cut, and we want to selectively load them.
        """
        if self.has_features:
            feats = self.features.load(
                start=self.start,
                duration=self.duration,
                channel_id=self.channel if channel is None else channel,
            )
            # Note: we forgive off-by-one errors in the feature matrix frames
            #       due to various hard-to-predict floating point arithmetic issues.
            #       If needed, we will remove or duplicate the last frame to be
            #       consistent with the manifests declared "num_frames".
            if feats.shape[0] - self.num_frames == 1:
                feats = feats[: self.num_frames, ...]
            elif feats.shape[0] - self.num_frames == -1:
                feats = np.concatenate((feats, feats[-1:, ...]), axis=0)
            return feats
        return None

    @rich_exception_info
    def load_audio(
        self, channel: Optional[Union[int, List[int]]] = None
    ) -> Optional[np.ndarray]:
        """
        Load the audio by locating the appropriate recording in the supplied Recording.
        The audio is trimmed to the [begin, end] range specified by the MultiCut.

        :param channel: optional int or list of int, the subset of channels to load (all by default).
        :return: a numpy ndarray with audio samples, with shape (C <channel>, N <samples>)
        """
        if self.has_recording:
            return self.recording.load_audio(
                channels=self.channel if channel is None else channel,
                offset=self.start,
                duration=self.duration,
            )
        return None

    @rich_exception_info
    def load_video(
        self,
        channel: Optional[Union[int, List[int]]] = None,
        with_audio: bool = True,
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Load the subset of video (and audio) from attached recording.
        The data is trimmed to the [begin, end] range specified by the MonoCut.

        :param channel: optional int or list of int, the subset of channels to load (all by default).
        :param with_audio: bool, whether to load and return audio alongside video. True by default.
        :return: a tuple of video tensor and optionally audio tensor (or ``None``),
            or ``None`` if this cut has no video.
        """
        if self.has_video:
            return self.recording.load_video(
                channels=self.channel if channel is None else channel,
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
        :param room_rng_seed: The seed for the room configuration.
        :param source_rng_seed: The seed for the source positions.
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
            if room_rng_seed is None:
                room_rng_seed = hash_str_to_int(str(uuid4()) + self.id)
            if source_rng_seed is None:
                source_rng_seed = room_rng_seed
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

    def merge_supervisions(
        self,
        merge_policy: str = "delimiter",
        merge_channels: bool = True,
        custom_merge_fn: Optional[Callable[[str, Iterable[Any]], Any]] = None,
    ) -> "MultiCut":
        """
        Return a copy of the cut that has all of its supervisions merged into
        a single segment. The ``channel`` attribute of all the segments in this case
        will be set to the union of all channels. If ``merge_channels`` is set to ``False``,
        the supervisions will be merged into a single segment per channel group. The
        ``channel`` attribute will not change in this case.

        The new start is the start of the earliest superivion, and the new duration
        is a minimum spanning duration for all the supervisions. The text fields of
        all segments are concatenated with a whitespace.

        :param merge_policy: one of "keep_first" or "delimiter". If "keep_first", we
            keep only the first segment's field value, otherwise all string fields
            (including IDs) are prefixed with "cat#" and concatenated with a hash symbol "#".
            This is also applied to ``custom`` fields. Fields with a ``None`` value are omitted.
        :param merge_channels: If true, we will merge all supervisions into a single segment.
            If false, we will merge supervisions per channel group. Default: True.
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

        if merge_channels:
            # Merge all supervisions into a single segment.
            all_channels = set()
            for s in sups:
                c = set(to_list(s.channel))
                all_channels.update(c)
            all_channels = sorted(all_channels)
            sups_by_channel = {tuple(all_channels): sups}  # `set` is not hashable
        else:
            # Merge supervisions per channel group.
            sups_by_channel = {
                tuple(c): list(csups)
                for c, csups in groupby(
                    sorted(sups, key=lambda s: s.channel), key=lambda s: s.channel
                )
            }

        msups = []
        text_overlap_warning = False
        for channel, csups in sups_by_channel.items():
            mstart = csups[0].start
            mend = csups[-1].end
            mduration = add_durations(mend, -mstart, sampling_rate=self.sampling_rate)

            custom_keys = set(
                k for s in csups if s.custom is not None for k in s.custom.keys()
            )
            alignment_keys = set(
                k for s in csups if s.alignment is not None for k in s.alignment.keys()
            )

            if (
                any(overlaps(s1, s2) for s1, s2 in zip(csups, csups[1:]))
                and any(s.text is not None for s in csups)
                and not text_overlap_warning
            ):
                warnings.warn(
                    "You are merging overlapping supervisions that have text transcripts. "
                    "The result is likely to be unusable if you are going to train speech "
                    f"recognition models (cut id: {self.id})."
                )
                text_overlap_warning = True

            msups.append(
                SupervisionSegment(
                    id=merge_func_(s.id for s in csups),
                    recording_id=csups[0].recording_id,
                    start=mstart,
                    duration=mduration,
                    channel=list(channel),
                    text=" ".join(s.text for s in csups if s.text),
                    speaker=merge_func_(s.speaker for s in csups if s.speaker),
                    language=merge_func_(s.language for s in csups if s.language),
                    gender=merge_func_(s.gender for s in csups if s.gender),
                    custom={
                        k: merge_custom(
                            k,
                            (
                                s.custom[k]
                                for s in csups
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
                                for s in csups
                                if s.alignment is not None and k in s.alignment
                            ),
                        )
                        for k in alignment_keys
                    },
                )
            )

        return fastcopy(self, supervisions=msups)

    @staticmethod
    def from_mono(*cuts: DataCut) -> "MultiCut":
        """
        Convert one or more MonoCut to a MultiCut. If multiple mono cuts are provided, they
        must match in all fields except the channel. Each cut must have a distinct channel.

        :param cuts: the input cut(s).
        :return: a MultiCut with a single track.
        """
        from .mono import MonoCut

        assert all(isinstance(c, MonoCut) for c in cuts), "All cuts must be MonoCuts"
        assert (
            sum(
                1 for _ in groupby(cuts, key=lambda c: (c.recording_id, c.start, c.end))
            )
            == 1
        ), "Cuts must match in all fields except channel"
        assert len(set(c.channel for c in cuts)) == len(
            cuts
        ), "All cuts must have a distinct channel"

        data = cuts[0].to_dict()
        data.pop("type")

        return MultiCut(
            **{
                **data,
                "channel": sorted([c.channel for c in cuts]),
                "supervisions": [s for c in cuts for s in c.supervisions],
            }
        )

    def to_mono(self, mono_downmix: bool = False) -> Union["DataCut", List["DataCut"]]:
        """
        Convert a MultiCut to either a list of MonoCuts (one per channel) or a single
        MonoCut obtained by downmixing all channels.

        :param mono_downmix: If true, we will downmix all channels into a single MonoCut.
            If false, we will return a list of MonoCuts, one per channel.
        :return: a list of MonoCuts or a single MonoCut.
        """
        from .mixed import MixedCut, MixTrack
        from .mono import MonoCut

        mono_cuts = [
            MonoCut(
                id=f"{self.id}-{channel}",
                recording=self.recording,
                start=self.start,
                duration=self.duration,
                channel=channel,
                supervisions=[
                    fastcopy(s, channel=channel)
                    for s in self.supervisions
                    if is_equal_or_contains(s.channel, channel)
                ],
                custom=self.custom,
            )
            for channel in to_list(self.channel)
        ]
        if not mono_downmix:
            return mono_cuts

        # Downmix the mono cuts into a single MixedCut.
        mixed_cut = MixedCut(
            id=self.id,
            tracks=[
                MixTrack(cut=mono_cut, offset=0.0, snr=None) for mono_cut in mono_cuts
            ],
        )
        return mixed_cut.to_mono()

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
