from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from intervaltree import Interval, IntervalTree
from typing_extensions import Literal

from lhotse.audio import AudioSource, Recording, VideoInfo
from lhotse.augmentation import AugmentFn
from lhotse.features import FeatureExtractor
from lhotse.supervision import SupervisionSegment
from lhotse.utils import (
    Decibels,
    Pathlike,
    Seconds,
    SetContainingAnything,
    add_durations,
    asdict_nonull,
    compute_num_samples,
    compute_num_windows,
    compute_start_duration_for_extended_cut,
    deprecated,
    fastcopy,
    ifnone,
    is_torchaudio_available,
    overlaps,
    to_hashable,
)

# One of the design principles for Cuts is a maximally "lazy" implementation, e.g. when mixing Cuts,
# we'd rather sum the feature matrices only after somebody actually calls "load_features". It helps to avoid
# an excessive storage size for data augmented in various ways.


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

        >>> from lhotse import LilcomChunkyWriter
        >>> with LilcomChunkyWriter('feats.lca') as storage:
        ...     cut_with_feats = cut.compute_and_store_features(
        ...         extractor=Fbank(),
        ...         storage=storage
        ...     )

    Cuts have several methods that allow their manipulation, transformation, and mixing.
    Some examples (see the respective methods documentation for details)::

        >>> cut_2_to_4s = cut.truncate(offset=2, duration=2)
        >>> cut_padded = cut.pad(duration=10.0)
        >>> cut_extended = cut.extend_by(duration=5.0, direction='both')
        >>> cut_mixed = cut.mix(other_cut, offset_other_by=5.0, snr=20)
        >>> cut_append = cut.append(other_cut)
        >>> cut_24k = cut.resample(24000)
        >>> cut_sp = cut.perturb_speed(1.1)
        >>> cut_vp = cut.perturb_volume(2.)
        >>> cut_rvb = cut.reverb_rir(rir_recording)

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
    has_video: bool
    video: Optional[VideoInfo]

    # The following is the list of methods implemented by the child classes.
    # They are not abstract methods because dataclasses do not work well with the "abc" module.
    # Check a specific child class for their documentation.
    from_dict: Callable[[Dict], "Cut"]
    load_audio: Callable[[], np.ndarray]
    load_video: Callable[[], Tuple[torch.Tensor, Optional[torch.Tensor]]]
    load_features: Callable[[], np.ndarray]
    compute_and_store_features: Callable
    drop_features: Callable
    drop_recording: Callable
    drop_supervisions: Callable
    truncate: Callable
    pad: Callable
    extend_by: Callable
    resample: Callable
    perturb_speed: Callable
    perturb_tempo: Callable
    perturb_volume: Callable
    reverb_rir: Callable
    map_supervisions: Callable
    merge_supervisions: Callable
    filter_supervisions: Callable
    fill_supervision: Callable
    with_features_path_prefix: Callable
    with_recording_path_prefix: Callable

    @property
    def end(self) -> Seconds:
        return add_durations(
            self.start, self.duration, sampling_rate=self.sampling_rate
        )

    def to_dict(self) -> dict:
        d = asdict_nonull(self)
        return {**d, "type": type(self).__name__}

    @property
    def has_overlapping_supervisions(self) -> bool:
        if len(self.supervisions) < 2:
            return False

        from cytoolz import sliding_window

        for left, right in sliding_window(
            2, sorted(self.supervisions, key=lambda s: s.start)
        ):
            if overlaps(left, right):
                return True
        return False

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

    def split(self, timestamp: Seconds) -> Tuple["Cut", "Cut"]:
        """
        Split a cut into two cuts at ``timestamp``, which is measured from the start of the cut.
        For example, a [0s - 10s] cut split at 4s yields:
            - left cut [0s - 4s]
            - right cut [4s - 10s]
        """
        assert 0 < timestamp < self.duration, f"0 < {timestamp} < {self.duration}"
        left = self.truncate(duration=timestamp)
        right = self.truncate(offset=timestamp)
        return left, right

    def mix(
        self,
        other: "Cut",
        offset_other_by: Seconds = 0.0,
        allow_padding: bool = False,
        snr: Optional[Decibels] = None,
        preserve_id: Optional[str] = None,
    ) -> "Cut":
        """Refer to :function:`~lhotse.cut.mix` documentation."""
        from .set import mix

        return mix(
            self,
            other,
            offset=offset_other_by,
            allow_padding=allow_padding,
            snr=snr,
            preserve_id=preserve_id,
        )

    def append(
        self,
        other: "Cut",
        snr: Optional[Decibels] = None,
        preserve_id: Optional[str] = None,
    ) -> "Cut":
        """
        Append the ``other`` Cut after the current Cut. Conceptually the same as ``mix`` but with an offset
        matching the current cuts length. Optionally scale down (positive SNR) or scale up (negative SNR)
        the ``other`` cut.
        Returns a MixedCut, which only keeps the information about the mix; actual mixing is performed
        during the call to ``load_features``.

        :param preserve_id: optional string ("left", "right"). When specified, append will preserve the cut ID
            of the left- or right-hand side argument. Otherwise, a new random ID is generated.
        """
        from .set import mix

        return mix(self, other, offset=self.duration, snr=snr, preserve_id=preserve_id)

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
            ax.axvspan(supervision.start, supervision.end, color="green", alpha=0.1)
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

    def plot_alignment(self, alignment_type: str = "word"):
        """
        Display the alignment on top of a spectrogram. Requires matplotlib to be installed.
        """
        import matplotlib.pyplot as plt

        from lhotse import Fbank
        from lhotse.utils import compute_num_frames

        assert (
            len(self.supervisions) == 1
        ), "Cannot plot alignment: there has to be exactly one supervision in a Cut."
        sup = self.supervisions[0]
        assert (
            sup.alignment is not None and alignment_type in sup.alignment
        ), f"Cannot plot alignment: missing alignment field or alignment type '{alignment_type}'"

        fbank = Fbank()
        sampling_rate = fbank.extractor.sampling_rate

        feats = self.resample(sampling_rate).compute_features(fbank)
        speaker = sup.speaker or "<unknown>"
        language = sup.language or "<unknown>"

        fig = plt.matshow(np.flip(feats.transpose(1, 0), 0))
        plt.title(
            "Cut ID:" + self.id + ", Speaker:" + speaker + ", Language:" + language
        )
        plt.tick_params(
            axis="both",
            which="major",
            labelbottom=True,
            labeltop=False,
            bottom=True,
            top=False,
        )

        for idx, item in enumerate(sup.alignment[alignment_type]):
            is_even = bool(idx % 2)
            end_frame = compute_num_frames(
                item.end,
                frame_shift=fbank.frame_shift,
                sampling_rate=sampling_rate,
            )
            plt.text(
                end_frame - 4,
                70 if is_even else 45,
                item.symbol,
                fontsize=12,
                color="w",
                rotation="vertical",
            )
            plt.axvline(end_frame, color="k")

        plt.show()

    def trim_to_supervisions(
        self,
        keep_overlapping: bool = True,
        min_duration: Optional[Seconds] = None,
        context_direction: Literal["center", "left", "right", "random"] = "center",
        keep_all_channels: bool = False,
    ) -> "CutSet":  # noqa: F821
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

        For the case of a multi-channel cut with multiple supervisions, we can either trim
        while respecting the supervision channels (in which case output cut has the same channels
        as the supervision) or ignore the channels (in which case output cut has the same channels
        as the input cut).

        .. hint:: If the resulting trimmed cut contains a single supervision, we set the cut id to
            the ``id`` of this supervision, for better compatibility with downstream tools, e.g.
            comparing the hypothesis of ASR with the reference in icefall.

        .. hint:: If a MultiCut is trimmed and the resulting trimmed cut contains a single channel,
            we convert it to a MonoCut.

        :param keep_overlapping: when ``False``, it will discard parts of other supervisions that overlap with the
            main supervision. In the illustration above, it would discard ``Sup2`` in ``Cut1`` and ``Sup1`` in ``Cut2``.
            In this mode, we guarantee that there will always be exactly one supervision per cut.
        :param min_duration: An optional duration in seconds; specifying this argument will extend the cuts
            that would have been shorter than ``min_duration`` with actual acoustic context in the recording/features.
            If there are supervisions present in the context, they are kept when ``keep_overlapping`` is true.
            If there is not enough context, the returned cut will be shorter than ``min_duration``.
            If the supervision segment is longer than ``min_duration``, the return cut will be longer.
        :param context_direction: Which direction should the cut be expanded towards to include context.
            The value of "center" implies equal expansion to left and right;
            random uniformly samples a value between "left" and "right".
        :param keep_all_channels: If ``True``, the output cut will have the same channels as the input cut. By default,
            the trimmed cut will have the same channels as the supervision.
        :return: a list of cuts.
        """
        from .mixed import MixedCut
        from .multi import MultiCut
        from .set import CutSet

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
                    direction=context_direction,
                )
            trimmed = self.truncate(
                offset=new_start,
                duration=new_duration,
                keep_excessive_supervisions=keep_overlapping,
                _supervisions_index=supervisions_index,
            )

            if not keep_overlapping:
                # Ensure that there is exactly one supervision per cut.
                trimmed = trimmed.filter_supervisions(lambda s: s.id == segment.id)

            if not keep_all_channels and not isinstance(trimmed, MixedCut):
                # For MixedCut, we can't change the channels since it is defined by the
                # number of channels in underlying tracks.

                # Ensure that all supervisions have the same channel.
                assert (
                    len(set(to_hashable(s.channel) for s in trimmed.supervisions)) == 1
                ), (
                    "Trimmed cut has supervisions with different channels. Either set "
                    "`keep_all_channels=True` to keep original channels or `keep_overlapping=False` "
                    "to retain only 1 supervision per trimmed cut."
                )
                trimmed.channel = trimmed.supervisions[0].channel

                # If we have a single-channel MultiCut, we will convert it into a MonoCut.
                if isinstance(trimmed, MultiCut) and trimmed.num_channels == 1:
                    trimmed = trimmed.to_mono()[0]

            if len(trimmed.supervisions) == 1:
                # If the trimmed cut contains a single supervision, we set the cut id to
                # the id of this supervision.
                trimmed.id = segment.id
            cuts.append(trimmed)
        return CutSet.from_cuts(cuts)

    def trim_to_alignments(
        self,
        type: str,
        max_pause: Optional[Seconds] = None,
        max_segment_duration: Optional[Seconds] = None,
        delimiter: str = " ",
        keep_all_channels: bool = False,
    ) -> "CutSet":  # noqa: F821
        """
        Splits the current :class:`.Cut` into its constituent alignment items (:class:`.AlignmentItem`).
        These cuts have identical start times and durations as the alignment item. Additionally,
        the `max_pause` option can be used to merge alignment items that are separated by a pause
        shorter than `max_pause`. If `max_segment_duration` is specified, we will keep merging
        consecutive segments until the duration of the merged segment exceeds `max_segment_duration`.

        For the case of a multi-channel cut with multiple alignments, we can either trim
        while respecting the supervision channels (in which case output cut has the same channels
        as the supervision) or ignore the channels (in which case output cut has the same channels
        as the input cut).

        .. hint:: If the resulting trimmed cut contains a single supervision, we set the cut id to
            the ``id`` of this supervision, for better compatibility with downstream tools, e.g.
            comparing the hypothesis of ASR with the reference in icefall.

        .. hint:: If a MultiCut is trimmed and the resulting trimmed cut contains a single channel,
            we convert it to a MonoCut.

        .. hint:: If you have a Cut with multiple supervision segments and you want to trim it to
            the word-level alignment, you can use the :meth:`.Cut.merge_supervisions` method
            first to merge the supervisions into a single one, followed by the
            :meth:`.Cut.trim_to_alignments` method. For example::

                >>> cut = cut.merge_supervisions(type='word', delimiter=' ')
                >>> cut = cut.trim_to_alignments(type='word', max_pause=1.0)

        .. hint:: The above technique can also be used to segment long cuts into roughly equal
            duration segments, while respecting alignment boundaries. For example, to split a
            Cut into 10s segments, you can do::

                >>> cut = cut.merge_supervisions(type='word', delimiter=' ')
                >>> cut = cut.trim_to_alignments(type='word', max_pause=10.0, max_segment_duration=10.0)

        :param type: The type of the alignment to trim to (e.g. "word").
        :param max_pause: The maximum pause allowed between the alignments to merge them. If ``None``,
            no merging will be performed. [default: None]
        :param delimiter: The delimiter to use when joining the alignment items.
        :param keep_all_channels: If ``True``, the output cut will have the same channels as the input cut. By default,
            the trimmed cut will have the same channels as the supervision.
        :param num_jobs: Number of parallel workers to process the cuts.
        :return: a CutSet object.
        """
        from lhotse.supervision import AlignmentItem

        if max_pause is None:
            # Set to a negative value so that no merging is performed.
            max_pause = -1.0

        if max_segment_duration is None:
            # Set to the cut duration so that resulting segments are always smaller.
            max_segment_duration = self.duration

        # For the implementation, we first create new supervisions for the cut, and then
        # use the `trim_to_supervisions` method to do the actual trimming.
        new_supervisions = []
        for segment in self.supervisions:
            if (
                segment.alignment is None
                or type not in segment.alignment
                or not segment.alignment[type]
            ):
                continue
            alignments = sorted(segment.alignment[type], key=lambda a: a.start)

            # Merge the alignments if needed. We also keep track of the indices of the
            # merged alignments in the original list. This is needed to create the
            # `alignment` field in the new supervisions.
            # NOTE: We use the `AlignmentItem` class here for convenience --- the merged
            # alignments are not actual alignment items, but rather just a way to keep
            # track of merged segments.
            merged_alignments = [(alignments[0], [0])]
            for i, item in enumerate(alignments[1:]):
                # If alignment item is blank, skip it. Sometimes, blank alignment items
                # are used to denote pauses in the utterance.
                if item.symbol.strip() == "":
                    continue
                prev_item, prev_indices = merged_alignments[-1]
                if (
                    item.start - prev_item.end <= max_pause
                    and item.end - prev_item.start <= max_segment_duration
                ):
                    new_item = AlignmentItem(
                        symbol=delimiter.join([prev_item.symbol, item.symbol]),
                        start=prev_item.start,
                        duration=item.end - prev_item.start,
                    )
                    merged_alignments[-1] = (new_item, prev_indices + [i + 1])
                else:
                    merged_alignments.append((item, [i + 1]))

            # Create new supervisions for the merged alignments.
            for i, (item, indices) in enumerate(merged_alignments):
                new_supervisions.append(
                    SupervisionSegment(
                        id=f"{segment.id}-{i}",
                        recording_id=segment.recording_id,
                        start=item.start - self.start,  # relative to the cut
                        duration=item.duration,
                        channel=segment.channel,
                        text=item.symbol,
                        language=segment.language,
                        speaker=segment.speaker,
                        gender=segment.gender,
                        alignment={type: [alignments[j] for j in indices]},
                    )
                )

        # Create a copy of this CutSet so that original supervisions do not get modified.
        new_cuts = fastcopy(self, supervisions=new_supervisions)
        return new_cuts.trim_to_supervisions(
            keep_overlapping=False,
            keep_all_channels=keep_all_channels,
        )

    def trim_to_supervision_groups(
        self,
        max_pause: Seconds = 0.0,
    ) -> "CutSet":  # noqa: F821
        """
        Return a new CutSet with Cuts based on supervision groups. A supervision group is
        a set of supervisions with no gaps between them (or gaps shorter than ``max_pause``).
        This is similar to the concept of an `utterance group` as described in this paper:
        https://arxiv.org/abs/2211.00482

        For example, the following cut::

                                                Cut
        ╔═════════════════════════════════════════════════════════════════════════════════╗
        ║┌──────────────────────┐                              ┌────────┐                 ║
        ║│ Hello this is John.  │                              │   Hi   │                 ║
        ║└──────────────────────┘                              └────────┘                 ║
        ║            ┌──────────────────────────────────┐            ┌───────────────────┐║
        ║            │     Hey, John. How are you?      │            │  What do you do?  │║
        ║            └──────────────────────────────────┘            └───────────────────┘║
        ╚═════════════════════════════════════════════════════════════════════════════════╝

        is transformed into two cuts::

                            Cut 1                                       Cut 2
        ╔════════════════════════════════════════════════╗    ╔═══════════════════════════╗
        ║┌──────────────────────┐                        ║    ║┌────────┐                 ║
        ║│ Hello this is John.  │                        ║    ║│   Hi   │                 ║
        ║└──────────────────────┘                        ║    ║└────────┘                 ║
        ║            ┌──────────────────────────────────┐║    ║      ┌───────────────────┐║
        ║            │     Hey, John. How are you?      │║    ║      │  What do you do?  │║
        ║            └──────────────────────────────────┘║    ║      └───────────────────┘║
        ╚════════════════════════════════════════════════╝    ╚═══════════════════════════╝

        For the case of a multi-channel cut with multiple supervisions, we keep all the channels
        in the recording.

        :param max_pause: An optional duration in seconds; if the gap between two supervisions
            is longer than this, they will be treated as separate groups. By default, this is
            set to 0.0, which means that no gaps are allowed between supervisions.
        :return: a ``CutSet``.
        """
        from .set import CutSet

        if not self.supervisions:
            return self
        supervisions = sorted(self.supervisions, key=lambda s: s.start)
        supervision_group = [supervisions[0]]
        cur_end = supervisions[0].end
        new_cuts = []
        group_idx = 0
        for sup in supervisions[1:]:
            if sup.start - cur_end <= max_pause:
                supervision_group.append(sup)
                cur_end = max(cur_end, sup.end)
            else:
                offset = supervision_group[0].start
                duration = add_durations(
                    cur_end, -offset, sampling_rate=self.sampling_rate
                )
                new_cuts.append(
                    self.truncate(
                        offset=offset,
                        duration=duration,
                        keep_excessive_supervisions=False,
                    ).with_id(f"{self.id}-{max_pause}-{group_idx}")
                )
                group_idx += 1
                supervision_group = [sup]
                cur_end = sup.end

        # Add the last group.
        if len(supervision_group) > 0:
            offset = supervision_group[0].start
            duration = add_durations(cur_end, -offset, sampling_rate=self.sampling_rate)
            new_cuts.append(
                self.truncate(
                    offset=offset,
                    duration=duration,
                    keep_excessive_supervisions=False,
                ).with_id(f"{self.id}-{max_pause}-{group_idx}")
            )
        # The total number of supervisions should be the same.
        assert sum(len(c.supervisions) for c in new_cuts) == len(self.supervisions), (
            "The total number of supervisions decreased after trimming to supervision groups.\n"
            f"{sum(len(c.supervisions) for c in new_cuts)} != {len(self.supervisions)}\n"
            "This is likely a bug. Please report it here: https://github.com/lhotse-speech/lhotse/issues/new, "
            "and provide the following information:\n"
            f"original cut: {self},\n"
            f"new cuts: {new_cuts}"
        )
        return CutSet.from_cuts(new_cuts)

    def cut_into_windows(
        self,
        duration: Seconds,
        hop: Optional[Seconds] = None,
        keep_excessive_supervisions: bool = True,
    ) -> "CutSet":  # noqa: F821
        """
        Return a list of shorter cuts, made by traversing this cut in windows of
        ``duration`` seconds by ``hop`` seconds.

        The last window might have a shorter duration if there was not enough audio,
        so you might want to use either filter or pad the results.

        :param duration: Desired duration of the new cuts in seconds.
        :param hop: Shift between the windows in the new cuts in seconds.
        :param keep_excessive_supervisions: bool. When a cut is truncated in the
            middle of a supervision segment, should the supervision be kept.
        :return: a list of cuts made from shorter duration windows.
        """
        from .set import CutSet

        if not hop:
            hop = duration

        if self.has_video:
            assert (duration * self.video.fps).is_integer(), (
                f"[cut.id={self.id}] Window duration must be defined to result in an integer number of video frames "
                f"(duration={duration} * fps={self.video.fps} = {duration * self.video.fps})."
            )
            assert (hop * self.video.fps).is_integer(), (
                "[cut.id={self.id}] Window hop must be defined to result in an integer number of video frames "
                f"(hop={hop} * fps={self.video.fps} = {hop* self.video.fps})."
            )

        new_cuts = []
        n_windows = compute_num_windows(self.duration, duration, hop)

        # Operation without an Interval Tree is O(nm), where `n` is the number
        # of resulting windows, and `m` is the number of supervisions in the cut.
        # With an Interval Tree, we can do it in O(nlog(m) + m)
        supervisions_index = self.index_supervisions(index_mixed_tracks=True)

        for i in range(n_windows):
            new_cuts.append(
                self.truncate(
                    offset=hop * i,
                    duration=duration,
                    keep_excessive_supervisions=keep_excessive_supervisions,
                    _supervisions_index=supervisions_index,
                ).with_id(f"{self.id}-{i}")
            )
        return CutSet.from_cuts(new_cuts)

    def index_supervisions(
        self, index_mixed_tracks: bool = False, keep_ids: Optional[Set[str]] = None
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
        from .mixed import MixedCut

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

    def save_audio(
        self,
        storage_path: Pathlike,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        augment_fn: Optional[AugmentFn] = None,
        **kwargs,
    ) -> "Cut":
        """
        Store this cut's waveform as audio recording to disk.

        :param storage_path: The path to location where we will store the audio recordings.
        :param encoding: Audio encoding argument supported by ``torchaudio.save``. See
            https://pytorch.org/audio/stable/backend.html#save (sox_io backend) and
            https://pytorch.org/audio/stable/backend.html#id3 (soundfile backend) for more details.
        :param bits_per_sample: Audio bits_per_sample argument supported by ``torchaudio.save``. See
            https://pytorch.org/audio/stable/backend.html#save (sox_io backend) and
            https://pytorch.org/audio/stable/backend.html#id3 (soundfile backend) for more details.
        :param augment_fn: an optional callable used for audio augmentation.
            Be careful with the types of augmentations used: if they modify
            the start/end/duration times of the cut and its supervisions,
            you will end up with incorrect supervision information when using this API.
            E.g. for speed perturbation, use ``CutSet.perturb_speed()`` instead.
        :param kwargs: additional arguments passed to ``Cut.load_audio()``. Example, if
            saving a MixedCut, we can specify `mono_downmix=True` to downmix the tracks
            to mono before saving.
        :return: a new Cut instance.
        """

        storage_path = Path(storage_path)
        samples = self.load_audio(**kwargs)
        # TODO(@desh2608): Here, `samples` may be a list of arrays (for example, the case
        # when cut is a MixedCut and we specify `mixed=False`). For such a case, we should
        # save all tracks separately.
        if augment_fn is not None:
            samples = augment_fn(samples, self.sampling_rate)

        if is_torchaudio_available():
            import torchaudio

            torchaudio.save(
                str(storage_path),
                torch.as_tensor(samples),
                sample_rate=self.sampling_rate,
                encoding=encoding,
                bits_per_sample=bits_per_sample,
            )
        else:
            import soundfile as sf

            sf.write(str(storage_path), samples, samplerate=self.sampling_rate)
        recording = Recording(
            id=storage_path.stem,
            sampling_rate=self.sampling_rate,
            num_samples=samples.shape[1],
            duration=samples.shape[1] / self.sampling_rate,
            sources=[
                AudioSource(
                    type="file",
                    channels=list(range(self.num_channels)),
                    source=str(storage_path),
                )
            ],
        )
        return fastcopy(
            recording.to_cut(),
            id=self.id,
            supervisions=self.supervisions,
            custom=self.custom if hasattr(self, "custom") else None,
            features=self.features if self.has_features else None,
        )

    def speakers_feature_mask(
        self,
        min_speaker_dim: Optional[int] = None,
        speaker_to_idx_map: Optional[Dict[str, int]] = None,
        use_alignment_if_exists: Optional[str] = None,
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
        assert self.has_features, (
            f"No features available. "
            f"Can't compute supervisions feature mask for cut with ID: {self.id}."
        )
        if speaker_to_idx_map is None:
            speaker_to_idx_map = {
                spk: idx
                for idx, spk in enumerate(
                    sorted(set(s.speaker for s in self.supervisions))
                )
            }
        num_speakers = len(speaker_to_idx_map)
        if min_speaker_dim is not None:
            num_speakers = min(min_speaker_dim, num_speakers)
        mask = np.zeros((num_speakers, self.num_frames))
        for supervision in self.supervisions:
            speaker_idx = speaker_to_idx_map[supervision.speaker]
            if (
                use_alignment_if_exists
                and supervision.alignment
                and use_alignment_if_exists in supervision.alignment
            ):
                for ali in supervision.alignment[use_alignment_if_exists]:
                    st = round(ali.start / self.frame_shift) if ali.start > 0 else 0
                    et = (
                        round(ali.end / self.frame_shift)
                        if ali.end < self.duration
                        else self.num_frames
                    )
                    mask[speaker_idx, st:et] = 1
            else:
                st = (
                    round(supervision.start / self.frame_shift)
                    if supervision.start > 0
                    else 0
                )
                et = (
                    round(supervision.end / self.frame_shift)
                    if supervision.end < self.duration
                    else self.num_frames
                )
                mask[speaker_idx, st:et] = 1
        return mask

    def speakers_audio_mask(
        self,
        min_speaker_dim: Optional[int] = None,
        speaker_to_idx_map: Optional[Dict[str, int]] = None,
        use_alignment_if_exists: Optional[str] = None,
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
        assert self.has_recording, (
            f"No recording available. "
            f"Can't compute supervisions audio mask for cut with ID: {self.id}."
        )
        if speaker_to_idx_map is None:
            speaker_to_idx_map = {
                spk: idx
                for idx, spk in enumerate(
                    sorted(set(s.speaker for s in self.supervisions))
                )
            }
        num_speakers = len(speaker_to_idx_map)
        if min_speaker_dim is not None:
            num_speakers = min(min_speaker_dim, num_speakers)
        mask = np.zeros((num_speakers, self.num_samples))
        for supervision in self.supervisions:
            speaker_idx = speaker_to_idx_map[supervision.speaker]
            if (
                use_alignment_if_exists
                and supervision.alignment
                and use_alignment_if_exists in supervision.alignment
            ):
                for ali in supervision.alignment[use_alignment_if_exists]:
                    st = (
                        compute_num_samples(ali.start, self.sampling_rate)
                        if ali.start > 0
                        else 0
                    )
                    et = (
                        compute_num_samples(ali.end, self.sampling_rate)
                        if ali.end < self.duration
                        else compute_num_samples(self.duration, self.sampling_rate)
                    )
                    mask[speaker_idx, st:et] = 1
            else:
                st = (
                    compute_num_samples(supervision.start, self.sampling_rate)
                    if supervision.start > 0
                    else 0
                )
                et = (
                    compute_num_samples(supervision.end, self.sampling_rate)
                    if supervision.end < self.duration
                    else compute_num_samples(self.duration, self.sampling_rate)
                )
                mask[speaker_idx, st:et] = 1
        return mask

    def supervisions_feature_mask(
        self, use_alignment_if_exists: Optional[str] = None
    ) -> np.ndarray:
        """
        Return a 1D numpy array with value 1 for **frames** covered by at least one supervision,
        and 0 for **frames** not covered by any supervision.

        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        from .set import compute_supervisions_frame_mask

        return compute_supervisions_frame_mask(
            self, use_alignment_if_exists=use_alignment_if_exists
        )

    def supervisions_audio_mask(
        self, use_alignment_if_exists: Optional[str] = None
    ) -> np.ndarray:
        """
        Return a 1D numpy array with value 1 for **samples** covered by at least one supervision,
        and 0 for **samples** not covered by any supervision.

        :param use_alignment_if_exists: optional str, key for alignment type to use for generating the mask. If not
            exists, fall back on supervision time spans.
        """
        assert self.has_recording, (
            f"No recording available. "
            f"Can't compute supervisions audio mask for cut with ID: {self.id}."
        )
        mask = np.zeros(self.num_samples, dtype=np.float32)
        for supervision in self.supervisions:
            if (
                use_alignment_if_exists
                and supervision.alignment
                and use_alignment_if_exists in supervision.alignment
            ):
                for ali in supervision.alignment[use_alignment_if_exists]:
                    st = round(ali.start * self.sampling_rate) if ali.start > 0 else 0
                    et = (
                        round(ali.end * self.sampling_rate)
                        if ali.end < self.duration
                        else round(self.duration * self.sampling_rate)
                    )
                    mask[st:et] = 1.0
            else:
                st = (
                    round(supervision.start * self.sampling_rate)
                    if supervision.start > 0
                    else 0
                )
                et = (
                    round(supervision.end * self.sampling_rate)
                    if supervision.end < self.duration
                    else round(self.duration * self.sampling_rate)
                )
                mask[st:et] = 1.0
        return mask

    def with_id(self, id_: str) -> "Cut":
        """Return a copy of the Cut with a new ID."""
        return fastcopy(self, id=id_)
