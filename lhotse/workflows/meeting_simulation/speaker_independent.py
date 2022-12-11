import functools
import logging
import random
from typing import List, Optional, Union

import numpy as np

from lhotse import RecordingSet, SupervisionSet
from lhotse.cut import CutSet, MixedCut, MixTrack, MonoCut
from lhotse.cut.set import mix
from lhotse.utils import uuid4
from lhotse.workflows.meeting_simulation.base import (
    BaseMeetingSimulator,
    reverberate_cuts,
)


class SpeakerIndependentMeetingSimulator(BaseMeetingSimulator):
    """
    This simulator uses the simulation method used in the end-to-end neural diarization (EEND)
    paper: https://arxiv.org/abs/1909.06247 (Algorithm 1). It samples segments of each speaker from the
    input CutSet, and concatenates them into speaker-specific channels, with pauses sampled
    from an exponential distribution. The speaker channels are then mixed together
    (possibly after adding room impulse responses) to create the simulated meeting.
    Since the speakers are simulated independently, the resulting mixtures can contain more
    overlap than is usually present in real meetings.

    In the paper, a single hyper-parameter `beta` is used which is equivalent to the scale
    parameter of the exponential distribution. Here, we use both `loc` and `scale`, where
    `loc` would mean the minimum silence duration between two consecutive utterances from
    the same speaker. These parameters can be either provided in initialization, or learned
    from a dataset using the `fit()` method.
    """

    def __init__(self, loc: float = 0.0, scale: float = 2.0):
        """
        :param loc: the minimum silence duration between two consecutive utterances from
            the same speaker. [Default: 0.0]
        :param scale: the scale parameter of the exponential distribution used to sample
            the silence duration between two consecutive utterances from a speaker.
            [Default: 2.0]
        """
        super().__init__()
        self.loc = loc
        self.scale = scale

    def __repr__(self):
        return self.__class__.__name__ + f"(loc={self.loc}, scale={self.scale})"

    def fit(self, meetings: Optional[SupervisionSet] = None) -> None:
        """
        Learn the distribution of the meeting parameters from a given dataset.
        :param meetings: a SupervisionSet containing the meetings to be used for
        """
        if meetings is None:
            logging.info(
                f"No meetings provided, using default parameters: loc={self.loc}, scale={self.scale}"
            )
            return

        assert isinstance(
            meetings, SupervisionSet
        ), "The meetings must be provided as a SupervisionSet."

        from cytoolz.itertoolz import groupby
        from scipy.stats import expon

        # Group the segments by recording ID and speaker ID.
        speaker_segments = groupby(
            lambda s: (s.recording_id, s.speaker),
            sorted(meetings, key=lambda s: (s.recording_id, s.speaker)),
        )

        # Compute the inter-speech intervals for each speaker.
        inter_speech_intervals = []
        for segments in speaker_segments.values():
            segments = sorted(segments, key=lambda s: s.start)
            for i in range(1, len(segments)):
                inter_speech_intervals.append(
                    max(0, segments[i].start - segments[i - 1].end)
                )

        # Fit an exponential distribution to the inter-speech intervals.
        self.loc, self.scale = expon.fit(inter_speech_intervals)

        print(f"Learned parameters: loc={self.loc:.2f}, scale={self.scale:.2f}")

    @functools.lru_cache(maxsize=128)
    def _get_speaker_cuts(self, base_cuts_idx: int, speaker: str) -> CutSet:
        """
        Return all cuts for the given speaker.
        """
        return list(
            self.cuts[base_cuts_idx]
            .filter(lambda cut: cut.supervisions[0].speaker == speaker)
            .cuts.values()
        )

    def _create_mixture(
        self, utterances: List[List[MonoCut]], silence_durations: List[np.array]
    ) -> MixedCut:
        """
        Create a MixedCut object from a list of speaker-wise MonoCuts and silence intervals.
        Each `track` in the resulting MixedCut represents a different speaker. Each `track`
        itself can be a MonoCut or a MixedCut (if the speaker has multiple utterances).
        """
        tracks = []
        for i, (spk_utterances, spk_silences) in enumerate(
            zip(utterances, silence_durations)
        ):
            track = spk_utterances[0]
            for sil, utt in zip(spk_silences[1:], spk_utterances[1:]):
                track = mix(track, utt, offset=track.duration + sil, allow_padding=True)
            # NOTE: First track must have an offset of 0.0.
            track = MixTrack(
                cut=track, type=type(track), offset=(0 if i == 0 else spk_silences[0])
            )
            tracks.append(track)
        return MixedCut(id=str(uuid4()), tracks=tracks)

    def simulate(
        self,
        *cuts: CutSet,
        num_meetings: int = 1,
        num_speakers_per_meeting: Union[int, List[int]] = 2,
        speaker_count_probs: Optional[List[float]] = None,
        min_utts_per_speaker: int = 5,
        max_utts_per_speaker: int = 10,
        seed: int = 0,
    ) -> CutSet:
        """
        Simulate the desired number of multi-speaker meetings.
        :param cuts: one or more CutSet containing the MonoCut objects to be used for simulation.
            If multiple CutSets are provided, each mixture will contain cuts sampled from
            a specific CutSet. This may be useful when we want to simulate mixtures with
            cuts from the same recording.
        :param num_meetings: the number of meetings to simulate.
        :param num_speakers_per_meeting: the number of speakers per meeting. If a list is
            provided, the number of speakers per meeting is sampled from this list.
            [Default: 2]
        :param speaker_count_probs: the probability of each number of speakers per meeting.
            [Default: None]
        :param min_utts_per_speaker: the minimum number of utterances per speaker to be
            used for simulation. [Default: 5]
        :param max_utts_per_speaker: the maximum number of utterances per speaker to be
            used for simulation. [Default: 10]
        :param seed: the random seed to be used for simulation. [Default: 0]
        """
        assert len(cuts) > 0, "At least one CutSet must be provided."

        if isinstance(num_speakers_per_meeting, int):
            num_speakers_per_meeting = [num_speakers_per_meeting]

        if speaker_count_probs is None:
            speaker_count_probs = [1.0 / len(num_speakers_per_meeting)] * len(
                num_speakers_per_meeting
            )

        assert len(num_speakers_per_meeting) == len(
            speaker_count_probs
        ), "The number of speakers per meeting and the number of probabilities must be the same."

        self.cuts = cuts
        # Make sure there are only MonoCuts in the CutSets.
        assert all(
            len(base_cuts.simple_cuts) == len(base_cuts) for base_cuts in cuts
        ), "The CutSets must contain only MonoCuts. "

        # Create random number generators with the given seed.
        npr = np.random.RandomState(seed)
        rand = random.Random(seed)

        # Reset speaker bucket cache.
        self._get_speaker_cuts.cache_clear()

        mixtures = []

        for _ in range(num_meetings):

            # Sample the cut-set that this meeting will be generated from.
            base_cuts_idx = npr.randint(len(cuts))
            base_cuts = cuts[base_cuts_idx]

            # Sample the number of speakers for this meeting.
            num_speakers = npr.choice(num_speakers_per_meeting, p=speaker_count_probs)

            # Sample the speakers for this meeting.
            speakers = rand.sample(
                base_cuts.speakers,
                k=min(num_speakers, len(base_cuts.speakers)),
            )

            # Sample the utterances for each speaker. We use `replace=True` here because
            # we may need to use the same utterance multiple times to get the desired
            # number of utterances per speaker.
            utterances = [
                rand.choices(
                    self._get_speaker_cuts(base_cuts_idx, speaker),
                    k=npr.randint(min_utts_per_speaker, max_utts_per_speaker + 1),
                )
                for speaker in speakers
            ]

            # Sample the silence durations between utterances for each speaker.
            silence_durations = [
                self.loc + npr.exponential(scale=self.scale, size=len(utterances[i]))
                for i in range(len(utterances))
            ]

            # Create the meeting.
            mixture = self._create_mixture(utterances, silence_durations)

            mixtures.append(mixture)

        return CutSet.from_cuts(mixtures)

    def reverberate(self, cuts: CutSet, rirs: Optional[RecordingSet] = None) -> CutSet:
        return reverberate_cuts(cuts, rirs)
