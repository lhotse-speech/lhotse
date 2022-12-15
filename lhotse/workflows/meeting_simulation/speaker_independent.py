import logging
from typing import List, Optional, Union

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from lhotse import RecordingSet, SupervisionSet
from lhotse.cut import CutSet, MixedCut, MixTrack, MonoCut
from lhotse.cut.set import mix
from lhotse.utils import uuid4
from lhotse.workflows.meeting_simulation.base import (
    BaseMeetingSimulator,
    create_sampler,
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
        cuts: CutSet,
        num_meetings: Optional[int] = None,
        num_repeats: Optional[int] = None,
        num_speakers_per_meeting: Union[int, List[int]] = 2,
        speaker_count_probs: Optional[List[float]] = None,
        max_duration_per_speaker: Optional[float] = 20.0,
        max_utterances_per_speaker: Optional[int] = 5,
        seed: int = 0,
    ) -> CutSet:
        """
        Simulate the desired number of multi-speaker meetings.
        :param cuts: CutSet containing the MonoCut objects to be used for simulation.
        :param num_meetings: the number of meetings to simulate.
            [Default: None]
        :param num_repeats: the number of times to repeat the provided cuts. This means that
            the number of simulated meetings depends on how many cuts are available.
        :param num_speakers_per_meeting: the number of speakers per meeting. If a list is
            provided, the number of speakers per meeting is sampled from this list.
            [Default: 2]
        :param speaker_count_probs: the probability of each number of speakers per meeting.
            [Default: None]
        :param max_duration_per_speaker: the maximum duration of a speaker's utterances.
            [Default: 20.0]
        :param max_utterances_per_speaker: the maximum number of utterances per speaker.
            [Default: 5]
        :param seed: the random seed to be used for simulation. [Default: 0]
        """
        if num_meetings is None and num_repeats is None:
            raise ValueError("Either num_meetings or num_repeats must be provided.")

        if isinstance(num_speakers_per_meeting, int):
            num_speakers_per_meeting = [num_speakers_per_meeting]

        if speaker_count_probs is None:
            speaker_count_probs = [1.0 / len(num_speakers_per_meeting)] * len(
                num_speakers_per_meeting
            )

        assert len(num_speakers_per_meeting) == len(
            speaker_count_probs
        ), "The number of speakers per meeting and the number of probabilities must be the same."

        # Make sure there are only MonoCuts in the CutSet.
        assert len(cuts) == len(cuts.simple_cuts), "Only MonoCuts are supported."

        cuts = cuts.repeat(times=num_repeats)

        # Create cuts sampler
        sampler = create_sampler(
            cuts,
            max_duration=max_duration_per_speaker,
            max_cuts=max_utterances_per_speaker,
            seed=seed,
        )
        # Create an iterator from the sampler
        sampler_iter = iter(sampler)

        # Create random number generators with the given seed.
        npr = np.random.RandomState(seed)

        mixtures = []

        pbar = tqdm(total=num_meetings)
        while True:
            pbar.update(1)

            # If the number of meetings is provided, stop when we reach that number.
            if num_meetings is not None and len(mixtures) >= num_meetings:
                break

            # Sample the number of speakers for this meeting.
            num_speakers = npr.choice(num_speakers_per_meeting, p=speaker_count_probs)

            # Sample from the sampler to get 1 batch per desired number of speakers.
            utterances = []
            finished = False
            for _ in range(num_speakers):
                try:
                    this_batch = next(sampler_iter).data
                except StopIteration:
                    # If we run out of data, finish simulation.
                    finished = True
                    break
                utterances.append(list(this_batch.values()))

            if finished:
                break

            # Sample the silence durations between utterances for each speaker.
            silence_durations = [
                self.loc + npr.exponential(scale=self.scale, size=len(utterances[i]))
                for i in range(len(utterances))
            ]

            # Create the meeting.
            mixture = self._create_mixture(utterances, silence_durations)

            mixtures.append(mixture)

        return CutSet.from_cuts(mixtures)

    def reverberate(self, cuts: CutSet, *rirs: RecordingSet) -> CutSet:
        return reverberate_cuts(cuts, *rirs)
