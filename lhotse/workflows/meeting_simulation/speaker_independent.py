import logging
from collections import defaultdict
from functools import partial
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from lhotse import RecordingSet, SupervisionSet
from lhotse.cut import CutSet, MixedCut, MixTrack
from lhotse.cut.set import mix
from lhotse.parallel import parallel_map
from lhotse.utils import uuid4
from lhotse.workflows.meeting_simulation.base import (
    MAX_TASKS_WAITING,
    BaseMeetingSimulator,
    MeetingSampler,
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
        self, utterances: List[CutSet], silence_durations: List[np.array]
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
            # Get list of cuts from CutSet
            spk_utterances = list(spk_utterances.data.values())
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
        num_jobs: int = 1,
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
        :param num_jobs: the number of jobs to use for simulation. Use more jobs to speed up
            simulation when you have large number of source utterances. [Default: 1]
        """
        if num_meetings is None and num_repeats is None:
            raise ValueError("Either num_meetings or num_repeats must be provided.")

        if num_meetings is not None:
            num_repeats = None

        if isinstance(num_speakers_per_meeting, int):
            num_speakers_per_meeting = [num_speakers_per_meeting]

        if speaker_count_probs is None:
            speaker_count_probs = [1.0 / len(num_speakers_per_meeting)] * len(
                num_speakers_per_meeting
            )

        # Create cuts sampler
        sampler = MeetingSampler(
            cuts,
            num_repeats=num_repeats,
            num_meetings=num_meetings,
            max_duration_per_speaker=max_duration_per_speaker,
            max_utterances_per_speaker=max_utterances_per_speaker,
            num_speakers_per_meeting=num_speakers_per_meeting,
            speaker_count_probs=speaker_count_probs,
            seed=seed,
        )
        sampler_iter = iter(sampler)

        work = partial(_simulate_worker, seed=seed, simulator=self)

        mixtures = []
        if num_jobs == 1:
            # Don't use multiprocessing if num_jobs == 1.
            for mixture in tqdm(map(work, sampler_iter), total=num_meetings):
                mixtures.append(mixture)
        else:
            for mixture in tqdm(
                parallel_map(
                    work,
                    sampler_iter,
                    num_jobs=num_jobs,
                    queue_size=num_jobs * MAX_TASKS_WAITING,
                ),
                total=num_meetings,
                desc="Simulating meetings",
            ):
                mixtures.append(mixture)

        return CutSet.from_cuts(mixtures)

    def reverberate(self, cuts: CutSet, *rirs: RecordingSet) -> CutSet:
        return reverberate_cuts(cuts, *rirs)


def _simulate_worker(
    utterances: CutSet,
    seed: int,
    simulator: SpeakerIndependentMeetingSimulator,
) -> MixedCut:
    # Create random number generators with the given seed.
    npr = np.random.RandomState(seed)

    # Group the cuts by speaker.
    utts_by_speaker = defaultdict(list)
    for utt in utterances:
        utts_by_speaker[utt.supervisions[0].speaker].append(utt)

    utterances = [CutSet.from_cuts(cuts) for cuts in utts_by_speaker.values()]

    # Sample the silence durations between utterances for each speaker.
    silence_durations = [
        simulator.loc + npr.exponential(scale=simulator.scale, size=len(utterances[i]))
        for i in range(len(utterances))
    ]

    # Create the meeting.
    mixture = simulator._create_mixture(utterances, silence_durations)
    return mixture
