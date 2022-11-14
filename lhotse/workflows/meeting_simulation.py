"""
This is an experimental workflow that can be used to simulate multi-speaker meetings from
a CutSet containing MonoCut objects.
"""
import abc
import logging
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

from lhotse import (
    CutSet,
    MonoCut,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
)


class BaseMeetingSimulator(abc.ABC):
    """
    Base class for meeting simulators. A simulator consists of a `fit()` and a `simulate()`
    method. The `fit()` method is used to learn the distribution of the meeting parameters
    (e.g. turn-taking, overlap ratio, etc.) from a given dataset (e.g. from a CutSet or a
    SupervisionSet). The `simulate()` method takes a CutSet containing MonoCut objects and
    simulates the desired number of multi-speaker meetings, based on the learned distribution.

    The base class should be inherited from and the `fit()` and `simulate()` methods
    should be implemented.

    Example usage:
    >>> simulator = MyMeetingSimulator()
    >>> simulator.fit(cuts)
    >>> simulated_cuts = simulator.simulate(mono_cuts, num_meetings=10)
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, meetings: Optional[SupervisionSet] = None) -> None:
        """
        Learn the distribution of the meeting parameters from a given dataset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def simulate(self, cuts: CutSet, num_meetings: int) -> CutSet:
        """
        Simulate the desired number of multi-speaker meetings.
        """
        raise NotImplementedError


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

    def __init__(self, loc: Optional[float] = 0.0, scale: Optional[float] = 2.0):
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

    def fit(self, meetings: Optional[SupervisionSet] = None) -> None:
        """
        Learn the distribution of the meeting parameters from a given dataset.
        :param meetings: a CutSet or SupervisionSet containing the meetings to be used for
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
        self.loc, self.scale, _ = expon.fit(inter_speech_intervals)

        logging.info(f"Learned parameters: loc={self.loc}, scale={self.scale}")

    def simulate(self, cuts: CutSet, num_meetings: int) -> CutSet:
        """
        Simulate the desired number of multi-speaker meetings.
        :param cuts: a CutSet containing the MonoCut objects to be used for simulation.
        :param num_meetings: the number of meetings to simulate.
        """
        from lhotse.cut import MixedCut
        from lhotse.dataset.sampling import CutSampler
        from lhotse.utils import compute_num_frames

        assert isinstance(cuts, CutSet), "The cuts must be provided as a CutSet."

        # Group the cuts by recording ID and speaker ID.
        cuts_by_recording_and_speaker = cuts.group_by(
            lambda c: (c.recording_id, c.speaker)
        )

        # Create a CutSampler for each speaker.
        cut_samplers = {
            (recording_id, speaker): CutSampler(cuts)
            for (recording_id, speaker), cuts in cuts_by_recording_and_speaker.items()
        }

        # Create the simulated meetings.
        simulated_cuts = []
        for _ in range(num_meetings):
            # Sample the number of speakers.
            num_speakers = self._sample_num_speakers()

            # Sample the speakers.
            speakers = self._sample_speakers(cut_samplers, num_speakers)

            # Sample the utterances for each speaker.
            utterances = self._sample_utterances(cut_samplers, speakers)

            # Create the simulated meeting.
            simulated_cuts.append(self._create_meeting(utterances))

        return CutSet.from_cuts(simulated_cuts)
