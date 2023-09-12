"""
This is an experimental workflow that can be used to simulate multi-speaker meetings from
a CutSet containing MonoCut objects.
"""
import abc
import random
from itertools import groupby
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from lhotse import RecordingSet, SupervisionSet
from lhotse.cut import CutSet
from lhotse.dataset.sampling import DynamicCutSampler
from lhotse.utils import fastcopy, is_module_available

MAX_TASKS_WAITING = 1000


class BaseMeetingSimulator(abc.ABC):
    """
    Base class for meeting simulators. A simulator consists of a `fit()`, a `simulate()`,
    and a `reverberate()` method.

    The `fit()` method is used to learn the distribution of the meeting parameters
    (e.g. turn-taking, overlap ratio, etc.) from a given dataset, presented in the form of
    a SupervisionSet. The parameters themselves are simulator specific.

    The `simulate()` method takes a CutSet containing MonoCut objects and simulates the
    desired number of multi-speaker meetings, based on the learned distribution. The output
    is a CutSet containing MixedCut objects, where each track represents a different speaker.

    The `reverberate()` method takes a CutSet containing MixedCut objects (usually the output
    of `simulate()`) and applies a reverberation effect to each track. We can apply single
    or multi-channel room impulse responses (RIRs) to each track. The output is a CutSet
    containing MixedCut objects, where each track represents a different speaker, convolved
    with a different RIR.

    The base class should be inherited from and the different methods should be implemented.

    The output is expected to be a CutSet containing MixedCut objects, where each track
    represents a different speaker, possibly convolved with a different RIR. This is
    analogous to the "mixture model" of speech signals.

    Example usage:
    >>> simulator = MyMeetingSimulator()
    >>> simulator.fit(cuts)
    >>> simulated_cuts = simulator.simulate(mono_cuts, num_meetings=10)
    >>> simulated_cuts = simulator.reverberate(simulated_cuts, rirs)
    """

    def __init__(self):
        if type(self) is BaseMeetingSimulator:
            raise TypeError(
                "BaseMeetingSimulator is an abstract base class and should not be instantiated."
            )

        if not is_module_available("scipy"):
            raise ImportError("Please 'pip install scipy' first.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @abc.abstractmethod
    def fit(self, meetings: Optional[SupervisionSet] = None) -> None:
        """
        Learn the distribution of the meeting parameters from a given dataset.
        """
        ...

    @abc.abstractmethod
    def simulate(
        self,
        cuts: CutSet,
        num_meetings: Optional[int] = None,
        num_repeats: Optional[int] = None,
    ) -> CutSet:
        """
        Simulate the desired number of multi-speaker meetings.
        """
        ...

    @abc.abstractmethod
    def reverberate(self, cuts: CutSet, *rirs: RecordingSet) -> CutSet:
        """
        Apply a reverberation effect to each track.
        """
        ...


class MeetingSampler:
    """
    Create a sampler that will be used to sample groups of utterances from the sources.
    The cuts are partitioned into speaker-wise buckets, and a SimpleCutSampler is created
    for each bucket. When we sample a group of utterances, we first sample the number of
    speakers in the meeting, and then sample the utterances of each speaker. This is done
    by sampling a batch from the corresponding SimpleCutSampler.

    :param cuts: a CutSet containing MonoCut objects.
    :param num_repeats: the number of times each cut will be repeated (by default, they
        are repeated infinitely).
    :param num_meetings: the number of meetings to simulate.
    :param num_speakers_per_meeting: the number of speakers per meeting.
    :param speaker_count_probs: the probabilities of the number of speakers per meeting.
    :param max_duration_per_speaker: the maximum duration of a speaker in a meeting.
    :param max_utterances_per_speaker: the maximum number of utterances of a speaker in a meeting.
    :param seed: the random seed.
    :return: a DynamicCutSampler object.
    """

    def __init__(
        self,
        cuts: CutSet,
        num_repeats: Optional[int] = None,
        num_meetings: Optional[int] = None,
        num_speakers_per_meeting: Union[int, List[int]] = 2,
        speaker_count_probs: Optional[List[float]] = None,
        max_duration_per_speaker: Optional[float] = 20.0,
        max_utterances_per_speaker: Optional[int] = 5,
        seed: int = 0,
    ):
        # Some basic checks
        assert all(n > 1 for n in num_speakers_per_meeting), (
            "The number of speakers per meeting must be greater than 1. "
            f"Got: {num_speakers_per_meeting}"
        )
        assert all(p > 0.0 for p in speaker_count_probs), (
            "The probabilities of the number of speakers per meeting must be greater than 0. "
            f"Got: {speaker_count_probs}"
        )
        assert sum(speaker_count_probs) == 1.0, (
            "The probabilities of the number of speakers per meeting must sum to 1. "
            f"Got: {speaker_count_probs}"
        )
        assert len(num_speakers_per_meeting) == len(
            speaker_count_probs
        ), "The number of speakers per meeting and the number of probabilities must be the same."

        # Create samplers for each bucket. We create this as a dict so that we can
        # efficiently remove items and also randomly sample items in constant time.
        # It also supports the len() function in constant time.
        # Note that a Python list is not a good choice here, because removing items
        # from a list is O(n). A set is also not a good choice, because randomly
        # sampling items from a set is O(n).
        self.samplers = {}
        for spk, spk_cuts in tqdm(
            groupby(
                sorted(cuts, key=lambda cut: cut.supervisions[0].speaker),
                lambda cut: cut.supervisions[0].speaker,
            ),
            desc="Creating samplers for each speaker...",
        ):
            sampler = DynamicCutSampler(
                CutSet.from_cuts(list(spk_cuts)).repeat(
                    times=num_repeats, preserve_id=False
                ),
                max_duration=max_duration_per_speaker,
                max_cuts=max_utterances_per_speaker,
                shuffle=True,
                seed=seed,
            )
            self.samplers[spk] = sampler

        self.num_speakers_per_meeting = num_speakers_per_meeting
        self.speaker_count_probs = speaker_count_probs

        self.npr = np.random.RandomState(seed)
        self.rng = random.Random(seed)
        self._remaining_meetings = num_meetings

    def __iter__(self):
        for sampler in self.samplers.values():
            iter(sampler)
        return self

    def __next__(self):
        # If we have sampled enough meetings, stop.
        if self._remaining_meetings is not None and self._remaining_meetings == 0:
            raise StopIteration()

        # If we don't have enough speakers, stop.
        if len(self.samplers) < min(self.num_speakers_per_meeting):
            raise StopIteration()

        # Sample the number of speakers for this meeting.
        N = min(
            self.npr.choice(self.num_speakers_per_meeting, p=self.speaker_count_probs),
            len(self.samplers),
        )

        # Sample speakers.
        this_batch_spk_ids = self.rng.sample(list(self.samplers.keys()), N)
        utterances = CutSet.from_cuts([])
        for spk_id in this_batch_spk_ids:
            sampler = self.samplers[spk_id]
            try:
                this_batch = next(sampler)
                utterances += this_batch
            except StopIteration:
                del self.samplers[spk_id]
                continue

        # shuffle the utterances
        utterances = utterances.shuffle()

        if self._remaining_meetings is not None:
            self._remaining_meetings -= 1
        return utterances if len(utterances) > 0 else next(self)


def reverberate_cuts(cuts: CutSet, *rirs: RecordingSet) -> CutSet:
    """
    Use provided RIRs to convolve each track of the input CutSet. The cuts here are
    MixedCut objects containing different speakers in different tracks. To reverberate,
    we choose a random RIR containing as many Recording objects as there are tracks
    in the MixedCut.

    If impulse responses are not provided, we use the fast randomized approximation
    method to simulate artificial single-channel RIRs.

    :param cuts: a CutSet containing MixedCut objects.
    :param rirs: one or more RecordingSet (each set is a group of RIRs from the same room).
    :return: a CutSet containing MixedCut objects reverberated with the provided RIRs.
    """
    out_cuts = []
    max_sources = max(len(rir_group) for rir_group in rirs) if len(rirs) > 0 else 0
    for cut in cuts:
        num_speakers = len(cut.tracks)
        if num_speakers <= max_sources:
            tracks = []
            # Choose a random RIR group containing as many recordings as there are
            # speakers (sources) in the mixed cut.
            rir_group = rirs.filter(lambda r: len(r) == num_speakers).random()
            for track, rir in zip(cut.tracks, rir_group):
                tracks.append(fastcopy(track, cut=track.cut.reverb_rir(rir)))
            out_cuts.append(fastcopy(cut, tracks=tracks))
        else:
            # We will use a fast random approximation to generate RIRs.
            out_cuts.append(cut.reverb_rir())

    return CutSet.from_cuts(out_cuts)
