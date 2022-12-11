"""
This is an experimental workflow that can be used to simulate multi-speaker meetings from
a CutSet containing MonoCut objects.
"""
import abc
from typing import Optional

from lhotse import RecordingSet, SupervisionSet
from lhotse.augmentation.utils import FastRandomRIRGenerator, fastcopy
from lhotse.cut import CutSet


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
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abc.abstractmethod
    def fit(self, meetings: Optional[SupervisionSet] = None) -> None:
        """
        Learn the distribution of the meeting parameters from a given dataset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def simulate(self, *cuts: CutSet, num_meetings: int = 1) -> CutSet:
        """
        Simulate the desired number of multi-speaker meetings.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reverberate(self, cuts: CutSet, rirs: Optional[RecordingSet] = None) -> CutSet:
        """
        Apply a reverberation effect to each track.
        """
        raise NotImplementedError


def reverberate_cuts(cuts: CutSet, *rirs: Optional[RecordingSet]) -> CutSet:
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
    max_sources = max(len(rir_group) for rir_group in rirs) if rirs is not None else 0
    for cut in cuts:
        tracks = []
        num_speakers = len(cut.tracks)
        if num_speakers <= max_sources:
            # Choose a random RIR group containing as many recordings as there are
            # speakers (sources) in the mixed cut.
            rir_group = rirs.filter(lambda r: len(r) == num_speakers).random()
            for track, rir in zip(cut.tracks, rir_group):
                tracks.append(fastcopy(track, cut=track.cut.reverb_rir(rir)))
        else:
            # Generate a fast randomized RIR with as many sources as there are speakers.
            rir_generator = FastRandomRIRGenerator()
