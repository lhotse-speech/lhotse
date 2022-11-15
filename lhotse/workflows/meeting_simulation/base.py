"""
This is an experimental workflow that can be used to simulate multi-speaker meetings from
a CutSet containing MonoCut objects.
"""
import abc
from typing import Optional

from lhotse import RecordingSet, SupervisionSet
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
