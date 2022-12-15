"""
This is an experimental workflow that can be used to simulate multi-speaker meetings from
a CutSet containing MonoCut objects.
"""
import abc
from collections import defaultdict
from typing import Optional

from torch.utils.data import Dataset

from lhotse import RecordingSet, SupervisionSet
from lhotse.cut import CutSet
from lhotse.dataset.sampling import RoundRobinSampler, SimpleCutSampler
from lhotse.utils import fastcopy


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
    def simulate(
        self,
        cuts: CutSet,
        num_meetings: Optional[int] = None,
        num_repeats: Optional[int] = None,
    ) -> CutSet:
        """
        Simulate the desired number of multi-speaker meetings.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reverberate(self, cuts: CutSet, *rirs: RecordingSet) -> CutSet:
        """
        Apply a reverberation effect to each track.
        """
        raise NotImplementedError


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


def create_sampler(
    cuts: CutSet, max_duration: float = None, max_cuts: int = None, seed: int = 0
) -> RoundRobinSampler:
    """
    Create a sampler that will be used to sample cuts from the input CutSet. The cuts
    are partitioned into speaker-wise buckets, and a DynamicCutSampler is created for
    each bucket. The samplers are then combined into a RoundRobinSampler, which will
    sample cuts from each bucket in a round-robin fashion.

    :param cuts: a CutSet containing MonoCut objects.
    :param max_duration: the maximum duration of the cuts in each batch.
    :param max_cuts: the maximum number of cuts in each batch.
    :param seed: the random seed.
    :return: a RoundRobinSampler object.
    """
    # Create buckets by speaker.
    buckets = defaultdict(list)
    for cut in cuts:
        buckets[cut.supervisions[0].speaker].append(cut)

    buckets = [CutSet.from_cuts(cuts) for cuts in buckets.values()]

    # Create samplers.
    samplers = []
    for bucket in buckets:
        samplers.append(
            SimpleCutSampler(
                bucket,
                max_duration=max_duration,
                max_cuts=max_cuts,
                shuffle=True,
                seed=seed,
            )
        )
    sampler = RoundRobinSampler(*samplers)
    return sampler


class CutReservoirDataset(Dataset):
    """
    A PyTorch Dataset that samples cuts from a CutSet using a sampler.
    """

    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, cuts: CutSet) -> CutSet:
        return cuts
