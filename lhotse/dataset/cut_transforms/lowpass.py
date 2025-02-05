import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

from lhotse import CutSet


@dataclass
class Lowpass:
    """
    Applies a low-pass filter to each Cut in a CutSet.

    The filter is applied with a probability of ``p``. When applied, the filter
    randomly selects a cutoff frequency from the list of provided frequencies,
    with optional weights controlling the selection.

    :param frequencies: A list of cutoff frequencies.
    :param weights: Optional weights for each frequency (default: equal weights).
    :param p: The probability of applying the low-pass filter (default: 0.5).
    :param randgen: An optional random number generator (default: a new instance).
    """

    frequencies: List[float]
    weights: Optional[List[float]] = None
    p: float = 0.5
    randgen: random.Random = None

    def __post_init__(self) -> None:
        if self.weights:
            assert len(self.weights) == len(self.frequencies)
        else:
            # all codecs have equal weights by default
            self.weights = [1.0 for _ in self.frequencies]

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.randgen is None:
            self.randgen = random.Random()

        lowpassed_cuts = []
        for cut in cuts:
            frequency, *_ = self.randgen.choices(self.frequencies, weights=self.weights)

            if self.randgen.random() <= self.p:
                new_cut = cut.lowpass(frequency)
                new_cut.id = f"{cut.id}_lowpassed{frequency:.0f}"
                lowpassed_cuts.append(new_cut)
            else:
                lowpassed_cuts.append(cut)

        return CutSet(lowpassed_cuts)
