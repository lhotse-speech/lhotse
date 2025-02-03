import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

from lhotse import CutSet


@dataclass
class Lowpass:
    """
    For every Cut,
    1) randomly choose a corner frequency from provided list
    2) apply the lowpass filter with chosen frequency with probability p
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
