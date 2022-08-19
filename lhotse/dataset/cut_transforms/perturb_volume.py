import random
from typing import Sequence, Union

from lhotse import CutSet


class PerturbVolume:
    """
    A transform on batch of cuts (``CutSet``) that perturbs the volume of the recordings
    with a given probability :attr:`p`.

    If the effect is applied, then one of the perturbation factors from the constructor's
    :attr:`factors` parameter is sampled with uniform probability.
    """

    def __init__(
        self,
        p: float,
        scale_low: float = 0.125,
        scale_high: float = 2.0,
        randgen: random.Random = None,
        preserve_id: bool = False,
    ) -> None:
        self.p = p
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.random = randgen
        self.preserve_id = preserve_id

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.random is None:
            self.random = random
        return CutSet.from_cuts(
            cut.perturb_volume(
                factor=self.random.uniform(self.scale_low, self.scale_high),
                affix_id=not self.preserve_id,
            )
            if self.random.random() <= self.p
            else cut
            for cut in cuts
        )
