import random
from typing import List, Sequence, Union

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
            factors: Union[float, Sequence[float]],
            p: float,
            randgen: random.Random = None
    ) -> None:
        self.factors = factors if isinstance(factors, Sequence) else [factors]
        self.p = p
        self.random = randgen

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.random is None:
            self.random = random
        return CutSet.from_cuts(
            cut.perturb_volume(factor=self.random.choice(self.factors))
            if self.random.random() >= self.p
            else cut
            for cut in cuts
        )
