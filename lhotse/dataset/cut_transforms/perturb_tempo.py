import random
from typing import Sequence, Union

from lhotse import CutSet


class PerturbTempo:
    """
    A transform on batch of cuts (``CutSet``) that perturbs the tempo of the recordings
    with a given probability :attr:`p`.

    If the effect is applied, then one of the perturbation factors from the constructor's
    :attr:`factors` parameter is sampled with uniform probability.
    """

    def __init__(
        self,
        factors: Union[float, Sequence[float]],
        p: float,
        randgen: random.Random = None,
        preserve_id: bool = False,
    ) -> None:
        self.factors = factors if isinstance(factors, Sequence) else [factors]
        self.p = p
        self.random = randgen
        self.preserve_id = preserve_id

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.random is None:
            self.random = random
        return CutSet.from_cuts(
            cut.perturb_tempo(
                factor=self.random.choice(self.factors), affix_id=not self.preserve_id
            )
            if self.random.random() <= self.p
            else cut
            for cut in cuts
        )
