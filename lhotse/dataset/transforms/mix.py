from typing import Optional, Tuple, Union

from lhotse import CutSet
from lhotse.utils import Decibels


class CutMix:
    """
    A transform for batches of cuts (CutSet's) that stochastically performs
    noise augmentation with a constant or varying SNR.
    """

    def __init__(
            self,
            cuts: Optional[CutSet] = None,
            snr: Optional[Union[Decibels, Tuple[Decibels, Decibels]]] = (10, 20),
            prob: float = 0.5
    ) -> None:
        """
        CutMix's constructor.

        :param cuts: a ``CutSet`` containing augmentation data, e.g. noise, music, babble.
        :param snr: either a float, a pair (range) of floats, or ``None``.
            It determines the SNR of the speech signal vs the noise signal that's mixed into it.
            When a range is specified, we will uniformly sample SNR in that range.
            When it's ``None``, the noise will be mixed as-is -- i.e. without any level adjustment.
            Note that it's different from ``snr=0``, which will adjust the noise level so that the SNR is 0.
        :param prob: a float probability in range [0, 1].
            Specifies the probability with which we will mix augment the cuts.
        """
        self.cuts = cuts
        self.snr = snr
        self.prob = prob

    def __call__(self, cuts: CutSet) -> CutSet:
        return (
            cuts
                .sort_by_duration(ascending=False)
                .mix(self.cuts, duration=cuts[0].duration, snr=self.snr, mix_prob=self.prob)
        )
