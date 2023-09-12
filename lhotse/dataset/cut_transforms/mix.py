import warnings
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
        cuts: CutSet,
        snr: Optional[Union[Decibels, Tuple[Decibels, Decibels]]] = (10, 20),
        p: float = 0.5,
        pad_to_longest: bool = True,
        preserve_id: bool = False,
        seed: int = 42,
        random_mix_offset: bool = False,
    ) -> None:
        """
        CutMix's constructor.

        :param cuts: a ``CutSet`` containing augmentation data, e.g. noise, music, babble.
        :param snr: either a float, a pair (range) of floats, or ``None``.
            It determines the SNR of the speech signal vs the noise signal that's mixed into it.
            When a range is specified, we will uniformly sample SNR in that range.
            When it's ``None``, the noise will be mixed as-is -- i.e. without any level adjustment.
            Note that it's different from ``snr=0``, which will adjust the noise level so that the SNR is 0.
        :param pad_to_longest: when `True`, each processed :class:`CutSet` will be padded with noise
            to match the duration of the longest Cut in a batch.
        :param preserve_id: When ``True``, preserves the IDs the cuts had before augmentation.
            Otherwise, new random IDs are generated for the augmented cuts (default).
        :param random_mix_offset: an optional bool.
            When ``True`` and the duration of the to be mixed in cut in longer than the original cut,
             select a random sub-region from the to be mixed in cut.
        """
        self.cuts = cuts
        if len(self.cuts) == 0:
            warnings.warn(
                "Empty CutSet in CutMix transform: it'll act as an identity transform."
            )
        self.snr = snr
        self.p = p
        self.pad_to_longest = pad_to_longest
        self.preserve_id = preserve_id
        self.seed = seed
        self.random_mix_offset = random_mix_offset

    def __call__(self, cuts: CutSet) -> CutSet:

        # Dummy transform - return
        if len(self.cuts) == 0:
            return cuts

        maybe_max_duration = (
            max(c.duration for c in cuts) if self.pad_to_longest else None
        )
        return cuts.mix(
            cuts=self.cuts,
            duration=maybe_max_duration,
            snr=self.snr,
            mix_prob=self.p,
            preserve_id="left" if self.preserve_id else None,
            seed=self.seed,
            random_mix_offset=self.random_mix_offset,
        )
