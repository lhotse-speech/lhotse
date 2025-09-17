import random
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

from lhotse import CutSet
from lhotse.dataset.dataloading import resolve_seed


@dataclass
class ClippingTransform:
    """
    Applies clipping to each Cut in a CutSet with a given probability.

    The clipping is applied with a probability of ``p``. The gain_db is
    randomly sampled from the provided range if an interval is given,
    or the fixed value is used if a single float is provided.

    :param gain_db: A single value or an interval (tuple/list with two values).
        The amount of gain in decibels to apply before clipping.
        If an interval is provided, the value is sampled uniformly.
    :param hard: If True, apply hard clipping (sharp cutoff); otherwise, apply soft clipping (saturation).
    :param normalize: If True, normalize the input signal to 0 dBFS before applying clipping.
    :param p: The probability of applying clipping (default: 0.5).
    :param seed: Random seed for reproducibility (default: 42).
    :param rng: Optional random number generator (overrides seed if provided).
    :param oversampling: Optional integer factor for oversampling before clipping.
    :param preserve_id: Whether to preserve the original cut ID (default: False).
    """

    gain_db: Union[float, Tuple[float, float]]
    normalize: bool = True
    p: float = 0.5
    p_hard: float = 0.5
    seed: Union[int, Literal["trng", "randomized"]] = 42
    rng: Optional[random.Random] = None
    oversampling: Optional[int] = 2
    preserve_id: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.gain_db, (tuple, list)):
            assert (
                len(self.gain_db) == 2
            ), f"Expected gain_db to be a tuple or a list with two values, got {self.gain_db}"
            min_gain, max_gain = self.gain_db
            assert (
                min_gain < max_gain
            ), f"Expected min_gain < max_gain, got {min_gain} >= {max_gain}"

        assert 0 <= self.p <= 1, f"Probability p must be between 0 and 1, got {self.p}"

        if self.rng is not None and self.seed is not None:
            raise ValueError("Either rng or seed must be provided, not both")
        if self.rng is None:
            self.rng = random.Random(resolve_seed(self.seed))

    def __call__(self, cuts: CutSet) -> CutSet:
        saturated_cuts = []
        for cut in cuts:
            if self.rng.random() <= self.p:
                if self.rng.random() <= self.p_hard:
                    hard = True
                else:
                    hard = False

                if isinstance(self.gain_db, (tuple, list)):
                    min_gain, max_gain = self.gain_db
                    gain_db = self.rng.uniform(min_gain, max_gain)
                else:
                    gain_db = self.gain_db

                new_cut = cut.clip_amplitude(
                    hard=hard,
                    gain_db=gain_db,
                    normalize=self.normalize,
                    affix_id=not self.preserve_id,
                    oversampling=self.oversampling,
                )
                saturated_cuts.append(new_cut)
            else:
                saturated_cuts.append(cut)

        return CutSet(saturated_cuts)
