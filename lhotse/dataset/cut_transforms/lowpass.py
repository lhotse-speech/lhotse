import math
import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

from lhotse import CutSet
from lhotse.dataset.dataloading import resolve_seed


@dataclass
class LowpassUsingResampling:
    """
    Applies a low-pass filter to each Cut in a CutSet by resampling the audio back and forth.
    """

    p: float = 0.5
    frequencies_interval: Tuple[float, float] = (3500, 8000)
    seed: Union[int, Literal["trng", "randomized"]] = 42
    rng: Optional[random.Random] = None
    preserve_id: bool = False

    def __post_init__(self) -> None:
        if self.rng is not None and self.seed is not None:
            raise ValueError("Either rng or seed must be provided, not both")
        if self.rng is None:
            self.rng = random.Random(resolve_seed(self.seed))

    def __call__(self, cuts: CutSet) -> CutSet:
        lowpassed_cuts = []
        for cut in cuts:
            if self.rng.random() <= self.p:
                low, high = self.frequencies_interval
                if high > cut.sampling_rate // 2:
                    raise ValueError(
                        f"Upper frequency limit {high} is greater than sampling rate / 2 ({cut.sampling_rate // 2})"
                    )

                # sampling from log-uniform[low, high] distribution
                cutoff_frequency = math.exp(
                    self.rng.uniform(math.log(low), math.log(high))
                )
                cutoff_frequency = int(cutoff_frequency)

                new_cut = cut.resample(cutoff_frequency * 2).resample(cut.sampling_rate)
                if not self.preserve_id:
                    new_cut.id = f"{cut.id}_lowpassed{cutoff_frequency:.0f}"
                lowpassed_cuts.append(new_cut)
            else:
                lowpassed_cuts.append(cut)

        return CutSet(lowpassed_cuts)
