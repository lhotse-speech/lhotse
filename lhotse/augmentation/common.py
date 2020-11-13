from typing import Callable

import numpy as np

# def augment_fn(audio: np.ndarray, sampling_rate: int) -> np.ndarray
AugmentFn = Callable[[np.ndarray, int], np.ndarray]
