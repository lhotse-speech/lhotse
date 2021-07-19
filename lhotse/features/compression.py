from typing import List

import lilcom
import numpy as np


def lilcom_compress_chunked(
        data: np.ndarray, tick_power: int = -5, do_regression=True, chunk_size: int = 100
) -> List[bytes]:
    num_frames, num_feats = data.shape
    compressed = []
    for begin in range(0, num_frames, chunk_size):
        compressed.append(
            lilcom.compress(
                data[begin: begin + chunk_size],
                tick_power=tick_power,
                do_regression=do_regression,
            )
        )
    return compressed
