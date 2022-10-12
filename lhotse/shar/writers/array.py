from io import BytesIO

import lilcom
import numpy as np
from typing_extensions import Literal

from lhotse.shar.writers.tar import TarWriter


class ArrayTarWriter:
    def __init__(
        self,
        pattern: str,
        shard_size: int,
        compression: Literal["numpy", "lilcom"] = "numpy",
        lilcom_tick_power: int = -5,
    ):
        self.compression = compression
        self.tar_writer = TarWriter(pattern, shard_size)
        self.lilcom_tick_power = lilcom_tick_power

    def __enter__(self):
        self.tar_writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.tar_writer.close()

    def write(self, key: str, value: np.ndarray) -> str:

        if self.compression == "lilcom":
            assert np.issubdtype(
                value.dtype, np.floating
            ), "Lilcom compression supports only floating-point arrays."
            data = lilcom.compress(value, tick_power=self.lilcom_tick_power)
            stream = BytesIO(data)
            ext = ".llc"
        else:
            stream = BytesIO()
            np.save(stream, value, allow_pickle=False)
            ext = ".npy"

        self.tar_writer.write(key + ext, stream)
