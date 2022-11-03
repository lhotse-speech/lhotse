import codecs
import json
from io import BytesIO
from typing import List, Optional, Union

import lilcom
import numpy as np
from typing_extensions import Literal

from lhotse import Features
from lhotse.array import Array, TemporalArray
from lhotse.shar.utils import to_shar_placeholder
from lhotse.shar.writers.tar import TarWriter


class ArrayTarWriter:
    """
    ArrayTarWriter writes numpy arrays or PyTorch tensors into a tar archive
    that is automatically sharded.

    For floating point tensors, we support the option to use `lilcom` compression.
    Note that `lilcom` is only suitable for log-space features such as log-Mel filter banks.

    Example::

        >>> with ArrayTarWriter("some_dir/fbank.%06d.tar", shard_size=100, compression="lilcom") as w:
        ...     w.write("fbank1", fbank1_array)
        ...     w.write("fbank2", fbank2_array)  # etc.

    It would create files such as ``some_dir/fbank.000000.tar``, ``some_dir/fbank.000001.tar``, etc.

    It's also possible to use ``ArrayTarWriter`` with automatic sharding disabled::

        >>> with ArrayTarWriter("some_dir/fbank.tar", shard_size=None, compression="numpy") as w:
        ...     w.write("fbank1", fbank1_array)
        ...     w.write("fbank2", fbank2_array)  # etc.

    See also: :class:`~lhotse.shar.writers.tar.TarWriter`, :class:`~lhotse.shar.writers.audio.AudioTarWriter`
    """

    def __init__(
        self,
        pattern: str,
        shard_size: Optional[int] = 1000,
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

    @property
    def output_paths(self) -> List[str]:
        return self.tar_writer.output_paths

    def write_placeholder(self, key: str) -> None:
        self.tar_writer.write(f"{key}.nodata", BytesIO())
        self.tar_writer.write(f"{key}.nometa", BytesIO(), count=False)

    def write(
        self,
        key: str,
        value: np.ndarray,
        manifest: Union[Features, Array, TemporalArray],
    ) -> None:

        # Write binary data
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

        # Write text manifest afterwards
        manifest = to_shar_placeholder(manifest)
        json_stream = BytesIO()
        print(
            json.dumps(manifest.to_dict()),
            file=codecs.getwriter("utf-8")(json_stream),
        )
        json_stream.seek(0)
        self.tar_writer.write(f"{key}.json", json_stream, count=False)
