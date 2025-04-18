import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

from lhotse import CutSet
from lhotse.augmentation.compress import Codec


@dataclass
class Compress:
    """
    Applies a lossy compression algorithm filter to each Cut in a CutSet. The audio is decompressed back to raw waveforms.

    The compression is applied with a probability of ``p``. The codec is
    randomly selected the list of provided codecs,
    with optional weights controlling the selection.
    If compression level is provided as an interval,
    then the actual value is sampled uniformly from the provided interval.

    :param codecs: A list of codecs (supported: opus, mp3, vorbis)
    :param compression_level: A single value or an interval. 0.0 = lowest compression (highest bitrate), 1.0 = highest compression (lowest bitrate)
    :param codec_weights: Optional weights for each codec (default: equal weights).
    :param p: The probability of applying the low-pass filter (default: 0.5).
    :param randgen: An optional random number generator (default: a new instance).
    """

    codecs: List[Codec]
    compression_level: Union[float, Tuple[float, float]] = 0.9
    codec_weights: Optional[List[float]] = None
    compress_custom_fields: bool = False
    p: float = 0.5
    randgen: random.Random = None

    def __post_init__(self) -> None:
        assert sorted(self.codecs) == sorted(list(set(self.codecs))), "duplicate codecs"

        if isinstance(self.compression_level, (Tuple, List)):
            assert len(self.compression_level) == 2
            min_compression, max_compression = self.compression_level
            assert min_compression < max_compression

        if self.codec_weights:
            assert len(self.codec_weights) == len(self.codecs)
        else:
            # all codecs have equal weights by default
            self.codec_weights = [1.0 for _ in self.codecs]

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.randgen is None:
            self.randgen = random.Random()

        compressed_cuts = []
        for cut in cuts:
            codec, *_ = self.randgen.choices(self.codecs, weights=self.codec_weights)

            if isinstance(self.compression_level, (Tuple, List)):
                min_compression, max_compression = self.compression_level
                compression_level = (
                    self.randgen.random() * (max_compression - min_compression)
                    + min_compression
                )
            else:
                compression_level = self.compression_level

            if self.randgen.random() <= self.p:
                new_cut = cut.compress(
                    codec=codec,
                    compression_level=compression_level,
                    compress_custom_fields=self.compress_custom_fields,
                )
                new_cut.id = f"{new_cut.id}_{codec}_{compression_level:.2f}"
                compressed_cuts.append(new_cut)
            else:
                compressed_cuts.append(cut)

        return CutSet(compressed_cuts)
