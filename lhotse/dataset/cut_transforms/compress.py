import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

from lhotse import CutSet
from lhotse.augmentation.compress import Codec
from lhotse.dataset.dataloading import resolve_seed


@dataclass
class Compress:
    """
    Applies a lossy compression algorithm filter to each Cut in a CutSet. The audio is decompressed back to raw waveforms.

    The compression is applied with a probability of ``p``. The codec is
    randomly selected the list of provided codecs,
    with optional weights controlling the selection.
    If compression level is provided as an interval,
    then the actual value is sampled uniformly from the provided interval.

    :param codecs: A list of codecs (supported: "opus", "mp3", "vorbis", "gsm")
    :param compression_level: A single value or an interval. 0.0 = lowest compression (highest bitrate), 1.0 = highest compression (lowest bitrate). If an interval is provided, the value is sampled uniformly.
    :param codec_weights: Optional weights for each codec (default: equal weights).
    :param p: The probability of applying the low-pass filter (default: 0.5).
    :param randgen: An optional random number generator (default: a new instance).
    :param preserve_id: Whether to preserve the original cut ID (default: False).
    """

    codecs: List[Codec]
    compression_level: Union[float, Tuple[float, float]] = 0.9
    codec_weights: Optional[List[float]] = None
    compress_custom_fields: bool = False
    p: float = 0.5
    seed: Union[int, Literal["trng", "randomized"]] = 42
    rng: Optional[random.Random] = None
    preserve_id: bool = False

    def __post_init__(self) -> None:
        assert sorted(self.codecs) == sorted(list(set(self.codecs))), "duplicate codecs"

        if isinstance(self.compression_level, (Tuple, List)):
            assert (
                len(self.compression_level) == 2
            ), f"Expected compression_level to be a tuple or a list with two values, got {self.compression_level}"
            min_compression, max_compression = self.compression_level
            assert (
                min_compression < max_compression
            ), f"Expected min_compression < max_compression, got {min_compression} >= {max_compression}"

        assert 0 <= self.p <= 1, f"Probability p must be between 0 and 1, got {self.p}"

        if self.codec_weights:
            assert len(self.codec_weights) == len(
                self.codecs
            ), f"Expected codec_weights to be a list with the same length as codecs, got len({self.codec_weights}) != len({self.codecs})"
            assert all(
                w >= 0 for w in self.codec_weights
            ), "All codec weights must be non-negative"
        else:
            # all codecs have equal weights by default
            self.codec_weights = [1.0 for _ in self.codecs]

        if self.rng is not None and self.seed is not None:
            raise ValueError("Either rng or seed must be provided, not both")
        if self.rng is None:
            self.rng = random.Random(resolve_seed(self.seed))

    def __call__(self, cuts: CutSet) -> CutSet:
        compressed_cuts = []
        for cut in cuts:
            if self.rng.random() <= self.p:
                if isinstance(self.compression_level, (Tuple, List)):
                    min_compression, max_compression = self.compression_level
                    compression_level = (
                        self.rng.random() * (max_compression - min_compression)
                        + min_compression
                    )
                else:
                    compression_level = self.compression_level

                codec, *_ = self.rng.choices(self.codecs, weights=self.codec_weights)
                new_cut = cut.compress(
                    codec=codec,
                    compression_level=compression_level,
                    compress_custom_fields=self.compress_custom_fields,
                )
                if not self.preserve_id:
                    new_cut.id = f"{new_cut.id}_{codec}_{compression_level:.2f}"
                compressed_cuts.append(new_cut)
            else:
                compressed_cuts.append(cut)

        return CutSet(compressed_cuts)
