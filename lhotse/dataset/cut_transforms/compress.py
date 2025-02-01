import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

from lhotse import CutSet


@dataclass
class Compress:
    """
    For every Cut,
    1) randomly choose a codec from the codec list,
    2) take compression level (uniformly sampled from interval if specified),
    3) then apply the codec with chosen compression level with probability :attr:`p`
    4) decode back to raw waveform
    """

    codecs: List[Literal["opus", "mp3", "vorbis"]]
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
