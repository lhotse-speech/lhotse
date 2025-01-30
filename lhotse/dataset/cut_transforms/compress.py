import random
from typing import List, Literal, Optional

from lhotse import CutSet, RecordingSet


class Compress:
    """
    A transform on batch of cuts (``CutSet``) that compresses each cut with a lossy codec with some probability :attr:`p`.
    """

    def __init__(
        self,
        codec: Literal["opus", "mp3", "vorbis"],
        compression_level: float,
        p: float = 0.5,
        preserve_id: bool = False,
        random=None,
    ) -> None:
        self.codec = codec
        self.compression_level = compression_level
        self.p = p
        self.preserve_id = preserve_id
        self.random = random

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.random is None:
            self.random = random.Random()
        return CutSet.from_cuts(
            cut.compress(codec=self.codec, compression_level=self.compression_level)
            if self.random.random() <= self.p
            else cut
            for cut in cuts
        )
