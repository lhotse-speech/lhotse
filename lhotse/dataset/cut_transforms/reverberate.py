import random
from typing import List, Optional

from lhotse import CutSet, RecordingSet


class ReverbWithImpulseResponse:
    """
    A transform on batch of cuts (``CutSet``) that convolves each cut with an impulse
    response with some probability :attr:`p`.
    The impulse response is chosen randomly from a specified CutSet of RIRs :attr:`rir_cuts`.
    If no RIRs are specified, we will generate them using a fast random generator (https://arxiv.org/abs/2208.04101).
    If `early_only` is set to True, convolution is performed only with the first 50ms of the impulse response.
    """

    def __init__(
        self,
        rir_recordings: Optional[RecordingSet] = None,
        p: float = 0.5,
        normalize_output: bool = True,
        randgen: random.Random = None,
        preserve_id: bool = False,
        early_only: bool = False,
        rir_channels: List[int] = [0],
    ) -> None:
        self.rir_recordings = list(rir_recordings) if rir_recordings is not None else []
        self.p = p
        self.normalize_output = normalize_output
        self.random = randgen
        self.preserve_id = preserve_id
        self.early_only = early_only
        self.rir_channels = rir_channels

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.random is None:
            self.random = random.Random()
        return CutSet.from_cuts(
            cut.reverb_rir(
                rir_recording=self.random.choice(self.rir_recordings)
                if self.rir_recordings
                else None,
                normalize_output=self.normalize_output,
                early_only=self.early_only,
                affix_id=not self.preserve_id,
                rir_channels=self.rir_channels,
            )
            if self.random.random() <= self.p
            else cut
            for cut in cuts
        )
