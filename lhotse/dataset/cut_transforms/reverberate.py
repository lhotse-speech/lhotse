import random

from lhotse import RecordingSet, CutSet


class ReverbWithImpulseResponse:
    """
    A transform on batch of cuts (``CutSet``) that convolves each cut with an impulse
    response with some probability :attr:`p`. The impulse response is chosen randomly from
    a specified CutSet of RIRs :attr:`rir_cuts`. If `early_only` is set to True, convolution
    is performed only with the first 50ms of the impulse response.
    """

    def __init__(
        self,
        rir_recordings: RecordingSet,
        p: float,
        normalize_output: bool = True,
        randgen: random.Random = None,
        preserve_id: bool = False,
        early_only: bool = False,
    ) -> None:
        self.rir_recordings = list(rir_recordings)
        self.p = p
        self.normalize_output = normalize_output
        self.random = randgen
        self.preserve_id = preserve_id
        self.early_only = early_only

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.random is None:
            self.random = random
        return CutSet.from_cuts(
            cut.reverb_rir(
                rir_recording=self.random.choice(self.rir_recordings),
                normalize_output=self.normalize_output,
                early_only=self.early_only,
                affix_id=not self.preserve_id,
            )
            if self.random.random() >= self.p
            else cut
            for cut in cuts
        )
