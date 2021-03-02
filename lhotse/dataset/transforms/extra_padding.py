from lhotse import CutSet
from lhotse.utils import LOG_EPSILON


class ExtraPadding:
    """
    A transform on batch of cuts (``CutSet``) that adds a number of
    extra context frames on both sides of the cut.

    It is intended mainly for training frame-synchronous ASR models
    with convolutional layers to avoid using padding inside of the
    hidden layers, by giving the model larger context in the input.

    This is best used as the first transform in the transform list
    for dataset - it will ensure that each individual cut gets extra
    context before concatenation, or that it will be filled with noise, etc.
    """

    def __init__(
            self,
            num_extra_frames: int,
            padding_value: int = LOG_EPSILON
    ) -> None:
        """
        ExtraPadding's constructor.

        :param num_extra_frames: The total number of frames to add to each cut.
            We will add half that number on each side of the cut ("both" directions padding).
        """
        self.num_extra_frames = num_extra_frames
        self.padding_value = padding_value

    def __call__(self, cuts: CutSet) -> CutSet:
        return CutSet.from_cuts(
            c.pad(
                num_frames=c.num_frames + self.num_extra_frames,
                pad_feat_value=self.padding_value,
                direction='both'
            ) for c in cuts
        )
