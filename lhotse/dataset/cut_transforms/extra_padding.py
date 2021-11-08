import random
from typing import Optional

from lhotse import CutSet
from lhotse.utils import LOG_EPSILON, Seconds, exactly_one_not_null


class ExtraPadding:
    """
    A transform on batch of cuts (``CutSet``) that adds a number of
    extra context frames/samples/seconds on both sides of the cut.
    Exactly one type of duration has to specified in the constructor.

    It is intended mainly for training frame-synchronous ASR models
    with convolutional layers to avoid using padding inside of the
    hidden layers, by giving the model larger context in the input.
    Another useful application is to shift the input by a little,
    so that the data seen after frame subsampling is a bit different,
    which makes this a data augmentation technique.

    This is best used as the first transform in the transform list
    for dataset - it will ensure that each individual cut gets extra
    context before concatenation, or that it will be filled with noise, etc.
    """

    def __init__(
        self,
        extra_frames: Optional[int] = None,
        extra_samples: Optional[int] = None,
        extra_seconds: Optional[Seconds] = None,
        pad_feat_value: float = LOG_EPSILON,
        randomized: bool = False,
        preserve_id: bool = False,
    ) -> None:
        """
        ExtraPadding's constructor.

        :param extra_frames: The total number of frames to add to each cut.
            We will add half that number on each side of the cut ("both" directions padding).
        :param extra_samples: The total number of samples to add to each cut.
            We will add half that number on each side of the cut ("both" directions padding).
        :param extra_seconds: The total duration in seconds to add to each cut.
            We will add half that number on each side of the cut ("both" directions padding).
        :param pad_feat_value: When padding a cut with precomputed features, what
            value should be used for padding (the default is a very low log-energy).
        :param randomized: When ``True``, we will sample a value from a uniform distribution of
            ``[0, extra_X]`` for each cut (for samples/frames -- sample an int,
            for duration -- sample a float).
        :param preserve_id: When ``True``, preserves the IDs the cuts had before augmentation.
            Otherwise, new random IDs are generated for the augmented cuts (default).
        """
        assert exactly_one_not_null(
            extra_frames, extra_samples, extra_seconds
        ), "For ExtraPadding, you have to specify exactly one of: frames, samples, or duration."
        self.extra_frames = extra_frames
        self.extra_samples = extra_samples
        self.extra_seconds = extra_seconds
        self.pad_feat_value = pad_feat_value
        self.randomized = randomized
        self.preserve_id = preserve_id

    def __call__(self, cuts: CutSet) -> CutSet:
        if self.extra_frames is not None:
            return CutSet.from_cuts(
                c.pad(
                    num_frames=c.num_frames
                    + maybe_sample_int(value=self.extra_frames, sample=self.randomized),
                    pad_feat_value=self.pad_feat_value,
                    direction="both",
                    preserve_id=self.preserve_id,
                )
                for c in cuts
            )
        if self.extra_samples is not None:
            return CutSet.from_cuts(
                c.pad(
                    num_samples=c.num_samples
                    + maybe_sample_int(
                        value=self.extra_samples, sample=self.randomized
                    ),
                    direction="both",
                    preserve_id=self.preserve_id,
                )
                for c in cuts
            )
        if self.extra_seconds is not None:
            return CutSet.from_cuts(
                c.pad(
                    duration=c.duration
                    + maybe_sample_float(
                        value=self.extra_seconds,
                        sample=self.randomized,
                    ),
                    pad_feat_value=self.pad_feat_value,
                    direction="both",
                    preserve_id=self.preserve_id,
                )
                for c in cuts
            )
        raise ValueError(
            "Implementation error in ExtraPadding (please report this issue)."
        )


def maybe_sample_int(value: int, sample: bool) -> int:
    if sample:
        value = random.randint(0, value)
    return value


def maybe_sample_float(value: float, sample: bool) -> float:
    if sample:
        value = random.uniform(0, value)
    return value
