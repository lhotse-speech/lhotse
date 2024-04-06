"""
Note: this module is very heavily based on a torchaudio tutorial about forced
alignment with Wav2Vec2 created by Moto Hira.

Link: https://pytorch.org/audio/stable/pipelines.html
"""
from functools import partial
from typing import Generator

from lhotse import CutSet, MonoCut
from lhotse.parallel import ParallelExecutor

from .asr_aligner import ASRForcedAligner
from .mms_aligner import MMSForcedAligner


def __get_aligner_class(bundle_name: str):
    if bundle_name == "MMS_FA":
        return MMSForcedAligner
    elif "ASR" in bundle_name:
        return ASRForcedAligner
    else:
        raise ValueError(f"Unknown bundle name: {bundle_name}")


def align_with_torchaudio(
    cuts: CutSet,
    bundle_name: str = "WAV2VEC2_ASR_BASE_960H",
    device: str = "cpu",
    normalize_text: bool = True,
    num_jobs: int = 1,
    verbose: bool = False,
    check_language: bool = True,
) -> Generator[MonoCut, None, None]:
    """
    Use a pretrained model from torchaudio (such as Wav2Vec2) to perform forced
    word-level alignment of a CutSet.

    This means that for every SupervisionSegment with a transcript, we will find the
    start and end timestamps for each of the words.

    We support cuts with multiple supervisions -- the forced alignment will be done
    for every supervision region separately, and attached to the relevant supervision.

    Note: this is an experimental feature of Lhotse, and is not guaranteed to yield
    high quality of data.

    See torchaudio's documentation and tutorials for more details:
    - https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
    - https://pytorch.org/audio/stable/pipelines.html

    :param cuts: input CutSet.
    :param bundle_name: name of the selected pretrained model from torchaudio.
        By default, we use WAV2VEC2_ASR_BASE_960H (English only).
        In order to use a multilingual alignment model, use bundle_name='MMS_FA'.
    :param device: device on which to run the computation.
    :param normalize_text: by default, we'll try to normalize the text by making
        it uppercase and discarding symbols outside of model's character level vocabulary.
        If this causes issues, turn the option off and normalize the text yourself.
    :param num_jobs: number of parallel jobs to use for alignment.
    :param verbose: if True, display the progress bar.
    :param check_language: if False, disable warning about the non-existent language tags in supervisions.
    :return: a generator of cuts that have the "alignment" field set in each of
        their supervisions.
    """
    AlignerClass = __get_aligner_class(bundle_name)
    aligner_init_fn = partial(
        AlignerClass,
        bundle_name=bundle_name,
        device=device,
        check_language=check_language,
    )
    processor = ParallelExecutor(
        init_fn=aligner_init_fn,
        num_jobs=num_jobs,
        verbose=verbose,
        description="Aligning",
    )
    return processor(cuts, normalize=normalize_text)
