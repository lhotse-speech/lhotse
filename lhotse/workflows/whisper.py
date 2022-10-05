import logging
from typing import Any, Generator, List, Optional, Union

import torch

from lhotse import CutSet, MonoCut, Recording, RecordingSet, SupervisionSegment
from lhotse.qa import trim_supervisions_to_recordings
from lhotse.utils import fastcopy, is_module_available


def annotate_with_whisper(
    manifest: Union[RecordingSet, CutSet],
    language: Optional[str] = None,
    model_name: str = "base",
    device: str = "cpu",
) -> Generator[MonoCut, None, None]:
    """
    Use OpenAI Whisper model to annotate either RECORDINGS_MANIFEST, RECORDINGS_DIR, or CUTS_MANIFEST.
    It will perform automatic segmentation, transcription, and language identification. If
    the first argument is a CutSet, it will overwrite the supervisions with the results of the inference.

    Note: this is an experimental feature of Lhotse, and is not guaranteed to yield
    high quality of data.

    See the original repo for more details: https://github.com/openai/whisper

    :param manifest: a ``RecordingSet`` or ``CutSet`` object.
    :param language: specify the language if known upfront, otherwise it will be auto-detected.
    :param model_name: one of available Whisper variants (base, medium, large, etc.).
    :param device: Where to run the inference (cpu, cuda, etc.).
    :return: a generator of cuts (use ``CutSet.open_writer()`` to write them).
    """
    assert is_module_available("whisper"), (
        "This function expects OpenAI Whisper to be installed. "
        "You can install it via 'pip install git+https://github.com/openai/whisper.git' "
        "(see https://github.com/openai/whisper for details)."
    )

    if isinstance(manifest, RecordingSet):
        yield from _annotate_recordings(manifest, language, model_name, device)
    elif isinstance(manifest, CutSet):
        yield from _annotate_cuts(manifest, language, model_name, device)
    else:
        raise ValueError("The ``manifest`` must be either a RecordingSet or a CutSet.")


def _annotate_recordings(
    recordings: RecordingSet, language: str, model_name: str, device: str
):
    """
    Helper function that annotates a RecordingSet with Whisper.
    """
    import whisper

    model = whisper.load_model(model_name, device=device)

    for recording in recordings:
        if recording.num_channels > 1:
            logging.warning(
                f"Skipping recording '{recording.id}'. It has {recording.num_channels} channels, "
                f"but we currently only support mono input."
            )
            continue
        audio = torch.from_numpy(recording.resample(16000).load_audio()).squeeze(0)
        result = whisper.transcribe(model=model, audio=audio, language=language)
        supervisions = [
            SupervisionSegment(
                id=f"{recording.id}-{segment['id']:06d}",
                recording_id=recording.id,
                start=round(segment["start"], ndigits=8),
                duration=round(segment["end"], ndigits=8),
                text=segment["text"].strip(),
                language=result["language"],
            )
            for segment in result["segments"]
        ]
        cut = recording.to_cut()
        if supervisions:
            supervisions = _postprocess_timestamps(supervisions)
            cut.supervisions = list(
                trim_supervisions_to_recordings(
                    recordings=recording, supervisions=supervisions, verbose=False
                )
            )
        yield cut


def _annotate_cuts(cuts: CutSet, language: str, model_name: str, device: str):
    """
    Helper function that annotates a CutSet with Whisper.
    """
    import whisper

    model = whisper.load_model(model_name, device=device)

    for cut in cuts:
        if cut.num_channels > 1:
            logging.warning(
                f"Skipping cut '{cut.id}'. It has {cut.num_channels} channels, "
                f"but we currently only support mono input."
            )
            continue
        audio = torch.from_numpy(cut.resample(16000).load_audio()).squeeze(0)
        result = whisper.transcribe(model=model, audio=audio, language=language)
        supervisions = [
            SupervisionSegment(
                id=f"{cut.id}-{segment['id']:06d}",
                recording_id=cut.recording_id,
                start=round(segment["start"], ndigits=8),
                duration=round(segment["end"], ndigits=8),
                text=segment["text"].strip(),
                language=result["language"],
            )
            for segment in result["segments"]
        ]
        new_cut = fastcopy(cut, supervisions=supervisions).merge_supervisions()
        yield new_cut


def _postprocess_timestamps(supervisions: List[SupervisionSegment]):
    """
    Whisper tends to have a lot of overlapping segments due to inaccurate end timestamps.
    Under a strong assumption that the input speech is non-overlapping, we can fix that
    by always truncating to the start timestamp of the next segment.
    """
    from cytoolz import sliding_window

    if len(supervisions) < 2:
        return supervisions
    out = []
    for cur, nxt in sliding_window(2, supervisions):
        if cur.end > nxt.start:
            cur = cur.trim(end=nxt.start)
        out.append(cur)
    out.append(nxt)
    return out
