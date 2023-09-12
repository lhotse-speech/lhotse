import logging
from typing import Generator, List, Optional, Union

import torch

from lhotse import CutSet, MonoCut, RecordingSet, SupervisionSegment, add_durations
from lhotse.qa import trim_supervisions_to_recordings
from lhotse.utils import fastcopy, is_module_available


def annotate_with_whisper(
    manifest: Union[RecordingSet, CutSet],
    model_name: str = "base",
    device: str = "cpu",
    force_nonoverlapping: bool = False,
    download_root: Optional[str] = None,
    **decode_options,
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
    :param force_nonoverlapping: if True, the Whisper segment time-stamps will be processed to make
        sure they are non-overlapping.
    :param download_root: if specified, the model will be downloaded to this directory. Otherwise,
        it will be downloaded to the default location specfied by whisper.
    :param decode_options: additional options to pass to the ``whisper.transcribe`` function.
    :return: a generator of cuts (use ``CutSet.open_writer()`` to write them).
    """
    assert is_module_available("whisper"), (
        "This function expects OpenAI Whisper to be installed. "
        "You can install it via 'pip install git+https://github.com/openai/whisper.git' "
        "(see https://github.com/openai/whisper for details)."
    )

    if isinstance(manifest, RecordingSet):
        yield from _annotate_recordings(
            manifest,
            model_name,
            device,
            force_nonoverlapping,
            download_root,
            **decode_options,
        )
    elif isinstance(manifest, CutSet):
        yield from _annotate_cuts(
            manifest,
            model_name,
            device,
            force_nonoverlapping,
            download_root,
            **decode_options,
        )
    else:
        raise ValueError("The ``manifest`` must be either a RecordingSet or a CutSet.")


def _annotate_recordings(
    recordings: RecordingSet,
    model_name: str,
    device: str,
    force_nonoverlapping: bool,
    download_root: Optional[str] = None,
    **decode_options,
):
    """
    Helper function that annotates a RecordingSet with Whisper.
    """
    import whisper

    model = whisper.load_model(model_name, device=device, download_root=download_root)

    for recording in recordings:
        if recording.num_channels > 1:
            logging.warning(
                f"Skipping recording '{recording.id}'. It has {recording.num_channels} channels, "
                f"but we currently only support mono input."
            )
            continue
        audio = torch.from_numpy(recording.resample(16000).load_audio()).squeeze(0)
        result = whisper.transcribe(model=model, audio=audio, **decode_options)
        # Create supervisions from segments while filtering out those with negative duration.
        supervisions = [
            SupervisionSegment(
                id=f"{recording.id}-{segment['id']:06d}",
                recording_id=recording.id,
                start=round(segment["start"], ndigits=8),
                duration=add_durations(
                    segment["end"], -segment["start"], sampling_rate=16000
                ),
                text=segment["text"].strip(),
                language=result["language"],
            )
            for segment in result["segments"]
            if segment["end"] - segment["start"] > 0
        ]
        cut = recording.to_cut()
        if supervisions:
            supervisions = (
                _postprocess_timestamps(supervisions)
                if force_nonoverlapping
                else supervisions
            )
            cut.supervisions = list(
                trim_supervisions_to_recordings(
                    recordings=recording, supervisions=supervisions, verbose=False
                )
            )
        yield cut


def _annotate_cuts(
    cuts: CutSet,
    model_name: str,
    device: str,
    force_nonoverlapping: bool,
    download_root: Optional[str] = None,
    **decode_options,
):
    """
    Helper function that annotates a CutSet with Whisper.
    """
    import whisper

    model = whisper.load_model(model_name, device=device, download_root=download_root)

    for cut in cuts:
        if cut.num_channels > 1:
            logging.warning(
                f"Skipping cut '{cut.id}'. It has {cut.num_channels} channels, "
                f"but we currently only support mono input."
            )
            continue
        audio = torch.from_numpy(cut.resample(16000).load_audio()).squeeze(0)
        result = whisper.transcribe(model=model, audio=audio, **decode_options)
        # Create supervisions from segments while filtering out those with negative duration.
        supervisions = [
            SupervisionSegment(
                id=f"{cut.id}-{segment['id']:06d}",
                recording_id=cut.recording_id,
                start=round(segment["start"], ndigits=8),
                duration=add_durations(
                    min(segment["end"], cut.duration),
                    -segment["start"],
                    sampling_rate=16000,
                ),
                text=segment["text"].strip(),
                language=result["language"],
            )
            for segment in result["segments"]
            if segment["end"] - segment["start"] > 0
        ]
        new_cut = fastcopy(
            cut,
            supervisions=_postprocess_timestamps(supervisions)
            if force_nonoverlapping
            else supervisions,
        )
        yield new_cut


def _postprocess_timestamps(supervisions: List[SupervisionSegment]):
    """
    Whisper tends to have a lot of overlapping segments due to inaccurate end timestamps.
    Under a strong assumption that the input speech is non-overlapping, we can fix that
    by always truncating to the start timestamp of the next segment.
    """
    from cytoolz import sliding_window

    supervisions = sorted(supervisions, key=lambda s: s.start)

    if len(supervisions) < 2:
        return supervisions
    out = []
    for cur, nxt in sliding_window(2, supervisions):
        if cur.end > nxt.start:
            cur = cur.trim(end=nxt.start)
        out.append(cur)
    out.append(nxt)
    return out
