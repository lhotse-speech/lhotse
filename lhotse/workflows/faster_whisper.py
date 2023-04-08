import logging
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Generator, List, Optional, Union

import numpy as np

from lhotse import (
    CutSet,
    MonoCut,
    Recording,
    RecordingSet,
    SupervisionSegment,
    add_durations,
)
from lhotse.qa import trim_supervisions_to_recordings
from lhotse.supervision import AlignmentItem
from lhotse.utils import fastcopy, is_module_available


def annotate_with_faster_whisper(
    manifest: Union[RecordingSet, CutSet],
    model_name: str = "base",
    device: str = "cpu",
    force_nonoverlapping: bool = False,
    download_root: Optional[str] = None,
    compute_type: str = "default",
    num_workers: int = 1,
    vad_filter: bool = False,
    add_alignments: bool = False,
    **decode_options,
) -> Generator[MonoCut, None, None]:
    """
    Use OpenAI Whisper model via faster-whisper and CTranslate2 to annotate either
    RECORDINGS_MANIFEST, RECORDINGS_DIR, or CUTS_MANIFEST. It will perform automatic segmentation,
    transcription, and language identification. If the first argument is a CutSet, it will
    overwrite the supervisions with the results of the inference.

    Note: this is an experimental feature of Lhotse, and is not guaranteed to yield
    high quality of data.

    See the original repo for more details: https://github.com/guillaumekln/faster-whisper

    :param manifest: a ``RecordingSet`` or ``CutSet`` object.
    :param language: specify the language if known upfront, otherwise it will be auto-detected.
    :param model_name: one of available Whisper variants (base, medium, large, etc.).
    :param device: Where to run the inference (cpu, cuda, etc.).
    :param force_nonoverlapping: if True, the Whisper segment time-stamps will be processed to make
        sure they are non-overlapping.
    :param download_root: Not supported by faster-whisper. Argument kept to maintain compatibility
        with annotate_with_whisper. Faster-whisper uses
    :param compute_type: Type to use for computation.
        See https://opennmt.net/CTranslate2/quantization.html.
    :param num_workers: Increasing the number of workers can improve the global throughput at the
        cost of increased memory usage.
    :param vad_filter: If True, use faster-whisper's built-in voice activity detection (SileroVAD).
    :param add_alignments: if True, add word alignments using timestamps obtained using the cross-
        attention pattern and dynamic time warping (Note: Less accurate than forced alignment).
    :param decode_options: additional options to pass to the ``whisper.transcribe`` function.
    :return: a generator of cuts (use ``CutSet.open_writer()`` to write them).
    """
    assert is_module_available("faster_whisper"), (
        "This function expects faster-whisper to be installed. "
        "You can install it via 'pip install faster-whisper' "
        "(see https://github.com/guillaumekln/faster-whisper/ for details)."
    )
    if not isinstance(manifest, RecordingSet) and not isinstance(manifest, CutSet):
        raise ValueError("The ``manifest`` must be either a RecordingSet or a CutSet.")

    model = _initialize_model(
        model_name, device, compute_type, num_workers, download_root
    )
    with ThreadPoolExecutor(num_workers) as ex:
        futures = []
        for item in manifest:
            futures.append(
                ex.submit(
                    _process_single_manifest,
                    item,
                    model,
                    force_nonoverlapping,
                    vad_filter,
                    add_alignments,
                    **decode_options,
                )
            )
        for item in as_completed(futures):
            yield item.result()


def _initialize_model(
    model_name: str,
    device: str,
    compute_type: str = "default",
    num_workers: int = 1,
    download_root: Optional[str] = None,
):
    import torch
    from faster_whisper import WhisperModel

    if num_workers > 1:
        # Limit num_workers to available GPUs
        num_workers = min(num_workers, torch.cuda.device_count())
    device_index = list(range(num_workers))
    return WhisperModel(
        model_name,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        num_workers=num_workers,
        # download_root=download_root,
    )


def _process_single_manifest(
    manifest: Union[Recording, MonoCut],
    model,
    force_nonoverlapping: bool,
    vad_filter: bool,
    add_alignments: bool = False,
    **decode_options,
) -> MonoCut:
    if isinstance(manifest, Recording):
        if manifest.num_channels > 1:
            logging.warning(
                f"Skipping recording '{manifest.id}'. It has {manifest.num_channels} channels, "
                f"but we currently only support mono input."
            )
            return []
        recording_id = manifest.id
    else:
        recording_id = manifest.recording_id
    audio = np.squeeze(manifest.resample(16000).load_audio())
    segments, info = model.transcribe(
        audio=audio,
        word_timestamps=add_alignments,
        vad_filter=vad_filter,
        **decode_options,
    )
    # Create supervisions from segments while filtering out those with negative duration.
    if add_alignments:
        supervisions = [
            SupervisionSegment(
                id=f"{manifest.id}-{segment_id:06d}",
                recording_id=recording_id,
                start=round(segment.start, ndigits=8),
                duration=add_durations(
                    segment.end, -segment.start, sampling_rate=16000
                ),
                text=segment.text.strip(),
                language=info.language,
            ).with_alignment(
                "word",
                [
                    AlignmentItem(
                        symbol=ws.word.strip(),
                        start=round(ws.start, ndigits=8),
                        duration=round(ws.end - ws.start, ndigits=8),
                        score=round(ws.probability, ndigits=3),
                    )
                    for ws in segment.words
                ],
            )
            for segment_id, segment in enumerate(segments)
            if segment.end - segment.start > 0
        ]
    else:
        supervisions = [
            SupervisionSegment(
                id=f"{manifest.id}-{segment_id:06d}",
                recording_id=recording_id,
                start=round(segment.start, ndigits=8),
                duration=add_durations(
                    segment.end, -segment.start, sampling_rate=16000
                ),
                text=segment.text.strip(),
                language=info.language,
            )
            for segment_id, segment in enumerate(segments)
            if segment.end - segment.start > 0
        ]

    if isinstance(manifest, Recording):
        cut = manifest.to_cut()
        if supervisions:
            supervisions = (
                _postprocess_timestamps(supervisions)
                if force_nonoverlapping
                else supervisions
            )
            cut.supervisions = list(
                trim_supervisions_to_recordings(
                    recordings=manifest, supervisions=supervisions, verbose=False
                )
            )
    else:
        cut = fastcopy(
            manifest,
            supervisions=_postprocess_timestamps(supervisions)
            if force_nonoverlapping
            else supervisions,
        )

    return cut


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
