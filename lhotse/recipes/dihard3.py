"""
About the DIHARD III corpus

    The DIHARD III corpus consists of multi-domain data prepared to evaluate
    "hard" speaker diarization. It was used for evaluation in the Third DIHARD
    Challenge, organized by NIST and LDC in Winter 2020. It consists of monologues,
    map task dialogues, broadcast interviews, sociolinguistic interviews, meeting 
    speech, speech in restaurants, clinical recordings, and YouTube videos.
    More details can be found at:
    https://dihardchallenge.github.io/dihard3/docs/third_dihard_eval_plan_v1.2.pdf
"""
import tarfile
from itertools import chain
from pathlib import Path
from typing import Optional, List, Dict, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob, urlretrieve_progress


def prepare_dihard3(
    audio_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    uem_manifest: Optional[bool] = True,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for the DIHARD III corpus.
    We create two manifests: one with recordings, and the other one with supervisions containing speaker id
    and timestamps.

    :param audio_dir: Path to downloaded DIHARD III corpus (LDC2020E12), e.g.
        `/data/corpora/LDC/LDC2020E12_Third_DIHARD_Challenge_Development_Data`
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param uem_manifest: If True, also return a SupervisionSet describing the UEM segments (see use in
        dataset.DiarizationDataset)
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    audio_paths = check_and_rglob(audio_dir, "*.flac")
    rttm_paths = list(check_and_rglob(audio_dir, "*.rttm"))
    uem_paths = list(check_and_rglob(audio_dir, "*.uem"))

    recordings = RecordingSet.from_recordings(
        Recording.from_file(audio_path, recording_id=audio_path.stem)
        for audio_path in audio_paths
    )
    supervisions = SupervisionSet.from_segments(
        chain.from_iterable(
            make_rttm_segments(
                rttm_path=[x for x in rttm_paths if x.stem == recording.id][0],
                recording=recording,
            )
            for recording in recordings
        )
    )
    if uem_manifest:
        uem = SupervisionSet.from_segments(
            chain.from_iterable(
                make_uem_segments(
                    uem_path=[x for x in uem_paths if x.stem == recording.id][0],
                    recording=recording,
                )
                for recording in recordings
            )
        )

    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_json(output_dir / "recordings.json")
        supervisions.to_json(output_dir / "supervisions.json")
        if uem_manifest:
            uem.to_json(output_dir / "uem.json")
    manifests = {"recordings": recordings, "supervisions": supervisions}
    if uem_manifest:
        manifests.update({"uem": uem})
    return manifests


def make_rttm_segments(
    rttm_path: Pathlike,
    recording: Recording,
) -> List[SupervisionSegment]:
    lines = rttm_path.read_text().splitlines()
    return [
        SupervisionSegment(
            id=f"{recording.id}-{speaker}-{int(100*float(start)):06d}-{int(100*(float(start)+float(duration))):06d}",
            recording_id=recording.id,
            start=float(start),
            duration=float(duration),
            speaker=speaker,
        )
        for _, _, channel, start, duration, _, _, speaker, _, _ in map(str.split, lines)
    ]


def make_uem_segments(
    uem_path: Pathlike, recording: Recording
) -> List[SupervisionSegment]:
    lines = uem_path.read_text().splitlines()
    return [
        SupervisionSegment(
            id=f"{recording.id}-{int(100*float(start)):06d}-{int(100*float(end)):06d}",
            recording_id=recording.id,
            start=float(start),
            duration=round(float(end) - float(start), ndigits=8),
        )
        for _, channel, start, end in map(str.split, lines)
    ]
