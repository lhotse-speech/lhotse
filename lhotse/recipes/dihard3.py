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
import logging
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob


def prepare_dihard3(
    dev_audio_dir: Pathlike,
    eval_audio_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    uem_manifest: Optional[bool] = True,
    num_jobs: Optional[int] = 1,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for the DIHARD III corpus.
    We create two manifests: one with recordings, and the other one with supervisions containing speaker id
    and timestamps.

    :param dev_audio_dir: Path to downloaded DIHARD III dev corpus (LDC2020E12), e.g.
        /data/corpora/LDC/LDC2020E12
    :param eval_audio_dir: Path to downloaded DIHARD III eval corpus (LDC2021E02), e.g.
        /data/corpora/LDC/LDC2021E02`
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param uem_manifest: If True, also return a SupervisionSet describing the UEM segments (see use in
        dataset.DiarizationDataset)
    :param num_jobs: int (default = 1), number of jobs to scan corpus directory for recordings
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    manifests = defaultdict(dict)
    for part in tqdm(["dev", "eval"], desc="Preparing DIHARD parts"):
        audio_dir = dev_audio_dir if part == "dev" else eval_audio_dir
        if audio_dir is None or not Path(audio_dir).exists():
            logging.warning(f"Nothing to be done for {part}")
            continue
        rttm_paths = list(check_and_rglob(audio_dir, "*.rttm"))
        uem_paths = list(check_and_rglob(audio_dir, "*.uem"))

        recordings = RecordingSet.from_dir(audio_dir, "*.flac", num_jobs=num_jobs)

        # Read metadata for recordings
        metadata = parse_metadata(list(check_and_rglob(audio_dir, "recordings.tbl"))[0])

        supervisions = SupervisionSet.from_segments(
            chain.from_iterable(
                make_rttm_segments(
                    rttm_path=[x for x in rttm_paths if x.stem == recording.id][0],
                    recording=recording,
                    metadata=metadata[recording.id],
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
            recordings.to_json(output_dir / f"recordings_{part}.json")
            supervisions.to_json(output_dir / f"supervisions_{part}.json")
            if uem_manifest:
                uem.to_json(output_dir / f"uem_{part}.json")
        manifests[part] = {"recordings": recordings, "supervisions": supervisions}
        if uem_manifest:
            manifests[part].update({"uem": uem})
    return manifests


def parse_metadata(metadata_path: Pathlike) -> Dict[str, Dict[str, Union[str, bool]]]:
    """
    Parses the recordings.tbl file and creates a dictionary from recording id to
    metadata containing the following keys: in_core, lang, domain, source
    """
    metadata = defaultdict(dict)
    with open(metadata_path, "r") as f:
        next(f)  # skip first line since it contains headers
        for line in f:
            reco_id, in_core, lang, domain, source, _, _, _, _ = line.strip().split()
            metadata[reco_id] = {
                "in_core": in_core == "True",
                "lang": lang,
                "domain": domain,
                "source": source,
            }
    return metadata


def make_rttm_segments(
    rttm_path: Pathlike, recording: Recording, metadata: Dict
) -> List[SupervisionSegment]:
    lines = rttm_path.read_text().splitlines()
    return [
        SupervisionSegment(
            id=f"{recording.id}-{speaker}-{int(100*float(start)):06d}-{int(100*(float(start)+float(duration))):06d}",
            recording_id=recording.id,
            start=float(start),
            duration=float(duration),
            speaker=speaker,
            language=metadata["lang"],
            custom=metadata,
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
        for _, _, start, end in map(str.split, lines)
    ]
