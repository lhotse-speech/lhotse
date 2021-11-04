"""
About the Callhome American English

    CALLHOME American English Speech was developed by the Linguistic Data
    Consortium (LDC) and consists of 120 unscripted 30-minute telephone
    conversations between native speakers of English.

    All calls originated in North America; 90 of the 120 calls were placed
    to various locations outisde of North America, while the remaining 30 calls
    were made within North America. Most participants called family members or
    close friends.

    This script support setup of two different tasks -- either ASR or SRE
    For ASR, the following LDC corpora are relevant
      Speech : LDC97S42
      Transcripts : LDC97T14
      Lexicon : LDC97L20 (not actually used)

    For SRE,  this script prepares data for speaker diarization on a portion
    of CALLHOME used in the 2000 NIST speaker recognition evaluation.
    The 2000 NIST SRE data is required. LDC catalog number LDC2001S97.
"""

import tarfile
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.utils import Pathlike, check_and_rglob, urlretrieve_progress


def prepare_callhome_english(
    audio_dir: Pathlike,
    rttm_dir: Optional[Pathlike] = None,
    transcript_dir: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
    absolute_paths: bool = False,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for the CallHome American English corpus.
    We create two manifests: one with recordings, and the other one with text
    supervisions.

    Depending on the value of transcript_dir, will prepare either
        * data for ASR task (expected LDC corpora ``LDC97S42`` and ``LDC97T14``)
        * or the SRE task (expected corpus ``LDC2001S97``)

    :param audio_dir: Path to ``LDC97S42``or ``LDC2001S97`` content
    :param transcript_dir: Path to the ``LDC97T14`` content
    :param rttm_dir: Path to the transcripts directory. If not provided,
        the transcripts will be downloaded.
    :param absolute_paths: Whether to return absolute or relative
        (to the corpus dir) paths for recordings.
    :return: A dict with manifests. The keys are:
        ``{'recordings', 'supervisions'}``.
    """
    # not sure if there is possible deeper level of integration,
    # as SRE does not contain/respect the train/eval/test splits?

    if transcript_dir is not None:
        return prepare_callhome_english_asr(
            audio_dir, transcript_dir, output_dir, absolute_paths
        )
    else:
        return prepare_callhome_english_sre(
            audio_dir, rttm_dir, output_dir, absolute_paths
        )


def prepare_callhome_english_sre(
    audio_dir: Pathlike,
    rttm_dir: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
    absolute_paths: bool = False,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for the Callhome American English portion prepartion.
    We create two manifests: one with recordings, and the other one with text
    supervisions.

    :param audio_dir: Path to ``LDC2001S97`` package.
    :param rttm_dir: Path to the transcripts directory. If not provided,
        the transcripts will be downloaded.
    :param output_dir: Directory where the manifests should be written.
        Can be omitted to avoid writing.
    :param absolute_paths: Whether to return absolute or relative
        (to the corpus dir) paths for recordings.
    :return: A dict with manifests.
        The keys are: ``{'recordings', 'supervisions'}``.
    """
    if rttm_dir is None:
        rttm_dir = download_callhome_metadata()
    rttm_path = rttm_dir / "fullref.rttm"
    supervisions = read_rttm(rttm_path)

    audio_paths = check_and_rglob(audio_dir, "*.sph")
    recordings = RecordingSet.from_recordings(
        Recording.from_file(p, relative_path_depth=None if absolute_paths else 4)
        for p in tqdm(audio_paths)
    )

    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_json(output_dir / "recordings.json")
        supervisions.to_json(output_dir / "supervisions.json")
    return {"recordings": recordings, "supervisions": supervisions}


def prepare_callhome_english_asr(
    audio_dir: Pathlike,
    transcript_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    absolute_paths: bool = False,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for the CallHome American English corpus.
    We create two manifests: one with recordings, and the other one with text
    supervisions.

    :param audio_dir: Path to ``LDC97S42`` content
    :param transcript_dir: Path to the ``LDC97T14`` content
    :param output_dir: Directory where the manifests should be written.
        Can be omitted to avoid writing.
    :param absolute_paths: Whether to return absolute or relative
        (to the corpus dir) paths for recordings.
    :return: A dict with manifests. The keys are:
        ``{'recordings', 'supervisions'}``.
    """
    audio_dir = Path(audio_dir)
    transcript_dir = Path(transcript_dir)

    manifests = {}

    for split in ["evaltest", "train", "devtest"]:
        audio_paths = check_and_rglob(
            # The LDC distribution has a typo.
            audio_dir / "data" / split.replace("evaltest", "evltest"),
            "*.sph",
        )
        recordings = RecordingSet.from_recordings(
            Recording.from_file(p, relative_path_depth=None if absolute_paths else 4)
            for p in tqdm(audio_paths)
        )

        transcript_paths = check_and_rglob(
            transcript_dir / "transcrpt" / split,
            "*.txt",
        )

        # TODO: Add text normalization like in Kaldi recipe.
        #       Not doing this right now as it's not needed for VAD/diarization...
        supervisions = []
        for p in transcript_paths:
            idx = 0
            postprocessed_lines = list()
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    continue
                try:
                    start, end, spk, text = line.split(maxsplit=3)
                    duration = float(Decimal(end) - Decimal(start))
                    if duration <= 0:
                        continue
                    postprocessed_lines.append(line)
                except InvalidOperation:
                    postprocessed_lines[-1] = postprocessed_lines[-1] + " " + line
                except ValueError:
                    postprocessed_lines[-1] = postprocessed_lines[-1] + " " + line

            for line in postprocessed_lines:
                recording_id = p.stem
                # example line:
                # 19.33 21.18 B: %ah Tayyib
                start, end, spk, text = line.split(maxsplit=3)
                spk = spk.replace(":", "")
                duration = float(Decimal(end) - Decimal(start))
                if duration <= 0:
                    continue
                start = float(start)
                supervisions.append(
                    SupervisionSegment(
                        recording_id=recording_id,
                        start=start,
                        duration=duration,
                        channel=ord(spk[0]) - ord("A"),
                        speaker=f"{recording_id}_{spk:0>2s}",
                        id=f"{recording_id}_{spk:0>2s}_{idx:0>5d}",
                        text=text,
                    )
                )
                idx += 1
        supervisions = SupervisionSet.from_segments(supervisions)

        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            recordings.to_json(output_dir / f"recordings_{split}.json")
            supervisions.to_json(output_dir / f"supervisions_{split}.json")

        manifests[split] = {"recordings": recordings, "supervisions": supervisions}

    return manifests


def download_callhome_metadata(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    url: str = "http://www.openslr.org/resources/10/sre2000-key.tar.gz",
) -> Path:
    target_dir = Path(target_dir)
    sre_dir = target_dir / "sre2000-key"
    if sre_dir.is_dir():
        return sre_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_name = "sre2000-key.tar.gz"
    tar_path = target_dir / tar_name
    if force_download or not tar_path.is_file():
        urlretrieve_progress(url, filename=tar_path, desc=f"Downloading {tar_name}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    return sre_dir


def read_rttm(path: Pathlike) -> SupervisionSet:
    lines = Path(path).read_text().splitlines()
    sups = []
    rec_cntr = Counter()
    for line in lines:
        _, recording_id, channel, start, duration, _, _, speaker, _, _ = line.split()
        start, duration, channel = float(start), float(duration), int(channel)
        if duration == 0.0:
            continue
        rec_cntr[recording_id] += 1
        sups.append(
            SupervisionSegment(
                id=f"{recording_id}_{rec_cntr[recording_id]}",
                recording_id=recording_id,
                start=start,
                duration=duration,
                channel=channel,
                speaker=f"{recording_id}_{speaker}",
                language="English",
            )
        )
    return SupervisionSet.from_segments(sups)
