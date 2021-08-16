"""
This script prepares data for speaker diarization on
a portion of CALLHOME used in the 2000 NIST speaker recognition evaluation.
The 2000 NIST SRE is required, and has an LDC catalog number LDC2001S97.
"""

import tarfile
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.utils import Pathlike, urlretrieve_progress, check_and_rglob


def prepare_callhome_english(
        audio_dir: Pathlike,
        rttm_dir: Optional[Pathlike] = None,
        output_dir: Optional[Pathlike] = None,
        absolute_paths: bool = False
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for the Switchboard corpus.
    We create two manifests: one with recordings, and the other one with text supervisions.
    When ``sentiment_dir`` is provided, we create another supervision manifest with sentiment annotations.

    :param audio_dir: Path to ``LDC2001S97`` package.
    :param rttm_dir: Path to the transcripts directory. If not provided, the transcripts will be downloaded.
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :param absolute_paths: Whether to return absolute or relative (to the corpus dir) paths for recordings.
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    if rttm_dir is None:
        rttm_dir = download_callhome_metadata()
    rttm_path = rttm_dir / 'fullref.rttm'
    supervisions = read_rttm(rttm_path)

    audio_paths = check_and_rglob(audio_dir, '*.sph')
    recordings = RecordingSet.from_recordings(
        Recording.from_file(p, relative_path_depth=None if absolute_paths else 4)
        for p in tqdm(audio_paths)
    )

    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_json(output_dir / 'recordings.json')
        supervisions.to_json(output_dir / 'supervisions.json')
    return {
        'recordings': recordings,
        'supervisions': supervisions
    }


def download_callhome_metadata(
        target_dir: Pathlike = '.',
        force_download: bool = False,
        url: str = "http://www.openslr.org/resources/10/sre2000-key.tar.gz"
) -> Path:
    target_dir = Path(target_dir)
    sre_dir = target_dir / 'sre2000-key'
    if sre_dir.is_dir():
        return sre_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_name = 'sre2000-key.tar.gz'
    tar_path = target_dir / tar_name
    if force_download or not tar_path.is_file():
        urlretrieve_progress(url, filename=tar_path, desc=f'Downloading {tar_name}')
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
                id=f'{recording_id}_{rec_cntr[recording_id]}',
                recording_id=recording_id,
                start=start,
                duration=duration,
                channel=channel,
                speaker=f'{recording_id}_{speaker}',
                language='English'
            )
        )
    return SupervisionSet.from_segments(sups)
