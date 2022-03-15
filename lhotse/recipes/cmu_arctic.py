"""
The CMU_ARCTIC databases were constructed at the Language Technologies Institute at Carnegie Mellon University
as phonetically balanced, US English single speaker databases designed for unit selection speech synthesis research.

A detailed report on the structure and content of the database and the recording environment etc is available as a
Carnegie Mellon University, Language Technologies Institute Tech Report CMU-LTI-03-177 and is also available here:
http://www.festvox.org/cmu_arctic/cmu_arctic_report.pdf

The databases consist of around 1150 utterances carefully selected from out-of-copyright texts from Project Gutenberg.
The databses include US English male (bdl) and female (slt) speakers (both experinced voice talent) as well as
other accented speakers.

The 1132 sentence prompt list is available from cmuarctic.data:
http://www.festvox.org/cmu_arctic/cmuarctic.data

The distributions include 16KHz waveform and simultaneous EGG signals.
Full phoentically labelling was perfromed by the CMU Sphinx using the FestVox based labelling scripts.
Complete runnable Festival Voices are included with the database distributions, as examples though better voices
can be made by improving labelling etc.

Note: The Lhotse recipe is currently not downloading or using the phonetic labeling.
"""
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm.auto import tqdm

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.qa import remove_missing_recordings_and_supervisions
from lhotse.utils import Pathlike, urlretrieve_progress

BASE_URL = "http://festvox.org/cmu_arctic/packed/"

SPEAKERS = (
    "aew",
    "ahw",
    "aup",
    "awb",
    "axb",
    "bdl",
    "clb",
    "eey",
    "fem",
    "gka",
    "jmk",
    "ksp",
    "ljm",
    "lnh",
    "rms",
    "rxr",
    "slp",
    "slt",
)

# Note: some genders and accents are missing, I filled in the metadata that
#       was easily available for now.
GENDER_MAP = {
    "bdl": "male",
    "slt": "female",
    "clb": "female",
    "rms": "male",
    "jmk": "male",
    "awb": "male",
    "ksp": "male",
}

ACCENT_MAP = {
    "bdl": "US Midwest",
    "slt": "US Midwest",
    "clb": "US",
    "rms": "US",
    "jmk": "Canadian Ontario",
    "awb": "Scottish South Eastern",
    "ksp": "Indian",
}


def download_cmu_arctic(
    target_dir: Pathlike = ".",
    speakers: Sequence[str] = SPEAKERS,
    force_download: Optional[bool] = False,
    base_url: Optional[str] = BASE_URL,
) -> Path:
    """
    Download and untar the CMU Arctic dataset.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param speakers: a list of speakers to download. By default, downloads all.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of CMU Arctic download site.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for spk in tqdm(speakers, desc="Downloading/unpacking CMU Arctic speakers"):
        name = f"cmu_us_{spk}_arctic"
        tar_name = f"{name}.tar.bz2"
        full_url = f"{base_url}{tar_name}"
        tar_path = target_dir / tar_name
        part_dir = target_dir / name
        completed_detector = part_dir / ".completed"
        if completed_detector.is_file():
            logging.info(f"Skiping {spk} because {completed_detector} exists.")
            continue
        if force_download or not tar_path.is_file():
            urlretrieve_progress(
                full_url, filename=tar_path, desc=f"Downloading {tar_name}"
            )
        shutil.rmtree(part_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=target_dir)
        completed_detector.touch()

    return target_dir


def prepare_cmu_arctic(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepares and returns the CMU Arctic manifests,
    which consist of Recordings and Supervisions.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a dict of {'recordings': ..., 'supervisions': ...}
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    recordings = RecordingSet.from_recordings(
        # Example ID: cmu_us_sup_arctic-arctic_a0001
        Recording.from_file(
            wav, recording_id=f"{_get_speaker(wav.parent.parent.name)}-{wav.stem}"
        )
        for wav in corpus_dir.rglob("*.wav")
    )
    supervisions = []
    for path in corpus_dir.rglob("txt.done.data"):
        lines = path.read_text().splitlines()
        speaker = _get_speaker(path.parent.parent.name)
        for l in lines:
            l = l[2:-2]  # get rid of parentheses and whitespaces on the edges
            seg_id, text = l.split(maxsplit=1)
            seg_id = f"{speaker}-{seg_id}"
            supervisions.append(
                SupervisionSegment(
                    id=seg_id,
                    recording_id=seg_id,
                    start=0,
                    duration=recordings[seg_id].duration,
                    text=text.replace('"', ""),  # get rid of quotation marks,
                    language="English",
                    speaker=speaker,
                    gender=GENDER_MAP.get(speaker),
                    custom={"accent": ACCENT_MAP.get(speaker)},
                )
            )
    supervisions = SupervisionSet.from_segments(supervisions)

    # There seem to be 20 recordings missing; remove the before validation
    recordings, supervisions = remove_missing_recordings_and_supervisions(
        recordings, supervisions
    )
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        recordings.to_json(output_dir / "cmu_arctic_recordings.json")
        supervisions.to_json(output_dir / "cmu_arctic_supervisions.json")

    return {"recordings": recordings, "supervisions": supervisions}


def _get_speaker(dirname: str) -> str:
    return dirname.split("_")[2]
