"""
The CMU_INDIC databases were constructed at the Language Technologies Institute at Carnegie Mellon University as phonetically balanced, single speaker databases designed for corpus based speech synthesis research. They are covering major languages spoken in the Indian subcontinet.
The distributions include the raw waveform files, with transcriptions in the language's native script (etc/txt.done.data file), and also complete built synthesis voices from these databases using CMU Clustergen statistical parameteric speech synthesizer.

Complete android voices for CMU Flite are voice built from these databases are available in the Google Play store. You can hear voices built from these databases here

CMU INDIC Databases

All 13 voices are available from packed
do_indic a script to download and build a full voice from these databases (assuming FestVox build tools are all installed.
These packed versions contain only the waveform files, and the txt.done.data file.
Acknowledgements

These datasets were collected and developed with help from Hear2Read. We acknowledge their contributions to making these practical languages for festvox. Special Thanks for to Suresh Bazaj.

Source: http://festvox.org/cmu_indic/
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

BASE_URL = "http://festvox.org/h2r_indic/"

SPEAKERS = (
    "ben_rm",
    "guj_ad",
    "guj_dp",
    "guj_kt",
    "hin_ab",
    "kan_plv",
    "mar_aup",
    "mar_slp",
    "pan_amp",
    "tam_sdr",
    "tel_kpn",
    "tel_sk",
    "tel_ss",
)

# Note: some genders and accents are missing, I filled in the metadata that
#       was easily available for now.
GENDER_MAP = {
    "tel_kpn": "F",
    "hin_ab": "F",
    "kan_plv": "F",
    "ben_rm": "F",
    "guj_ad": "M",
    "mar_slp": "F",
    "guj_dp": "F",
    "tam_sdr": "F",
    "guj_kt": "F",
    "pan_amp": "F",
    "tel_ss": "F",
    "tel_sk": "M",
    "mar_aup": "M",
}

LANGUAGE_MAP = {
    "ben": "Bengali",
    "guj": "Gujarati",
    "kan": "Kannada",
    "hin": "Hindi",
    "mar": "Marathi",
    "pan": "Punjabi",
    "tam": "Tamil",
    "tel": "Telugu",
}


def download_cmu_indic(
    target_dir: Pathlike = ".",
    speakers: Sequence[str] = SPEAKERS,
    force_download: Optional[bool] = False,
    base_url: Optional[str] = BASE_URL,
) -> Path:
    """
    Download and untar the CMU Indic dataset.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param speakers: a list of speakers to download. By default, downloads all.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of CMU Arctic download site.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for spk in tqdm(speakers, desc="Downloading/unpacking CMU Indic speakers"):
        name = f"cmu_indic_{spk}"
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


def prepare_cmu_indic(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepares and returns the CMU Indic manifests,
    which consist of Recordings and Supervisions.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a dict of {'recordings': ..., 'supervisions': ...}
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    recordings = RecordingSet.from_recordings(
        # Example ID: cmu_indic_ben_rm_bn_00001
        Recording.from_file(
            wav, recording_id=f"{_get_speaker(wav.parent.parent.name)}-{wav.stem}"
        )
        for wav in corpus_dir.rglob("*.wav")
    )
    supervisions = []
    for path in corpus_dir.rglob("txt.done.data"):
        lines = path.read_text().splitlines()
        speaker = _get_speaker(path.parent.parent.name)
        lang_code = speaker.split("_")[0]  # example: 'ben_rm' -> 'ben' (Bengali)
        try:
            # Example contents of voice.feats file:
            #   variant guj
            #   age 28
            #   gender male
            #   description Built with build_cg_rfs_voice, 3 rf and 3 dur
            #   gujarati_data h2r_prompts
            #   prompt_dur 59.27min
            age = int(
                (path.parent / "voice.feats")
                .read_text()
                .splitlines()[1]
                .replace("age ", "")
                .strip()
            )
        except:
            age = None
        for l in lines:
            l = l[2:-2]  # get rid of parentheses and whitespaces on the edges
            seg_id, text = l.split(maxsplit=1)
            seg_id = f"{speaker}-{seg_id}"
            language = LANGUAGE_MAP[lang_code]
            is_english = "arctic" in seg_id

            # Determine available custom meta-data to attach.
            custom = None
            if is_english or age is not None:
                custom = {}
                if is_english:
                    custom["accent"] = language
                if age is not None:
                    custom["age"] = age

            supervisions.append(
                SupervisionSegment(
                    id=seg_id,
                    recording_id=seg_id,
                    start=0,
                    duration=recordings[seg_id].duration,
                    text=text.replace('"', ""),  # get rid of quotation marks,
                    language="English" if is_english else language,
                    speaker=speaker,
                    gender=GENDER_MAP.get(speaker),
                    custom=custom,
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
        recordings.to_json(output_dir / "cmu_indic_recordings.json")
        supervisions.to_json(output_dir / "cmu_indic_supervisions.json")

    return {"recordings": recordings, "supervisions": supervisions}


def _get_speaker(dirname: str) -> str:
    return dirname.split("_", maxsplit=2)[2]
