"""
About the HI_MIA corpus
HI_MIA is a far-field text-dependent speaker verification data
published by Beijing Shell Shell Technology Co.,Ltd.
The contents are wake-up words "Hi, Mia" in Chinese(ni hao mi ya; 你好，米雅).
It' publicly available on https://www.openslr.org/85

The HI_MIA_CW is a supplemental database of the HI_MIA database.
The specific text of the audios is the HI-MIA confusion words in Chinese,
which are the negative samples for wake-up words "hi, Mia"(ni hao mi ya; 你好, 米雅).
It' publicly available on https://www.openslr.org/120
"""

import logging
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract

# HI_MIA contains train.tar.gz, dev.tar.gz, test.tar.gz and test_v2.tar.gz
# According to https://www.openslr.org/85,
# test.tar.gz is deprecated because of corrupted audio files.
# So here we download test_v2.tar.gz instead of test.tar.gz
# Following three files are going to be downloaded:
#   https://us.openslr.org/resources/85/train.tar.gz
#   https://us.openslr.org/resources/85/dev.tar.gz
#   https://us.openslr.org/resources/85/test_v2.tar.gz
#
# The extracted folder of test_v2.tar.gz is "test" rather than "test_v2".

# HI_MIA_CW contains data.tgz and resource.tgz
# https://us.openslr.org/resources/120/data.tgz
# will be extracted to folder "16k_wav_file"
#
# Transcriptions.
# https://us.openslr.org/resources/120/resource.tgz
# will be extracted to folder "resource"

SOURCE_FILE = {
    "train": "train.tar.gz",
    "dev": "dev.tar.gz",
    "test": "test_v2.tar.gz",
    # "cw_test" contains following two entries.
    "data": "data.tgz",
    "resource": "resource.tgz",
}

EXTRACTED_FOLDER = {
    "train": "train",
    "dev": "dev",
    "test": "test",
    # "cw_test" contains following two entries.
    "data": "16k_wav_file",
    "resource": "resource",
}

CW_PARTS = ["cw_test"]
CW_SOURCE_FILE_LIST = ["data", "resource"]
CW_FILES = ["data.tgz", "resource.tgz"]

HI_MIA_PARTS = ["train", "dev", "test"]
HI_MIA_AND_CW_PARTS = HI_MIA_PARTS + CW_PARTS


def _validate_dataset_parts(
    dataset_parts: Optional[Union[str, Sequence[str]]] = "auto",
) -> bool:
    valid_dataset_parts = HI_MIA_AND_CW_PARTS + ["auto", "himia"]

    def validate_a_dataset(dataset_name: str) -> bool:
        assert dataset_name in valid_dataset_parts, (
            f"{dataset_name} is not a valid subset. "
            f"You may want to select one from {valid_dataset_parts}"
        )
        return True

    if isinstance(dataset_parts, str):
        validate_a_dataset(dataset_parts)
        return True
    assert isinstance(dataset_parts, tuple)
    for dataset_name in dataset_parts:
        validate_a_dataset(dataset_name)
    return True


def download_himia(
    target_dir: Pathlike = ".",
    dataset_parts: Optional[Union[str, Sequence[str]]] = "auto",
    force_download: bool = False,
    base_url: str = "http://www.openslr.org/resources",
) -> Path:
    """
    Downdload and untar HI_MIA and HI_MIA_CW datasets.
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param dataset_parts: "auto", "himia"
        or a list of splits (e.g. "train", "dev", "test", "cw_test") to download.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to extracted directory with data.
    """
    target_dir = Path(target_dir)
    corpus_dir = target_dir / "HiMia"

    _validate_dataset_parts(dataset_parts)
    if dataset_parts == "auto":
        dataset_parts = HI_MIA_PARTS + CW_SOURCE_FILE_LIST
    elif dataset_parts == "himia":
        dataset_parts = HI_MIA_PARTS
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    files_to_download = []
    # Example: when command with `-p test -p cw_test`
    # All previous if/elif/elif branches are not executed.
    # We need following loop to expand "cw_test" to CW_SOURCE_FILE_LIST
    for dataset_name in dataset_parts:
        if dataset_name == "cw_test":
            files_to_download += CW_SOURCE_FILE_LIST
        else:
            files_to_download.append(dataset_name)

    tar_files = [SOURCE_FILE[part] for part in files_to_download]
    ext_folders = [EXTRACTED_FOLDER[part] for part in files_to_download]

    for tar_name, ext_name in zip(tar_files, ext_folders):
        # HI_MIA_CW is from
        # https://us.openslr.org/resources/120/
        # HI_MIA is from
        # https://us.openslr.org/resources/85/
        is_cw = True if tar_name in CW_FILES else False
        url_suffix_index = 120 if is_cw else 85
        url = f"{base_url}/{url_suffix_index}"

        tar_path = target_dir / tar_name
        completed_detector_dir = corpus_dir / ext_name
        if is_cw:
            completed_detector_dir = corpus_dir / "cw_test" / ext_name
        completed_detector = completed_detector_dir / ".completed"

        extracted_dir = (
            completed_detector_dir
            if "resource.tgz" == tar_name
            else completed_detector_dir.parent
        )

        if completed_detector.is_file():
            logging.info(
                f"Skipping download and extraction of {tar_name} because {completed_detector} exists."
            )
            continue
        resumable_download(
            f"{url}/{tar_name}", filename=tar_path, force_download=force_download
        )

        logging.info(f"Extracting {tar_name}.")
        shutil.rmtree(completed_detector_dir, ignore_errors=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tar:
            safe_extract(tar, path=extracted_dir)
        completed_detector.touch()

    return corpus_dir


_TOTAL_NUM_WAVS = {"train": 993083, "dev": 164640, "test": 165120, "cw_test": 16343}


def _prepare_train_dev_test(
    corpus_dir: Path,
    part: str,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param part: dataset part, one of ["train", "dev", "test"]
    :return: the RecodingSet and SupervisionSet given a dataset part.
    """
    logging.info(f"Processing HI_MIA subset: {part}")
    suffix_path = "" if part == "test" else "SPEECHDATA"
    scp_file_name = "wav" if part == "test" else part
    dir_of_wav_scp = corpus_dir / f"{part}/{suffix_path}/"
    wav_scp_path = dir_of_wav_scp / f"{scp_file_name}.scp"
    assert wav_scp_path.is_file(), f"{wav_scp_path}"
    recordings = []
    supervisions = []

    wav_path = "wav/" if part == "test" else ""
    with open(wav_scp_path) as wav_scp_f:
        for wav_entry in tqdm(wav_scp_f, total=_TOTAL_NUM_WAVS[part]):
            wav_entry = wav_entry.strip()
            audio_path = dir_of_wav_scp / f"{wav_path}" / f"{wav_entry}"
            audio_path = audio_path.resolve()
            audio_file_name = audio_path.stem
            # Example of audio_file_name: SV0297_2_00_F0041
            # speaker: SV0297
            # See rule of the filename at: https://aishelltech.com/wakeup_data
            speaker = audio_file_name.split("_")[0]
            # According to https://www.aishelltech.com/wakeup_data
            # The wake-up word is "你好，米雅".
            # The comma is removed to save space.
            text = "你好米雅"
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue
            recording = Recording.from_file(audio_path)
            recordings.append(recording)
            segment = SupervisionSegment(
                id=audio_file_name,
                recording_id=audio_file_name,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language="Chinese",
                speaker=speaker,
                text=text.strip(),
            )
            supervisions.append(segment)
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    return recording_set, supervision_set


def _prepare_cw_test(corpus_path: Path) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet of test dataset.
    :param corpus_dir: Pathlike, the path of the data dir.
    :return: the RecodingSet and SupervisionSet of test dataset.
    """
    logging.info("Processing HI_MIA_CW dataset")
    recordings = []
    supervisions = []
    cw_test_path = corpus_path / "cw_test/16k_wav_file"
    logging.info(f"Searching wav files in {cw_test_path}")
    transcript_path = corpus_path / "cw_test/resource/transcription.txt"
    transcript_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            # Examples of line:
            # 0001_M02_01_fast_0001.wav▸你好米
            # 0001_M02_01_fast_0002.wav▸你好你好
            # 0001_M02_01_fast_0003.wav▸你好亚

            idx_transcript = line.split()
            transcript = " ".join(idx_transcript[1:])
            transcript_dict[idx_transcript[0]] = transcript

    assert len(transcript_dict) == _TOTAL_NUM_WAVS["cw_test"]

    for wav_name in tqdm(transcript_dict, total=_TOTAL_NUM_WAVS["cw_test"]):
        audio_path = cw_test_path / wav_name
        audio_path = audio_path.resolve()
        text = transcript_dict[wav_name]
        assert audio_path.is_file(), f"{audio_path} does not exist."
        recording = Recording.from_file(audio_path)
        recordings.append(recording)
        audio_file_name = audio_path.stem
        speaker = audio_file_name.split("_")[0]
        segment = SupervisionSegment(
            id=audio_file_name,
            recording_id=audio_file_name,
            start=0.0,
            duration=recording.duration,
            channel=0,
            language="Chinese",
            speaker=speaker,
            text=text.strip(),
        )
        supervisions.append(segment)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    return recording_set, supervision_set


def prepare_himia(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: "auto", "himia"
        or a list of splits (e.g. "train", "dev", "test", "cw_test") to download.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    _validate_dataset_parts(dataset_parts)
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    manifests = defaultdict(dict)
    if dataset_parts == "auto":
        dataset_parts = HI_MIA_AND_CW_PARTS
    elif dataset_parts == "himia":
        dataset_parts = HI_MIA_PARTS
    elif isinstance(dataset_parts, str):
        if dataset_parts == "cw_test":
            dataset_parts = CW_PARTS
        else:
            dataset_parts = [dataset_parts]

    for part in tqdm(
        dataset_parts,
        desc="Process HI_MIA and HI_MIA_CW dataset.",
    ):
        if "cw_test" == part:
            recording_set, supervision_set = _prepare_cw_test(corpus_dir)
        else:
            recording_set, supervision_set = _prepare_train_dev_test(corpus_dir, part)
        if output_dir is not None:
            supervision_set.to_file(output_dir / f"himia_supervisions_{part}.jsonl.gz")
            recording_set.to_file(output_dir / f"himia_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
