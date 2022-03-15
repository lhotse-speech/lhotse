"""
The following description is taken from the official website: 
https://www.robots.ox.ac.uk/~vgg/data/voxceleb/

VoxCeleb is an audio-visual dataset consisting of short clips of human speech, extracted 
from interview videos uploaded to YouTube. VoxCeleb contains speech from speakers spanning 
a wide range of different ethnicities, accents, professions and ages. There are a total of
7000+ speakers and 1 million utterances.

All speaking face-tracks are captured "in the wild", with background chatter, laughter, 
overlapping speech, pose variation and different lighting conditions. VoxCeleb consists 
of both audio and video, comprising over 2000 hours of speech. Each segment is at least 
3 seconds long.

The dataset consists of two versions, VoxCeleb1 and VoxCeleb2. Each version has it's own 
train/test split. For each version, the YouTube URLs, face detections and tracks, audio files, 
cropped face videos and speaker meta-data are provided. There is no overlap between the 
two versions.

- VoxCeleb1: VoxCeleb1 contains over 100,000 utterances for 1,251 celebrities.
  http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- VoxCeleb2: VoxCeleb2 contains over a million utterances for 6,112 identities.
  http://www.robots.ox.ac.uk/~vgg/data/voxceleb2/ 

LICENSE: The VoxCeleb dataset is available to download for commercial/research purposes 
under a Creative Commons Attribution 4.0 International License. The copyright remains with 
the original owners of the video.

This Lhotse recipe prepares the VoxCeleb1 and VoxCeleb2 datasets.
"""
import logging
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from collections import defaultdict, namedtuple

from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from tqdm.auto import tqdm

from lhotse import (
    MonoCut,
    CutSet,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
)
from lhotse.utils import Pathlike, urlretrieve_progress
from lhotse.qa import validate_recordings_and_supervisions
from lhotse.manipulation import combine

VOXCELEB1_PARTS_URL = [
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip",
]

VOXCELEB2_PARTS_URL = [
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaa",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partab",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partac",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partad",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partae",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partaf",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partag",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_dev_aac_partah",
    "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip",
]

VOXCELEB1_TRIALS_URL = "http://www.openslr.org/resources/49/voxceleb1_test_v2.txt"

SpeakerMetadata = namedtuple(
    "SpeakerMetadata", ["id", "name", "gender", "nationality", "split"]
)


def download_voxceleb1(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download and unzip the VoxCeleb1 data.

    .. note:: A "connection refused" error may occur if you are downloading without a password.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: bool, if True, download the archive even if it already exists.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "vox1_dev_wav.zip"
    zip_path = target_dir / zip_name
    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        # Download the data in parts
        for url in VOXCELEB1_PARTS_URL:
            urlretrieve_progress(
                url, desc=f"Downloading VoxCeleb1 {url.split('/')[-1]}"
            )
        # Combine the parts for dev set
        with open(zip_name, "wb") as outFile:
            for file in target_dir.glob("vox1_dev_wav_part*"):
                with open(file, "rb") as inFile:
                    shutil.copyfileobj(inFile, outFile)
    logging.info(f"Unzipping dev...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    logging.info(f"Unzipping test...")
    with zipfile.ZipFile(target_dir / "vox1_test_wav.zip") as zf:
        zf.extractall(target_dir)

    return target_dir


def download_voxceleb2(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download and unzip the VoxCeleb2 data.

    .. note:: A "connection refused" error may occur if you are downloading without a password.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: bool, if True, download the archive even if it already exists.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "vox2_aac.zip"
    zip_path = target_dir / zip_name
    if zip_path.exists() and not force_download:
        logging.info(f"Skipping {zip_name} because file exists.")
    else:
        # Download the data in parts
        for url in VOXCELEB2_PARTS_URL:
            urlretrieve_progress(
                url, desc=f"Downloading VoxCeleb2 {url.split('/')[-1]}"
            )
        # Combine the parts for dev set
        with open(zip_name, "wb") as outFile:
            for file in target_dir.glob("vox2_dev_aac_part*"):
                with open(file, "rb") as inFile:
                    shutil.copyfileobj(inFile, outFile)
    logging.info(f"Unzipping dev...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    logging.info(f"Unzipping test...")
    with zipfile.ZipFile(target_dir / "vox2_test_aac.zip") as zf:
        zf.extractall(target_dir)

    return target_dir


def prepare_voxceleb(
    voxceleb1_root: Optional[Pathlike] = None,
    voxceleb2_root: Optional[Pathlike] = None,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the VoxCeleb v1 and v2 corpora.

    The manifests are created in a dict with three splits: train, dev and test, for each
    of the two versions.
    Each split contains a RecordingSet and SupervisionSet in a dict under keys 'recordings' and 'supervisions'.

    :param voxceleb1_root: Path to the VoxCeleb v1 dataset.
    :param voxceleb2_root: Path to the VoxCeleb v2 dataset.
    :param output_dir: Path to the output directory.
    :param num_jobs: Number of parallel jobs to run.
    :return: A dict with standard corpus splits ("train" and "test") containing the manifests.

    NOTE: We prepare the data using the Kaldi style split, i.e., the whole VoxCeleb2
    ("dev" and "test") and the training portion ("dev") of VoxCeleb1 are put into the
    "train" split. The "test" split contains the "test" portion of VoxCeleb1. So if
    VoxCeleb1 is not provided, no "test" split is created in the output manifests.

    Example usage:

    .. code-block:: python

        >>> from lhotse.recipes.voxceleb import prepare_voxceleb
        >>> manifests = prepare_voxceleb(voxceleb_v1_root='/path/to/voxceleb1',
        ...                               voxceleb_v2_root='/path/to/voxceleb2',
        ...                               output_dir='/path/to/output',
        ...                               num_jobs=4)

    NOTE: If VoxCeleb1 is provided, we also prepare the trials file using the list provided
    in http://www.openslr.org/resources/49/voxceleb1_test_v2.txt. This file is used in the
    Kaldi recipes for VoxCeleb speaker verification. This is prepared as 2 tuples of the form
    (CutSet, CutSet) with identical id's, one for each of positive pairs and negative pairs.
    These are stored in the dict under keys 'pos_trials' and 'neg_trials', respectively.
    For evaluation purpose, the :class:`lhotse.dataset.sampling.CutPairsSampler`
    can be used to sample from this tuple.
    """
    voxceleb1_root = Path(voxceleb1_root) if voxceleb1_root else None
    voxceleb2_root = Path(voxceleb2_root) if voxceleb2_root else None
    if not (voxceleb1_root or voxceleb2_root):
        raise ValueError("Either VoxCeleb1 or VoxCeleb2 path must be provided.")

    output_dir = Path(output_dir) if output_dir is not None else None
    manifests = defaultdict(dict)
    if voxceleb1_root:
        logging.info("Preparing VoxCeleb1...")
        manifests.update(_prepare_voxceleb_v1(voxceleb1_root, num_jobs))
        manifests.update(_prepare_voxceleb_trials(manifests["test"]))
    else:
        logging.info(
            "VoxCeleb1 not provided, no test split or trials file will be created..."
        )
    if voxceleb2_root:
        logging.info("Preparing VoxCeleb2...")
        v2_manifests = _prepare_voxceleb_v2(voxceleb2_root, num_jobs)
        if "train" in manifests:
            manifests["train"]["recordings"] = combine(
                manifests["train"]["recordings"], v2_manifests["recordings"]
            )
            manifests["train"]["supervisions"] = combine(
                manifests["train"]["supervisions"], v2_manifests["supervisions"]
            )
        else:
            manifests["train"] = v2_manifests

    for split in ("train", "test"):
        recordings = manifests[split]["recordings"]
        supervisions = manifests[split]["supervisions"]
        validate_recordings_and_supervisions(recordings, supervisions)
        if output_dir is not None:
            recordings.to_file(output_dir / f"recordings_voxceleb_{split}.jsonl.gz")
            supervisions.to_file(output_dir / f"supervisions_voxceleb_{split}.jsonl.gz")

    # Write the trials cut sets to the output directory
    if output_dir is not None:
        if "pos_trials" in manifests:
            for i, cuts in enumerate(manifests["pos_trials"]):
                cuts.to_file(output_dir / f"pos_trials_voxceleb_utt{i+1}.jsonl.gz")
        if "neg_trials" in manifests:
            for i, cuts in enumerate(manifests["neg_trials"]):
                cuts.to_file(output_dir / f"neg_trials_voxceleb_utt{i+1}.jsonl.gz")

    return manifests


def _prepare_voxceleb_v1(
    corpus_path: Pathlike,
    num_jobs: int,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the VoxCeleb1 corpus. The manifests are created in a dict with
    2 splits: train ("dev") and test.
    """
    speaker_metadata = {}
    with open(corpus_path / "vox1_meta.csv", "r") as f:
        next(f)
        for line in f:
            spkid, name, gender, nationality, split = line.strip().split("\t")
            speaker_metadata[spkid] = SpeakerMetadata(
                id=spkid, name=name, gender=gender, nationality=nationality, split=split
            )
    with ProcessPoolExecutor(num_jobs) as ex:
        recordings = []
        supervisions = []
        futures = []
        for p in (corpus_path / "wav").rglob("*.wav"):
            futures.append(ex.submit(_process_file, p, speaker_metadata))
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing VoxCeleb1",
            leave=False,
        ):
            recording, supervision = future.result()
            recordings.append(recording)
            supervisions.append(supervision)
        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
    manifests = defaultdict(dict)
    # Split into dev and test sets based on the split of the speakers.
    for split in ("dev", "test"):
        manifests[split]["supervisions"] = supervision_set.filter(
            lambda s: s.custom["split"] == split
        )
        split_ids = [s.recording_id for s in manifests[split]["supervisions"]]
        manifests[split]["recordings"] = recording_set.filter(
            lambda r: r.id in split_ids
        )
    manifests["train"] = manifests.pop("dev")
    return manifests


def _prepare_voxceleb_trials(
    manifests: Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]
) -> Dict[str, Tuple[CutSet, CutSet]]:
    """
    Prepare the trials file for the VoxCeleb1 corpus.
    """
    recordings = manifests["recordings"]
    supervisions = manifests["supervisions"]
    cuts_utt1_pos, cuts_utt2_pos, cuts_utt1_neg, cuts_utt2_neg = [], [], [], []
    urlretrieve_progress(VOXCELEB1_TRIALS_URL, filename="voxceleb_trials.txt")
    with open("voxceleb_trials.txt", "r") as f:
        for idx, line in enumerate(f):
            target, utt1, utt2 = line.strip().split(" ")
            # id10270/x6uYqmx31kE/00001.wav -> id10270-x6uYqmx31kE-00001
            utt1 = "-".join(utt1.split(".")[0].split("/"))
            utt2 = "-".join(utt2.split(".")[0].split("/"))
            if utt1 not in recordings or utt2 not in recordings:
                logging.warning(
                    f"Trial {idx} contains unknown recording: {utt1} or {utt2}"
                )
                continue
            if target == "1":
                cuts_utt1_pos.append(
                    MonoCut(
                        id=f"trial-{idx}",
                        recording=recordings[utt1],
                        start=0,
                        duration=recordings[utt1].duration,
                        supervisions=supervisions[utt1],
                        channel=0,
                    )
                )
                cuts_utt2_pos.append(
                    MonoCut(
                        id=f"trial-{idx}",
                        recording=recordings[utt2],
                        start=0,
                        duration=recordings[utt2].duration,
                        supervisions=supervisions[utt2],
                        channel=0,
                    )
                )
            else:
                cuts_utt1_neg.append(
                    MonoCut(
                        id=f"trial-{idx}",
                        recording=recordings[utt1],
                        start=0,
                        duration=recordings[utt1].duration,
                        supervisions=supervisions[utt1],
                        channel=0,
                    )
                )
                cuts_utt2_neg.append(
                    MonoCut(
                        id=f"trial-{idx}",
                        recording=recordings[utt2],
                        start=0,
                        duration=recordings[utt2].duration,
                        supervisions=supervisions[utt2],
                        channel=0,
                    )
                )
    return {
        "pos_trials": (
            CutSet.from_cuts(cuts_utt1_pos),
            CutSet.from_cuts(cuts_utt2_pos),
        ),
        "neg_trials": (
            CutSet.from_cuts(cuts_utt1_neg),
            CutSet.from_cuts(cuts_utt2_neg),
        ),
    }


def _prepare_voxceleb_v2(
    corpus_path: Pathlike,
    num_jobs: int,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the VoxCeleb2 corpus. The manifests are created the same dict
    without any splits since the whole data is used in the final "train" split.
    """
    # Read the speaker metadata.
    speaker_metadata = {}
    with open(corpus_path / "vox2_meta.csv", "r") as f:
        next(f)
        for line in f:
            spkid, _, gender, split = map(str.strip, line.split(","))
            speaker_metadata[spkid] = SpeakerMetadata(
                id=spkid, name="", gender=gender, nationality="", split=split
            )
    # Read the wav files and prepare manifests
    with ProcessPoolExecutor(num_jobs) as ex:
        recordings = []
        supervisions = []
        futures = []
        for p in (corpus_path / split).glob("*.wav"):
            futures.append(
                ex.submit(_process_file, p, speaker_metadata, type="command")
            )
        for future in tqdm(
            futures,
            total=len(futures),
            desc=f"Processing VoxCeleb2 {split} split...",
            leave=False,
        ):
            recording, supervision = future.result()
            recordings.append(recording)
            supervisions.append(supervision)
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    manifests = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }
    return manifests


def _process_file(
    file_path: Pathlike,
    speaker_metadata: Dict[str, SpeakerMetadata],
) -> Tuple[Recording, SupervisionSegment]:
    """
    Process a single wav file and return a Recording and a SupervisionSegment.
    """
    speaker_id = file_path.parent.parent.stem
    session_id = file_path.parent.stem
    uttid = file_path.stem
    recording_id = f"{speaker_id}-{session_id}-{uttid}"
    recording = Recording.from_file(file_path, recording_id=recording_id)
    supervision = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        speaker=speaker_id,
        gender=speaker_metadata[speaker_id].gender,
        start=0.0,
        duration=recording.duration,
        custom={
            "speaker_name": speaker_metadata[speaker_id].name,
            "nationality": speaker_metadata[speaker_id].nationality,
            "split": speaker_metadata[speaker_id].split,
        },
    )
    return recording, supervision
