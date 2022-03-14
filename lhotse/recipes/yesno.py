"""
About the yes no dataset:

This dataset was created for the Kaldi project (see kaldi.sf.net), by a
contributor who prefers to remain anonymous. The main point of the dataset
is to provide an easy and fast way to test out the Kaldi scripts for free.

The archive "waves_yesno.tar.gz" contains 60 .wav files, sampled at 8 kHz.
All were recorded by the same male speaker, in Hebrew. In each file, the
individual says 8 words; each word is either the Hebrew for "yes" or "no",
so each file is a random sequence of 8 yes-es or noes. There is no separate
transcription provided; the sequence is encoded in the filename, with 1 for
yes and 0 for no, for instance:

# tar -xvzf waves_yesno.tar.gz
waves_yesno/1_0_1_1_1_0_1_0.wav
waves_yesno/0_1_1_0_0_1_1_0.wav
...

The dataset can be downloaded from the following address:

    https://www.openslr.org/1/

"""

import logging
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, urlretrieve_progress

_DEFAULT_URL = "http://www.openslr.org/resources/1/waves_yesno.tar.gz"


def download_yesno(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    url: Optional[str] = _DEFAULT_URL,
) -> Path:
    """Download and untar the dataset.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
        The extracted files are saved to target_dir/waves_yesno/*.wav
    :param force_download: Bool, if True, download the tar file no matter
        whether it exists or not.
    :param url: str, the url to download the dataset.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = target_dir / "waves_yesno"

    tar_path = target_dir / "waves_yesno.tar.gz"

    completed_detector = extracted_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping - {completed_detector} exists.")
        return extracted_dir

    if force_download or not tar_path.is_file():
        urlretrieve_progress(
            f"{url}", filename=tar_path, desc=f"Downloading waves_yesno.tar.gz"
        )

    shutil.rmtree(extracted_dir, ignore_errors=True)

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)

    completed_detector.touch()

    return extracted_dir


def _prepare_dataset(
    dataset: List[Pathlike],
) -> Tuple[List[Recording], List[SupervisionSegment]]:
    """Build a list of Recording and SupervisionSegment from a list
    of sound filenames.

    :param dataset: List[Pathlike], a list of sound filenames
    :return: a tuple containing a list of Recording and a list
        of SupervisionSegment
    """
    word_map = {"0": "NO", "1": "YES"}

    recordings = []
    supervisions = []
    for audio_path in dataset:
        words = audio_path.stem.split("_")
        assert len(words) == 8
        assert set(words).union({"0", "1"}) == {"0", "1"}, f"words is: {words}"

        words = [word_map[w] for w in words]
        text = " ".join(words)

        recording = Recording.from_file(audio_path)
        recordings.append(recording)

        segment = SupervisionSegment(
            id=audio_path.stem,
            recording_id=audio_path.stem,
            start=0.0,
            duration=recording.duration,
            channel=0,
            language="Hebrew",
            text=text,
        )
        supervisions.append(segment)

    return recordings, supervisions


def prepare_yesno(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply
    read and return them.

    :param corpus_dir: Pathlike, the path of the data dir. It's expected to
        contain wave files with the pattern x_x_x_x_x_x_x_x.wav, where there
        are 8 x's and each x is either 1 or 0.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is either "train" or "test", and the value is
        Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    wave_files = list(corpus_dir.glob("*.wav"))
    assert len(wave_files) == 60

    wave_files.sort()
    train_set = wave_files[::2]
    test_set = wave_files[1::2]

    assert len(train_set) == 30
    assert len(test_set) == 30

    manifests = defaultdict(dict)
    for name, dataset in zip(["train", "test"], [train_set, test_set]):
        recordings, supervisions = _prepare_dataset(dataset)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_json(output_dir / f"supervisions_{name}.json")
            recording_set.to_json(output_dir / f"recordings_{name}.json")

        manifests[name] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests
