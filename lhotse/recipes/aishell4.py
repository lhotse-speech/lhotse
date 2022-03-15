"""
The AISHELL-4 is a sizable real-recorded Mandarin speech dataset collected by 8-channel 
circular microphone array for speech processing in conference scenarios. The dataset 
consists of 211 recorded meeting sessions, each containing 4 to 8 speakers, with a total 
length of 120 hours. This dataset aims to bridge the advanced research on multi-speaker 
processing and the practical application scenario in three aspects. With real recorded 
meetings, AISHELL-4 provides realistic acoustics and rich natural speech characteristics 
in conversation such as short pause, speech overlap, quick speaker turn, noise, etc. 
Meanwhile, the accurate transcription and speaker voice activity are provided for each 
meeting in AISHELL-4. This allows the researchers to explore different aspects in meeting 
processing, ranging from individual tasks such as speech front-end processing, speech 
recognition and speaker diarization, to multi-modality modeling and joint optimization 
of relevant tasks. We also release a PyTorch-based training and evaluation framework as 
a baseline system to promote reproducible research in this field. The baseline system 
code and generated samples are available at: https://github.com/felixfuyihui/AISHELL-4

The dataset can be downloaded from: https://openslr.org/111/

NOTE: The following recordings have annotation issues in the TextGrid files:
20200622_M_R002S07C01, 20200710_M_R002S06C01

NOTE about speaker ids: The speaker ids are assigned "locally" in the dataset, i.e., same
ids may be assigned to different speakers in different meetings. This may cause an
issue when training ASR models. To avoid this issue, we use the global speaker ids
which are assigned "globally" in the dataset, i.e., the tuple (meeting_id, local_spk_id)
is assigned a unique global_spk_id.
"""

import logging
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, urlretrieve_progress, is_module_available


def download_aishell4(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    base_url: Optional[str] = "http://www.openslr.org/resources",
) -> Path:
    """
    Downdload and untar the dataset
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to downloaded and extracted directory with data.
    """
    url = f"{base_url}/111"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_tar_names = [
        "train_L.tar.gz",
        "train_M.tar.gz",
        "train_S.tar.gz",
        "test.tar.gz",
    ]
    for tar_name in dataset_tar_names:
        tar_path = target_dir / tar_name
        if force_download or not tar_path.is_file():
            urlretrieve_progress(
                f"{url}/{tar_name}", filename=tar_path, desc=f"Downloading {tar_name}"
            )
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=target_dir)

    return target_dir


def prepare_aishell4(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    if not is_module_available("textgrid"):
        raise ValueError(
            "To prepare AISHELL-4 data, please 'pip install textgrid' first."
        )
    import textgrid

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    manifests = defaultdict(dict)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    global_spk_id = {}
    for part in ["train_L", "train_M", "train_S", "test"]:
        recordings = []
        supervisions = []
        wav_path = corpus_dir / part / "wav"
        for audio_path in wav_path.rglob("*.flac"):
            idx = audio_path.stem

            try:
                tg = textgrid.TextGrid.fromFile(
                    f"{corpus_dir}/{part}/TextGrid/{idx}.TextGrid"
                )
            except ValueError:
                logging.warning(
                    f"{idx} has annotation issues. Skipping this recording."
                )
                continue

            recording = Recording.from_file(audio_path)
            recordings.append(recording)

            for tier in tg.tiers:
                local_spk_id = tier.name
                key = (idx, local_spk_id)
                if key not in global_spk_id:
                    global_spk_id[key] = f"SPK{len(global_spk_id)+1:04d}"
                spk_id = global_spk_id[key]
                for j, interval in enumerate(tier.intervals):
                    if interval.mark != "":
                        start = interval.minTime
                        end = interval.maxTime
                        text = interval.mark
                        segment = SupervisionSegment(
                            id=f"{idx}-{spk_id}-{j}",
                            recording_id=idx,
                            start=start,
                            duration=round(end - start, 4),
                            channel=0,
                            language="Chinese",
                            speaker=spk_id,
                            text=text.strip(),
                        )
                        supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(output_dir / f"supervisions_{part}.jsonl")
            recording_set.to_file(output_dir / f"recordings_{part}.jsonl")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
