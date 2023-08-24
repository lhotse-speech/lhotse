"""
The AliMeeting Mandarin corpus, originally designed for ICASSP 2022 Multi-channel
Multi-party Meeting Transcription Challenge (M2MeT), is recorded from real meetings,
including far-field speech collected by an 8-channel microphone array as well as
near-field speech collected by each participants' headset microphone. The dataset
contains 118.75 hours of speech data in total, divided into 104.75 hours for training
(Train), 4 hours for evaluation (Eval) and 10 hours as test set (Test), according to
M2MeT challenge arrangement. Specifically, the Train, Eval and Test sets contain 212,
8 and 20 meeting sessions respectively, and each session consists of a 15 to 30-minute
discussion by 2-4 participants. AliMeeting covers a variety of aspects in real-world
meetings, including diverse meeting rooms, various number of meeting participants and
different speaker overlap ratios. High-quality transcriptions are provided as well.
The dataset can be used for tasks in meeting rich transcriptions, including speaker
diarization and multi-speaker automatic speech recognition.

More details and download link: https://openslr.org/119/
"""

import logging
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.recipes.utils import normalize_text_alimeeting
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, resumable_download, safe_extract


def download_ali_meeting(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    base_url: Optional[
        str
    ] = "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/",
) -> Path:
    """
    Downdload and untar the dataset
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to downloaded and extracted directory with data.
    """
    url = f"{base_url}/AliMeeting/openlr"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_tar_names = [
        "Train_Ali_far.tar.gz",
        "Train_Ali_near.tar.gz",
        "Eval_Ali.tar.gz",
        "Test_Ali.tar.gz",
    ]
    for tar_name in dataset_tar_names:
        tar_path = target_dir / tar_name
        resumable_download(
            f"{url}/{tar_name}", filename=tar_path, force_download=force_download
        )
        with tarfile.open(tar_path) as tar:
            safe_extract(tar, path=target_dir)

    return target_dir


def prepare_ali_meeting(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    mic: Optional[str] = "far",
    normalize_text: str = "none",
    save_mono: bool = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str, "near" or "far", specifies whether to prepare the near-field or far-field data. May
        also specify "ihm", "sdm", "mdm" (similar to AMI recipe), where "ihm" and "mdm" are the same as "near"
        and "far" respectively, and "sdm" is the same as "far" with a single channel.
    :param normalize_text: str, the text normalization type. Available options: "none", "m2met".
    :param save_mono: bool, if True, save the mono recordings for sdm mic. This can speed up
        feature extraction since all channels will not be loaded.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    if not is_module_available("textgrid"):
        raise ValueError(
            "To prepare AliMeeting data, please 'pip install textgrid' first."
        )
    import textgrid

    if save_mono and mic != "sdm":
        logging.warning(
            "save_mono is True, but mic is not 'sdm'. Ignoring save_mono option."
        )
        save_mono = False

    if save_mono and not output_dir:
        raise ValueError(
            "save_mono is True, but output_dir is not specified. "
            "Please specify output_dir to save the mono recordings."
        )

    mic_orig = mic
    mic = "near" if mic_orig in ["ihm", "near"] else "far"

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    manifests = defaultdict(dict)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for part in ["Train", "Eval", "Test"]:
        if save_mono:
            output_dir_mono = output_dir / "alimeeting_sdm" / part
            output_dir_mono.mkdir(parents=True, exist_ok=True)

        recordings = []
        supervisions = []
        # Eval and Test may further be inside another folder (since the "far" and "near" are grouped together)
        corpus_dir_split = corpus_dir
        if part == "Eval" or part == "Test":
            corpus_dir_split = (
                corpus_dir / f"{part}_Ali"
                if (corpus_dir / f"{part}_Ali").is_dir()
                else corpus_dir
            )
        wav_paths = corpus_dir_split / f"{part}_Ali_{mic}" / "audio_dir"
        text_paths = corpus_dir_split / f"{part}_Ali_{mic}" / "textgrid_dir"

        # For 'near' setting:
        #  - wav files have names like R0003_M0046_F_SPK0093.wav
        #  - textgrid files have names like R0003_M0046_F_SPK0093.TextGrid
        # Speaker ID information is present in the file name itself

        # For 'far' setting:
        #  - wav files have names like R0015_M0151_MS002.wav
        #  - textgrid files have names like R0015_M015.TextGrid
        # Speaker ID information is present inside the TextGrid file

        for text_path in tqdm(
            list(text_paths.rglob("*.TextGrid")), desc=f"Preparing {part}"
        ):
            session_id = text_path.stem

            if mic == "near":
                _, _, gender, spk_id = session_id.split("_")
                spk_id = spk_id[3:]  # SPK1953 -> 1953

            try:
                tg = textgrid.TextGrid.fromFile(str(text_path))
            except ValueError:
                logging.warning(
                    f"{session_id} has annotation issues. Skipping this recording."
                )
                continue

            wav_path = list(wav_paths.rglob(f"{session_id}*.wav"))[0]

            if save_mono:
                # use sox to extract first channel of the wav file
                wav_path_mono = output_dir_mono / wav_path.name
                if not wav_path_mono.is_file():
                    cmd = f"sox {wav_path} -c 1 {wav_path_mono}"
                    subprocess.run(cmd, shell=True, check=True)
                recording = Recording.from_file(wav_path_mono, recording_id=session_id)
            else:
                recording = Recording.from_file(wav_path, recording_id=session_id)

            recordings.append(recording)

            for tier in tg.tiers:
                if mic == "far":
                    parts = tier.name.split("_")
                    if len(parts) == 4:
                        _, _, gender, spk_id = parts
                    elif len(parts) == 2:
                        gender, spk_id = parts
                    spk_id = spk_id[3:]  # SPK1953 -> 1953

                for i, interval in enumerate(tier.intervals):
                    if interval.mark != "":
                        start = interval.minTime
                        end = interval.maxTime
                        text = interval.mark
                        segment = SupervisionSegment(
                            id=f"{session_id}-{spk_id}-{i}",
                            recording_id=recording.id,
                            start=start,
                            duration=round(end - start, 4),
                            channel=0
                            if mic_orig in ["near", "ihm", "sdm"]
                            else list(range(8)),
                            language="Chinese",
                            speaker=spk_id,
                            gender=gender,
                            text=normalize_text_alimeeting(
                                text.strip(), normalize=normalize_text
                            ),
                        )
                        supervisions.append(segment)

        # Fix manifests
        recording_set, supervision_set = fix_manifests(
            RecordingSet.from_recordings(recordings),
            SupervisionSet.from_segments(supervisions),
        )
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir
                / f"alimeeting-{mic_orig}_supervisions_{part.lower()}.jsonl.gz"
            )
            recording_set.to_file(
                output_dir / f"alimeeting-{mic_orig}_recordings_{part.lower()}.jsonl.gz"
            )

        manifests[part.lower()] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests
