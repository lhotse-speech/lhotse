"""
Thchs is an open-source Chinese Speech Corpus Released by CSLT@Tsinghua University.
Publicly available on https://www.openslr.org/resources/18
THCHS-30 (26 hours)

"""


import logging
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract


def download_thchs_30(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    base_url: str = "http://www.openslr.org/resources",
) -> Path:
    """
    Downdload and untar the dataset
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to downloaded and extracted directory with data.
    """
    url = f"{base_url}/18"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "thchs"
    dataset_tar_name = "data_thchs30.tgz"
    for tar_name in [dataset_tar_name]:
        tar_path = target_dir / tar_name
        extracted_dir = corpus_dir / tar_name[:-4]
        completed_detector = extracted_dir / ".completed"
        if completed_detector.is_file():
            logging.info(
                f"Skipping download {tar_name} because {completed_detector} exists."
            )
            continue
        resumable_download(
            f"{url}/{tar_name}", filename=tar_path, force_download=force_download
        )
        shutil.rmtree(extracted_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            safe_extract(tar, path=corpus_dir)
        completed_detector.touch()

    return corpus_dir


"""
data_thchs30/data/B11_374.wav.trn
Its content is as follows:
徐 希君 肖 金生 刘 文华 屈 永利 王开 宇 骆 瑛 等 也 被 分别 判处 l = 六年 至 十 五年 有期徒刑
xu2 xi1 jun1 xiao1 jin1 sheng1 liu2 wen2 hua2 qu1 yong3 li4 wang2 kai1 yu3 luo4 ying1 deng3 ye3 bei4 fen1 bie2 pan4 chu3 liu4 nian2 zhi4 shi2 wu3 nian2 you3 qi1 tu2 xing2
x v2 x i1 j vn1 x iao1 j in1 sh eng1 l iu2 uu un2 h ua2 q v1 ii iong3 l i4 uu uang2 k ai1 vv v3 l uo4 ii ing1 d eng3 ii ie3 b ei4 f en1 b ie2 p an4 ch u3 l iu4 n ian2 zh ix4 sh ix2 uu u3 n ian2 ii iu3 q i1 t u2 x ing2
"""


def text_normalize(line: str):
    line = line.replace(" l =", "")
    line = line.upper()
    return line


def prepare_thchs_30(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    path = corpus_dir / "data_thchs30" / "data"
    transcript_dict = {}
    for text_path in path.rglob("**/*.wav.trn"):
        idx = Path(text_path.stem).stem
        with open(text_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx == 0:
                    line = text_normalize(line)
                    transcript_dict[idx] = line
                continue

    manifests = defaultdict(dict)
    dataset_parts = ["train", "dev", "test"]
    for part in tqdm(
        dataset_parts,
        desc="Process thchs_30 audio, it takes about 19 seconds.",
    ):
        logging.info(f"Processing thchs_30 subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []
        wav_path = corpus_dir / "data_thchs30" / f"{part}"
        for audio_path in wav_path.rglob("**/*.wav"):
            # logging.info(f"Processing audio path {audio_path}")
            idx = audio_path.stem
            speaker = idx.split("_")[0]
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                logging.warning(f"{audio_path} has no transcript.")
                continue
            text = transcript_dict[idx]
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue
            recording = Recording.from_file(audio_path)
            recordings.append(recording)
            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
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

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"thchs_30_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"thchs_30_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
