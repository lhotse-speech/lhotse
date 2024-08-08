"""
This recipe supports Chinese TTS corpora: WenetSpeech4TTS.

Paper: https://arxiv.org/abs/2406.05763v3
HuggingFace Dataset: https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS

Download using huggingface-cli:
huggingface-cli login
huggingface-cli download --repo-type dataset --local-dir $DATA_DIR Wenetspeech4TTS/WenetSpeech4TTS

Extract the downloaded data:
for folder in Standard Premium Basic; do
  for file in "$folder"/*.tar.gz; do
    tar -xzvf "$file" -C "$folder"
  done
done
"""
import logging
import re
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm import tqdm

from lhotse import (
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike, resumable_download, safe_extract

WENETSPEECH4TTS = (
    "Basic",
    "Premium",
    "Standard",
)


def prepare_wenetspeech4tts(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "Basic",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'Basic', 'Premium'.
        By default we will prepare all parts.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: the number of parallel workers parsing the data.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if dataset_parts == "all" or dataset_parts[0] == "all":
        dataset_parts = WENETSPEECH4TTS
    elif isinstance(dataset_parts, str):
        assert (
            dataset_parts in WENETSPEECH4TTS
        ), f"Unsupported dataset part: {dataset_parts}"
        dataset_parts = [dataset_parts]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir, prefix="wenetspeech4tts"
        )

    basic_wav_scp_dict = {}
    premium_wav_scp_dict = {}
    standard_wav_scp_dict = {}
    with open(corpus_dir / "filelists" / "Basic_filelist.lst") as f:
        for line in f:
            line = line.strip().split()
            basic_wav_scp_dict[line[0]] = line[1]
            if "Basic" not in line[1]:
                standard_wav_scp_dict[line[0]] = line[1]
            if "Premium" in line[1]:
                premium_wav_scp_dict[line[0]] = line[1]

    basic_dns_mos_dict = {}
    premium_dns_mos_dict = {}
    standard_dns_mos_dict = {}
    with open(corpus_dir / "DNSMOS_P808Scores" / "Basic_DNSMOS.lst") as f:
        for line in f:
            line = line.strip().split()
            basic_dns_mos_dict[line[0]] = float(line[1])
    with open(corpus_dir / "DNSMOS_P808Scores" / "Premium_DNSMOS.lst") as f:
        for line in f:
            line = line.strip().split()
            premium_dns_mos_dict[line[0]] = float(line[1])
    with open(corpus_dir / "DNSMOS_P808Scores" / "Standard_DNSMOS.lst") as f:
        for line in f:
            line = line.strip().split()
            standard_dns_mos_dict[line[0]] = float(line[1])

    for part in dataset_parts:
        if manifests_exist(part=part, output_dir=output_dir, prefix="wenetspeech4tts"):
            logging.info(f"WenetSpeech4TTS subset: {part} already prepared - skipping.")
            continue
        recordings = []
        supervisions = []
        if part == "Premium":
            wav_scp_dict = premium_wav_scp_dict
            dns_mos_dict = premium_dns_mos_dict
        elif part == "Standard":
            wav_scp_dict = standard_wav_scp_dict
            dns_mos_dict = standard_dns_mos_dict
        else:
            wav_scp_dict = basic_wav_scp_dict
            dns_mos_dict = basic_dns_mos_dict
        for wav_name, wav_path in tqdm(
            wav_scp_dict.items(), desc=f"Preparing WenetSpeech4TTS {part}"
        ):
            # get the actual wav path, remove the prefix '../'
            # e.g. ../Premium/WenetSpeech4TTS_Premium_9/wavs/X0000015306_83500032_S00110-S00112.wav -> Premium/WenetSpeech4TTS_Premium_9/wavs/X0000015306_83500032_S00110-S00112.wav
            assert wav_path.startswith("../")
            wav_path = corpus_dir / wav_path[3:]
            if not wav_path.is_file():
                logging.warning(f"No such file: {wav_path}")
                continue
            recording = Recording.from_file(wav_path)
            recordings.append(recording)

            # get the text path
            # e.g. ../Premium/WenetSpeech4TTS_Premium_9/txts/X0000015306_83500032_S00110-S00112.txt
            txt_path = (
                wav_path.parent.parent
                / "txts"
                / wav_path.name.replace("wavs", "txts").replace(".wav", ".txt")
            )
            if not txt_path.is_file():
                logging.warning(f"No such file: {txt_path}")
                continue
            with open(txt_path, "r") as f:
                lines = f.readlines()
                text = lines[0].strip().split("\t")[1]
                timestamp = lines[1].strip()
            supervisions.append(
                SupervisionSegment(
                    id=wav_name,
                    recording_id=wav_name,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    language="Chinese",
                    text=text,
                    custom={
                        "timestamp": timestamp,
                        "dns_mos": dns_mos_dict.get(wav_name, None),
                    },
                )
            )
        recordings = RecordingSet.from_recordings(recordings)
        supervisions = SupervisionSet.from_segments(supervisions)
        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        if output_dir is not None:
            supervisions.to_file(
                output_dir / f"wenetspeech4tts_supervisions_{part}.jsonl.gz"
            )
            recordings.to_file(
                output_dir / f"wenetspeech4tts_recordings_{part}.jsonl.gz"
            )

        manifests[part] = {"recordings": recordings, "supervisions": supervisions}

    return manifests
