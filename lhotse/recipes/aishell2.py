"""
AISHELL2 (~1000 hours) if available(https://www.aishelltech.com/aishell_2).
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def text_normalize(line: str) -> str:
    """
    Modified from https://github.com/wenet-e2e/wenet/blob/main/examples/multi_cn/s0/local/aishell2_data_prep.sh#L50

    """
    line = line.replace("Ａ", "A")
    line = line.replace("Ｔ", "T")
    line = line.replace("Ｍ", "M")
    line = line.replace("𫖯", "頫")
    line = line.replace("，", "")
    line = line.replace("?", "")
    line = line.replace("-", " ")
    line = line.upper()
    """
    The below code is to only remove "'" of mandarin.
    The "'" of english remains unchanged.

    for example:
    IC0001W0061     听流年
    IC0001W0062     听beat it
    IC0001W0063     听独角戏
    IC0001W0064     听心雨
    IC0001W0065     听Yesterday Once More
    IC0001W0066     听广岛之恋
    IC0001W0067     听一生有你
    IC0010W0228	    Here's
    IC0012W0161	    I'm
    IC0013W0018	    It's
    IC0017W0126	    Nothing'sGChange
    IC0020W0392	    She's
    IC0022W0444	    That's
    IC0073W0058	    搬不走的要及时'关停并转'
    IC0085W0187     帮我放一首歌Let's
    IC0392W0410	    对低收入群体的帮助也更大'
    IC0975W0451	    明年二月底'小成'
    ID0114W0368	    我感觉就是在不断'拉抽屉'
    ID0115W0198	    我公司员工不存在持有'和泰创投'股份的情况

    ---->>
    IC0001W0061 听流年
    IC0001W0062 听BEAT IT
    IC0001W0063 听独角戏
    IC0001W0064 听心雨
    IC0001W0065 听YESTERDAY ONCE MORE
    IC0001W0066 听广岛之恋
    IC0001W0067 听一生有你
    IC0010W0228 HERE'S
    IC0012W0161 I'M
    IC0013W0018 IT'S
    IC0017W0126 NOTHING'SGCHANGE
    IC0020W0392 SHE'S
    IC0022W0444 THAT'S
    IC0073W0058 搬不走的要及时关停并转
    IC0085W0187 帮我放一首歌LET'S
    IC0392W0410 对低收入群体的帮助也更大
    IC0975W0451 明年二月底小成
    ID0114W0368 我感觉就是在不断拉抽屉
    ID0115W0198 我公司员工不存在持有和泰创投股份的情况

    """
    new_line = []
    line = list(line)
    for i, char in enumerate(line):
        if char == "'" and "\u4e00" <= line[i - 1] <= "\u9fff":
            char = char.replace("'", "")
        new_line.append(char)
    line = "".join(new_line)
    line = line.upper()
    return line


def prepare_aishell2(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part,
             and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)
    dataset_parts = ["train", "dev", "test"]
    for part in tqdm(
        dataset_parts,
        desc="Process aishell2 audio, it takes about 55  minutes using 40 cpu jobs.",
    ):
        logging.info(f"Processing aishell2 subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)

        if part == "train":
            transcript_path = corpus_dir / "AISHELL-2" / "iOS" / "data" / "trans.txt"
            wav_path = corpus_dir / "AISHELL-2" / "iOS" / "data" / "wav"
        else:
            # using dev_ios, test_ios
            transcript_path = corpus_dir / "AISHELL-2" / "iOS" / f"{part}" / "trans.txt"
            wav_path = corpus_dir / "AISHELL-2" / "iOS" / f"{part}" / "wav"

        transcript_dict = {}
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                idx_transcript = line.split()
                content = " ".join(idx_transcript[1:])
                content = text_normalize(content)
                transcript_dict[idx_transcript[0]] = content

        supervisions = []
        recordings = RecordingSet.from_dir(
            path=wav_path, pattern="*.wav", num_jobs=num_jobs
        )

        for audio_path in wav_path.rglob("**/*.wav"):

            idx = audio_path.stem
            speaker = audio_path.parts[-2]
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                logging.warning(f"{audio_path} has no transcript.")
                continue
            text = transcript_dict[idx]
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue

            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recordings.duration(idx),
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
                output_dir / f"aishell2_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"aishell2_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests
