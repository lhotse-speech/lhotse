"""
# SpokenWOZ dataset

SpokenWOZ is a large-scale multi-domain speech-text dataset
for spoken task-oriented dialogue modeling, which consists of 203k turns,
5.7k dialogues and 249 hours audios from realistic human-to-human spoken conversations.

The data is split into training, dev, and test sets.
The dataset is distributed under the CC BY-NC 4.0 license.


## Why SpokenWOZ?

The majority of existing TOD datasets are constructed via writing or paraphrasing
from annotators rather than being collected from realistic spoken conversations.
The written TDO datasets may not be representative of the way people naturally speak
in real-world conversations, and make it difficult to train and evaluate models
that are specifically designed for spoken TOD.
Additionally, the robustness issue, such as ASR noise, also can not be fully explored
using these written TOD datasets. Different exsiting spoken TOD datasets,
we introduce common spoken characteristics in SpokenWOZ, such like word-by-word processing
and commonsense in spoken language.
SpokenWOZ also includes cross-turn detection and reasoning slot detection
as new challenges to better handle these spoken characteristics.


## Data structure

There are 5,700 dialogues ranging form single-domain to multi-domain in SpokenWOZ.
The test sets contain 1k examples.
Dialogues with MUL in the name refers to multi-domain dialogues.
Dialogues with SNG refers to single-domain dialogues. Each dialogue consists of a goal,
multiple user and system utterances, dialogue state, dialogue act, corresponding audio and ASR transcription.

The file name of the audio is consistent with the id of the dialogue, for example,
the corresponding audio file for MUL0032 is MUL0032.wav.

The dialogue goal for each dialogue is recorded in the "goal" field.
The dialogue goal holds the fields involved in the dialogue as well as
the slots involved and the corresponding values.

The dialogue state for each dialogue is recorded in the "metadata" field in every turn the same as MultiWOZ 2.1.
The  state have two sections: semi, book. Semi refers to slots from a particular domain.
Book refers to booking slots for a particular domain. The joint accuracy metrics includes ALL slots.

The dialogue act for each dialogue is recorded in the "dialogue_act" and "span_info" field in every turn:

```
{
  "$dialogue_id": {
  "log":{
    "$turn_id": {
      "dialogue_act": {
        "$act_name": [
          [
            "$slot_name",
            "$action_value"
          ]
        ]
      },
      "span_info": [
        [
          "$act_name"
          "$slot_name",
          "$action_value"
          "$start_charater_index",
          "$exclusive_end_character_index"
        ]
  }
}
```

The ASR transcription for each dialogue is recorded in the "words" field in every turn.

```
{
  "$dialogue_id": {
  "log":{
    "$turn_id": {
      "words": [
        {
        "$word_context": "$word",
        "$begin_time": "$begintime",
        "end_time": "$endtime",
        "channel_id": "$channel",
        "word_index": "$index",
        }
  }
}
```


## Citation

[1] Website https://spokenwoz.github.io/SpokenWOZ-github.io/
[2] Arxiv pre-print
```
@article{si2023spokenwoz,
  title={SpokenWOZ: A Large-Scale Speech-Text Dataset for Spoken Task-Oriented Dialogue in Multiple Domains},
  author={Si, Shuzheng and Ma, Wentao and Wu, Yuchuan and Dai, Yinpei and Gao, Haoyu and Lin, Ting-En and Li, Hangyu and Yan, Rui and Huang, Fei and Li, Yongbin},
  journal={arXiv preprint arXiv:2305.13040},
  year={2023},
  url={https://arxiv.org/abs/2305.13040}
}
```

"""
import json
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from tqdm import tqdm

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import AlignmentItem
from lhotse.utils import Pathlike, resumable_download, safe_extract

SPOKENWOZ_BASE_URL = (
    "https://spokenwoz.oss-cn-wulanchabu.aliyuncs.com/{modality}_5700_{part}.tar.gz"
)

MODALITIES = ("text", "audio")
# Parts how the dataset is distributed.
PARTS = ("test", "train_dev")
# Splits how the dataset is used and saved in Lhotse manifests.
SPLITS = ("test", "dev", "train")


def get_spokenwoz_metadata(corpus_dir: Pathlike) -> Dict[str, Any]:
    """
    Helper function which loads the metadata not included explicitly to the Lhotse manifests.
    """
    td = Path(corpus_dir) / "text_5700_train_dev"
    metadata = {
        "ontology.json": json.load(open(td / "ontology.json", "r")),
        "README.md": open(td / "README.md").read(),
        "db": {
            "data": {
                "attraction": json.load(open(td / "db" / "attraction_db.json", "r")),
                "hospital": json.load(open(td / "db" / "hospital_db.json", "r")),
                "hotel": json.load(open(td / "db" / "hotel_db.json", "r")),
                "police": json.load(open(td / "db" / "police_db.json", "r")),
                "restaurant": json.load(open(td / "db" / "restaurant_db.json", "r")),
                "taxi": json.load(open(td / "db" / "taxi_db.json", "r")),
                "train": json.load(open(td / "db" / "train_db.json", "r")),
            },
            "value_set": json.load(open(td / "db" / "value_set.json", "r")),
        },
    }
    return metadata


def download_spokenwoz(
    target_dir: Pathlike = ".",
    dataset_parts: Optional[Union[str, Sequence[str]]] = "all",
    force_download: Optional[bool] = False,
) -> Path:
    """
    Download and untar the SpokenWOZ dataset.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param dataset_parts: "all", or a list of parts "train_dev",  or "test" to download.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if dataset_parts == "all" or dataset_parts[0] == "all":
        dataset_parts = PARTS
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    dataset_parts = [
        (part, modality) for part in dataset_parts for modality in MODALITIES
    ]
    for part, modality in tqdm(dataset_parts, desc=f"Downloading SpokenWOZ parts"):
        if part not in PARTS:
            logging.warning(
                f"Skipping invalid dataset part name: {part} (possible choices: {PARTS})"
            )
            continue
        url = SPOKENWOZ_BASE_URL.format(modality=modality, part=part)
        tar_name = f"{modality}_5700_{part}.tar.gz"
        tar_path = target_dir / tar_name
        part_dir = target_dir / f"{modality}_5700_{part}"
        target_dir.mkdir(parents=True, exist_ok=True)
        completed_detector = part_dir / ".completed"
        if completed_detector.is_file():
            logging.info(
                f"Skipping {modality}-{part} because {completed_detector} exists."
            )
            continue
        resumable_download(url, filename=tar_path, force_download=force_download)
        shutil.rmtree(part_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            safe_extract(tar, path=target_dir)
        completed_detector.touch()


def prepare_spokenwoz(
    corpus_dir: Pathlike,
    dataset_splits: Union[str, Sequence[str]] = "all",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names 'train_dev', or 'test'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: the number of parallel workers parsing the data.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    manifests = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_splits == "all" or dataset_splits[0] == "all":
        dataset_splits = SPLITS
    elif isinstance(dataset_splits, str):
        dataset_splits = [dataset_splits]

    text_dirs = {
        "train": corpus_dir / f"text_5700_train_dev",
        "dev": corpus_dir / f"text_5700_train_dev",
        "test": corpus_dir / f"text_5700_test",
    }
    audio_dirs = {
        "train": corpus_dir / f"audio_5700_train_dev",
        "dev": corpus_dir / f"audio_5700_train_dev",
        "test": corpus_dir / f"audio_5700_test",
    }

    dialogue_ids = {"train": None, "dev": None, "test": None}
    exclude_dialogue_ids = {"train": None, "dev": None, "test": []}
    if "train" in dataset_splits or "dev" in dataset_splits:
        # It is actually not a json file, but a list of dialogue ids. One per line.
        dialogue_ids["dev"] = (
            open(corpus_dir / "text_5700_train_dev" / "valListFile.json")
            .read()
            .splitlines()
        )
        train_dev = list(json.load(open(text_dirs["train"] / "data.json", "r")).keys())
        dialogue_ids["train"] = list(set(train_dev) - set(dialogue_ids["dev"]))
        exclude_dialogue_ids["train"] = dialogue_ids["dev"]
        exclude_dialogue_ids["dev"] = dialogue_ids["train"]
    if "test" in dataset_splits:
        dialogue_ids_data = list(
            json.load(open(text_dirs["test"] / "data.json", "r")).keys()
        )
        dialogue_ids_list = (
            open(corpus_dir / "text_5700_test" / "testListFile.json")
            .read()
            .splitlines()
        )
        # Take into account that the assert fails ie the data are different!
        # assert dialogue_ids_data == dialogue_ids_list, f"The testListFile.json does not match the data.json: {dialogue_ids_data} != {dialogue_ids_list}"
        dialogue_ids["test"] = dialogue_ids_list

    for split in tqdm(dataset_splits, desc="Preparing spokenWOZ parts"):
        if manifests_exist(part=split, output_dir=output_dir, prefix="spokenwoz"):
            logging.info(f"SpokenWOZ subset: {split} already prepared - skipping.")
            continue

        recordings, supervisions = _spokenwoz_manifests(
            audio_dirs[split],
            text_dirs[split],
            dialogue_ids[split],
            exclude_dialogue_ids[split],
            num_jobs=num_jobs,
        )

        if output_dir is not None:
            supervisions.to_file(
                output_dir / f"spokenwoz_supervisions_{split}.jsonl.gz"
            )
            recordings.to_file(output_dir / f"spokenwoz_recordings_{split}.jsonl.gz")
        manifests[split] = {"recordings": recordings, "supervisions": supervisions}

    return manifests


def _spokenwoz_manifests(
    audio_dir: Pathlike,
    text_dir: Pathlike,
    dialogue_ids: List[str],
    exclude_dialogue_ids: List[str],
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """Opens the SpokenWOZ audio directory, list all the wav recordings in the directory.
    Loads the data.json file from the text directory.
    See the spokenwoz module doc string for the 'Data structure' overview at the top of this file.
    For each dialogue and turn we will create SupervisionSegments.
    We will add the the dialog_act span_info, tag={user,system}, to the custom dictionary of the SupervisionSegment.
    We will fill the text and alignment using the words field from the data.json file.
    ATM the turn['log']['metadata'] field is not saved to the SupervisionSegment.
    """

    audio_dir = Path(audio_dir)
    wav_files = dict(
        ((wavf.stem, wavf) for wavf in audio_dir.iterdir() if wavf.suffix == ".wav")
    )

    with open(text_dir / "data.json", "r") as f:
        data = json.load(f)

    missing_dialogues_data = [did for did in dialogue_ids if did not in data.keys()]
    assert (
        len(missing_dialogues_data) == 0
    ), f"The dialogues are missing from the data.json: {missing_dialogues_data}"
    missing_audio_files = [did for did in dialogue_ids if did not in wav_files.keys()]
    assert (
        len(missing_audio_files) == 0
    ), f"The dialogues do not have corresponding recording:{missing_audio_files}"

    exclude_pattern = (
        "|".join(f"{did}\.wav" for did in exclude_dialogue_ids)
        if exclude_dialogue_ids
        else None
    )
    recordings = RecordingSet.from_dir(
        audio_dir, "*.wav", num_jobs=num_jobs, exclude_pattern=exclude_pattern
    )

    supervisions = []
    for did, dv in data.items():
        for i, turn in enumerate(dv["log"]):

            words = turn["words"]
            word_alignments = [
                AlignmentItem(
                    start=w["BeginTime"] / 1000.0,
                    duration=(w["EndTime"] - w["BeginTime"]) / 1000.0,
                    symbol=w["Word"],
                )
                for w in words
            ]

            supervisions.append(
                SupervisionSegment(
                    id=f"{did}-{i:03d}",
                    recording_id=did,
                    start=words[0]["BeginTime"] / 1000.0,
                    duration=(words[-1]["EndTime"] - words[0]["BeginTime"]) / 1000.0,
                    channel=0 if turn["tag"] == "user" else 1,  # user: 0, system: 1
                    text=turn["text"],
                    language="en",
                    custom={
                        "dialogue_act": turn["dialog_act"],
                        "span_info": turn["span_info"],
                        "tag": turn["tag"],
                    },
                    alignment={"words": word_alignments},
                )
            )

    supervisions = SupervisionSet.from_segments(supervisions)
    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    return recordings, supervisions
