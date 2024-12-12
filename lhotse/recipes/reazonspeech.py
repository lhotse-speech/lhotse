"""
ReazonSpeech is an open-source dataset that contains a diverse set of natural Japanese speech,
collected from terrestrial television streams. It contains more than 35,000 hours of audio.

The dataset is available on Hugging Face. For more details, please visit:

Dataset: https://huggingface.co/datasets/reazon-research/reazonspeech
Paper: https://research.reazon.jp/_static/reazonspeech_nlp2023.pdf
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import CutSet, fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.parallel import parallel_map
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available

REAZONSPEECH = (
    "tiny",
    "small",
    "medium",
    "large",
    "all",
    "small-v1",
    "medium-v1",
    "all-v1",
)

PUNCTUATIONS = {ord(x): "" for x in "、。「」『』，,？！!!?!?"}
ZENKAKU = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９"
HANKAKU = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ZEN2HAN = str.maketrans(ZENKAKU, HANKAKU)


def normalize(s):
    """
    Convert full-width characters to half-width, and remove punctuations.
    :param s: str, input string.
    :return: str, normalized string.
    """
    if is_module_available("num2words"):
        import num2words
    else:
        raise ImportError(
            "To process the ReazonSpeech corpus, please install optional dependency: pip install num2words"
        )
    s = s.translate(PUNCTUATIONS).translate(ZEN2HAN)
    conv = lambda m: num2words.num2words(m.group(0), lang="ja")
    return re.sub(r"\d+\.?\d*", conv, s)


def write_to_json(data, filename):
    """
    Writes data to a JSON file.
    :param data: The data to write.
    :param filename: The name of the file to write to.
    """

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def download_reazonspeech(
    target_dir: Pathlike = ".",
    dataset_parts: Optional[Union[str, Sequence[str]]] = "auto",
    num_jobs: int = 1,
) -> Path:
    """
    Download the ReazonSpeech dataset.
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param dataset_parts: the parts of the dataset to download (e.g. small, medium, or large).
    :param num_jobs: the number of processes to download and format.
    :return: the path to downloaded data and the JSON file.
    """
    if is_module_available("datasets"):
        import soundfile as sf
        from datasets import Audio, load_dataset
    else:
        raise ImportError(
            "To process the ReazonSpeech corpus, please install optional dependencies: pip install datasets soundfile"
        )
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "ReazonSpeech"

    if dataset_parts == "auto":
        dataset_parts = ("small-v1",)
    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    for part in dataset_parts:
        logging.info(f"Downloading ReazonSpeech part: {part}")
        ds = load_dataset(
            "reazon-research/reazonspeech",
            part,
            trust_remote_code=True,
            cache_dir=corpus_dir,
            num_proc=num_jobs,
        )["train"]

    # Prepare data for JSON export
    def format_example(example: dict, idx: int) -> dict:
        example["id"] = str(idx)
        example["audio_filepath"] = example["audio"]["path"]
        example["text"] = normalize(example["transcription"])
        example["duration"] = sf.info(example["audio"]["path"]).duration
        return example

    ds = ds.cast_column("audio", Audio(decode=True))  # Hack: don't decode to speedup
    ds = ds.map(
        format_example,
        with_indices=True,
        remove_columns=ds.column_names,
        num_proc=num_jobs,
    )

    # Write data to a JSON file
    ds.to_json(
        corpus_dir / "dataset.json",
        num_proc=num_jobs,
        force_ascii=False,
        indent=4,
        lines=False,
        batch_size=ds.num_rows,
    )

    return corpus_dir


def prepare_reazonspeech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike],
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: int, number of parallel threads used for 'parse_utterance' calls.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    # Split the dataset into train, dev, and test
    with open(corpus_dir / "dataset.json", "r", encoding="utf-8") as file:
        full = json.load(file)
        dev = full[:1000]
        test = full[1000:1100]
        train = full[1100:]

        write_to_json(train, corpus_dir / "train.json")
        write_to_json(dev, corpus_dir / "dev.json")
        write_to_json(test, corpus_dir / "test.json")

    parts = ("train", "dev", "test")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Maybe some manifests already exist: we can read them and save a bit of preparation time.
    manifests = read_manifests_if_cached(
        dataset_parts=parts,
        output_dir=output_dir,
        prefix="reazonspeech",
        suffix="jsonl.gz",
        lazy=True,
    )

    for part in parts:
        logging.info(f"Processing ReazonSpeech subset: {part}")
        if manifests_exist(
            part=part, output_dir=output_dir, prefix="reazonspeech", suffix="jsonl.gz"
        ):
            logging.info(f"ReazonSpeech subset: {part} already prepared - skipping.")
            continue

        filename = corpus_dir / f"{part}.json"
        with open(filename, "r", encoding="utf-8") as file:
            items = json.load(file)

        with RecordingSet.open_writer(
            output_dir / f"reazonspeech_recordings_{part}.jsonl.gz"
        ) as rec_writer, SupervisionSet.open_writer(
            output_dir / f"reazonspeech_supervisions_{part}.jsonl.gz"
        ) as sup_writer, CutSet.open_writer(
            output_dir / f"reazonspeech_cuts_{part}.jsonl.gz"
        ) as cut_writer:
            for recording, segment in tqdm(
                parallel_map(
                    parse_utterance,
                    items,
                    num_jobs=num_jobs,
                ),
                desc="Processing reazonspeech JSON entries",
            ):
                # Fix and validate the recording + supervisions
                recordings, segments = fix_manifests(
                    recordings=RecordingSet.from_recordings([recording]),
                    supervisions=SupervisionSet.from_segments([segment]),
                )
                validate_recordings_and_supervisions(
                    recordings=recordings, supervisions=segments
                )
                # Create the cut since most users will need it anyway.
                # There will be exactly one cut since there's exactly one recording.
                cuts = CutSet.from_manifests(
                    recordings=recordings, supervisions=segments
                )
                # Write the manifests
                rec_writer.write(recordings[0])
                sup_writer.write(segments[0])
                cut_writer.write(cuts[0])

        manifests[part] = {
            "recordings": RecordingSet.from_jsonl_lazy(rec_writer.path),
            "supervisions": SupervisionSet.from_jsonl_lazy(sup_writer.path),
            "cuts": CutSet.from_jsonl_lazy(cut_writer.path),
        }

    return dict(manifests)


def parse_utterance(item: Any) -> Optional[Tuple[Recording, SupervisionSegment]]:
    """
    Process a single utterance from the ReazonSpeech dataset.
    :param item: The utterance to process.
    :return: A tuple containing the Recording and SupervisionSegment.
    """
    recording = Recording.from_file(item["audio_filepath"], recording_id=item["id"])
    segments = SupervisionSegment(
        id=item["id"],
        recording_id=item["id"],
        start=0.0,
        duration=item["duration"],
        channel=0,
        language="Japanese",
        text=item["text"],
    )
    return recording, segments
