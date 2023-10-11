"""
Description taken from the official website of Russian Open Speech To Text
(https://learn.microsoft.com/en-us/azure/open-datasets/dataset-open-speech-text?tabs=azure-storage)

This Russian speech to text (STT) dataset includes:

~16 million utterances
~20,000 hours
2.3 TB (uncompressed in .wav format in int16), 356G in opus
All files were transformed to opus, except for validation datasets
The main purpose of the dataset is to train speech-to-text models.

See https://learn.microsoft.com/en-us/azure/open-datasets/dataset-open-speech-text?tabs=azure-storage for more details about Russian Open Speech To Text
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm

from lhotse import (
    fix_manifests,
    validate_recording,
    validate_recordings_and_supervisions,
    validate_supervision,
)
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

EXCLUDED_PARTS = (
    "asr_public_phone_calls_1",  # 75% quality
    "asr_public_phone_calls_2",  # 75% quality
    "asr_public_stories_1",  # 80% quality
    "asr_public_stories_2",  # 80% quality
)

RU_OPEN_STT_TRAIN_PARTS = (
    "asr_calls_2_val",
    "private_buriy_audiobooks_2",
    "public_lecture_1",
    "public_series_1",
    "public_speech_manifest",
    "public_youtube1120",
    "public_youtube1120_hq",
    "public_youtube700",
    "radio_2",
    "radio_pspeech_sample_manifest",
    "radio_v4_add_manifest",
    "radio_v4_manifest",
    "tts_russian_addresses_rhvoice_4voices",
)

RU_OPEN_STT_DEV_PARTS = ("buriy_audiobooks_2_val",)

RU_OPEN_STT_TEST_PARTS = ("public_youtube700_val",)


def _read_raw_manifest(corpus_dir):
    """Build a list of wav dir and corresponding txt dir pairs from a list
    of raw manifests.

    :param corpus_dir: Path, dir of the ru_open_stt data.
    :return: a list of all the wav and corresponding txt absolute dir pairs of the whole dataset.
    """
    print(f"->Start reading raw manifests...")
    raw_train_manifest, raw_dev_manifest, raw_test_manifest = set(), set(), set()
    raw_manifest_dir = corpus_dir / "manifests"
    for manifest in sorted(raw_manifest_dir.iterdir()):
        m_name = str(manifest).split("/")[-1].split(".")[0]
        if m_name in RU_OPEN_STT_DEV_PARTS:
            target = raw_dev_manifest
        elif m_name in RU_OPEN_STT_TEST_PARTS:
            target = raw_test_manifest
        elif m_name in RU_OPEN_STT_TRAIN_PARTS:
            target = raw_train_manifest
        else:
            continue
        print(f"----Start reading {m_name}...")
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f.readlines():
                relative_wav = line.split(",")[0]
                absolute_wav = corpus_dir / "untar" / relative_wav
                assert absolute_wav.is_file(), f"{absolute_wav} is not exist!"
                if not absolute_wav.with_suffix(".txt").is_file():
                    continue
                target.add(absolute_wav)
        print(f"----Finish reading {m_name}!")
    print(
        f"<-Finish reading raw manifests, total samples: {len(raw_train_manifest) + len(raw_dev_manifest) + len(raw_test_manifest)}!"
    )
    return raw_train_manifest, raw_dev_manifest, raw_test_manifest


def prepare_ru_open_stt(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: Number of workers to extract manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with
             the keys 'recordings' and 'supervisions'.
    """
    print(f"num of threads: {num_jobs}")
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)
    raw_train_manifest, raw_dev_manifest, raw_test_manifest = _read_raw_manifest(
        corpus_dir
    )
    for split, raw_manifest in zip(
        ["train", "dev", "test"],
        [raw_train_manifest, raw_dev_manifest, raw_test_manifest],
    ):
        manifests[split] = {"recordings": [], "supervisions": []}
        with ProcessPoolExecutor(num_jobs) as ex:
            for recording, segment in tqdm(
                ex.map(parse_utterance, raw_manifest),
                desc=f"Processing RU_OPEN_STT {split} manifests",
            ):
                if recording is not None:
                    manifests[split]["recordings"].append(recording)
                    manifests[split]["supervisions"].append(segment)
        recordings, supervisions = fix_manifests(
            recordings=RecordingSet.from_recordings(manifests[split]["recordings"]),
            supervisions=SupervisionSet.from_segments(manifests[split]["supervisions"]),
        )
        validate_recordings_and_supervisions(
            recordings=recordings, supervisions=supervisions
        )
        if output_dir is not None:
            supervisions.to_file(
                output_dir / f"ru_open_stt_supervisions_{split}.jsonl.gz"
            )
            recordings.to_file(output_dir / f"ru_open_stt_recordings_{split}.jsonl.gz")
        manifests[split] = {
            "recordings": recordings,
            "supervisions": supervisions,
        }
    return manifests


def parse_utterance(
    wav_dir: Tuple,
) -> Tuple[Recording, Dict[str, List[SupervisionSegment]]]:
    text_dir = wav_dir.with_suffix(".txt")
    recording_id = str(wav_dir).split(".")[0].split("/")[-4:]
    recording_id = "_".join(recording_id)
    try:
        recording = Recording.from_file(wav_dir, recording_id=recording_id, force_opus_sampling_rate=16000)
        validate_recording(recording)
        with open(text_dir, "r", encoding="utf-8") as f:
            text = f.read().strip()
        segment = SupervisionSegment(
            id=recording_id,
            recording_id=recording_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            language="Russian",
            text=text,
        )
        validate_supervision(segment)
        return recording, segment
    except:
        return None, None


if __name__ == "__main__":
    corpus_dir = Path("/mgData2/yangb/data/ru_open_stt")
    output_dir = Path("/mgData2/yangb/icefall/egs/ru_open_stt/ASR/data/test_manifests")
    prepare_ru_open_stt(corpus_dir=corpus_dir, output_dir=output_dir, num_jobs=20)
