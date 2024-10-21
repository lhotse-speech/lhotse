"""
The Emilia dataset is constructed from a vast collection of speech data sourced
from diverse video platforms and podcasts on the Internet, covering various
content genres such as talk shows, interviews, debates, sports commentary, and
audiobooks. This variety ensures the dataset captures a wide array of real
human speaking styles. The initial version of the Emilia dataset includes a
total of 101,654 hours of multilingual speech data in six different languages:
English, French, German, Chinese, Japanese, and Korean.

See also
https://emilia-dataset.github.io/Emilia-Demo-Page/

Please note that Emilia does not own the copyright to the audio files; the
copyright remains with the original owners of the videos or audio. Users are
permitted to use this dataset only for non-commercial purposes under the
CC BY-NC-4.0 license.

Please refer to
https://huggingface.co/datasets/amphion/Emilia-Dataset
or
https://openxlab.org.cn/datasets/Amphion/Emilia
to download the dataset.

Note that you need to apply for downloading.

"""

from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.serialization import load_jsonl
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def _parse_utterance(
    data_dir: Path,
    line: dict,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    """
    :param data_dir: Path to the data directory
    :param line: dict, it looks like below::

        {
          "id": "DE_B00000_S00000_W000029",
          "wav": "DE_B00000/DE_B00000_S00000/mp3/DE_B00000_S00000_W000029.mp3",
          "text": " Und es gibt auch einen Stadtplan von Tegun zu sehen.",
          "duration": 3.228,
          "speaker": "DE_B00000_S00000",
          "language": "de",
          "dnsmos": 3.3697
        }

    :return: a tuple of "recording" and "supervision"
    """
    full_path = data_dir / line["wav"]

    if not full_path.is_file():
        return None

    recording = Recording.from_file(
        path=full_path,
        recording_id=full_path.stem,
    )
    segment = SupervisionSegment(
        id=recording.id,
        recording_id=recording.id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        text=line["text"],
        language=line["language"],
        speaker=line["speaker"],
        custom={"dnsmos": line["dnsmos"]},
    )

    return recording, segment


def prepare_emilia(
    corpus_dir: Pathlike,
    lang: str,
    num_jobs: int,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions

    :param corpus_dir: Pathlike, the path of the data dir.
                       We assume the directory has the following structure:
                       corpus_dir/raw/openemilia_all.tar.gz,
                       corpus_dir/raw/DE,
                       corpus_dir/raw/DE/DE_B00000.jsonl,
                       corpus_dir/raw/DE/DE_B00000/DE_B00000_S00000/mp3/DE_B00000_S00000_W000000.mp3,
                       corpus_dir/raw/EN, etc.
    :param lang: str, one of en, zh, de, ko, ja, fr
    :param num_jobs: int, number of threads for processing jsonl files
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: The RecordingSet and SupervisionSet with the keys 'recordings' and
             'supervisions'.
    """
    if lang is None:
        raise ValueError("Please provide --lang")

    lang_uppercase = lang.upper()
    if lang_uppercase not in ("DE", "EN", "FR", "JA", "KO", "ZH"):
        raise ValueError(
            "Please provide a valid language. "
            f"Choose from de, en, fr, ja, ko, zh. Given: {lang}"
        )

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    data_dir = corpus_dir / "raw" / lang_uppercase
    assert data_dir.is_dir(), f"No such directory: {data_dir}"

    jsonl_files = data_dir.glob("*.jsonl")

    recordings = []
    supervisions = []
    futures = []

    with ThreadPoolExecutor(num_jobs) as ex:
        for jsonl_file in jsonl_files:
            for item in tqdm(
                # Note: People's Speech manifest.json is really a JSONL.
                load_jsonl(jsonl_file),
                desc=f"Processing {jsonl_file} with {num_jobs} jobs",
            ):
                futures.append(
                    ex.submit(
                        _parse_utterance,
                        data_dir,
                        item,
                    )
                )

        for future in tqdm(futures, desc="Collecting futures"):
            result = future.result()
            if result is None:
                continue

            recording, segment = result

            recordings.append(recording)
            supervisions.append(segment)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(
        recordings=recording_set, supervisions=supervision_set
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        supervision_set.to_file(
            output_dir / f"emilia_supervisions_{lang_uppercase}.jsonl.gz"
        )
        recording_set.to_file(
            output_dir / f"emilia_recordings_{lang_uppercase}.jsonl.gz"
        )

    manifests = dict()
    manifests[lang_uppercase] = {
        "recordings": recording_set,
        "supervisions": supervision_set,
    }

    return manifests
