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

import logging
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

from tqdm.auto import tqdm

from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.audio import Recording
from lhotse.recipes.utils import create_cuts_per_recording
from lhotse.serialization import load_jsonl
from lhotse.supervision import SupervisionSegment
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


def _iter_utterances(
    data_dir: Path, num_jobs: int, chunk_size: int = 1000
) -> Iterator[Tuple[Recording, SupervisionSegment]]:
    """Yields (Recording, SupervisionSegment) for every utterance listed in the JSONL files of ``data_dir``."""

    def lines():
        for jsonl_file in sorted(data_dir.glob("*.jsonl")):
            logging.info(f"Processing {jsonl_file} with {num_jobs} jobs")
            yield from load_jsonl(jsonl_file)

    items = lines()
    with ThreadPoolExecutor(num_jobs) as ex:
        while True:
            chunk = list(islice(items, chunk_size))
            if not chunk:
                break
            futures = [ex.submit(_parse_utterance, data_dir, item) for item in chunk]
            for future in futures:
                result = future.result()
                if result is not None:
                    yield result


def prepare_emilia(
    corpus_dir: Pathlike,
    lang: str,
    num_jobs: int,
    output_dir: Optional[Pathlike] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]]:
    """
    Returns the manifests which consist of the Recordings, Supervisions and Cuts.
    When ``output_dir`` is given, the manifests are streamed to disk in a single pass
    and returned lazily; otherwise, eager manifests are returned.

    :param corpus_dir: Pathlike, the path of the data dir.
                       We assume the directory has the following structure:
                       corpus_dir/raw/openemilia_all.tar.gz,
                       corpus_dir/raw/DE,
                       corpus_dir/raw/DE/DE_B00000.jsonl,
                       corpus_dir/raw/DE/DE_B00000/DE_B00000_S00000/mp3/DE_B00000_S00000_W000000.mp3,
                       corpus_dir/raw/EN, etc.
    :param lang: str, one of en, zh, de, ko, ja, fr
    :param num_jobs: int, number of threads for processing jsonl files
    :param output_dir: Pathlike, the path where to write the manifests. Can be omitted to avoid writing.
        The files are named ``emilia-<lang>_<manifest>_all.jsonl.gz`` (e.g. ``emilia-de_cuts_all.jsonl.gz``).
    :return: A Dict with the single part ``'all'``, whose value is a Dict with the keys
        'recordings', 'supervisions', and 'cuts' for the given language.
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

    utterances = tqdm(
        _iter_utterances(data_dir, num_jobs), desc=f"Processing Emilia {lang_uppercase}"
    )

    if output_dir is None:
        recordings, supervisions = [], []
        for recording, segment in utterances:
            recordings.append(recording)
            supervisions.append(segment)
        recordings = RecordingSet.from_recordings(recordings)
        supervisions = SupervisionSet.from_segments(supervisions)
        return {
            "all": {
                "recordings": recordings,
                "supervisions": supervisions,
                "cuts": create_cuts_per_recording(recordings, supervisions),
            }
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_dir / f"emilia-{lang.lower()}_{name}_all.jsonl.gz"
        for name in ("recordings", "supervisions", "cuts")
    }
    with (
        RecordingSet.open_writer(paths["recordings"], overwrite=True) as recs_writer,
        SupervisionSet.open_writer(
            paths["supervisions"], overwrite=True
        ) as sups_writer,
        CutSet.open_writer(paths["cuts"], overwrite=True) as cuts_writer,
    ):
        for recording, segment in utterances:
            recs_writer.write(recording)
            sups_writer.write(segment)
            cut = recording.to_cut()
            cut.supervisions = [segment]
            cuts_writer.write(cut)
    return {
        "all": {
            "recordings": RecordingSet.from_jsonl_lazy(paths["recordings"]),
            "supervisions": SupervisionSet.from_jsonl_lazy(paths["supervisions"]),
            "cuts": CutSet.from_jsonl_lazy(paths["cuts"]),
        }
    }
