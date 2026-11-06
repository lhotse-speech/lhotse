"""
Description taken from official website: https://datasets.kensho.com/datasets/spgispeech
SPGISpeech consists of 5,000 hours of recorded company earnings calls and their respective
transcriptions. The original calls were split into slices ranging from 5 to 15 seconds in
length to allow easy training for speech recognition systems. Calls represent a broad
cross-section of international business English; SPGISpeech contains approximately 50,000
speakers, one of the largest numbers of any speech corpus, and offers a variety of L1 and
L2 English accents. The format of each WAV file is single channel, 16kHz, 16 bit audio.

Transcription text represents the output of several stages of manual post-processing.
As such, the text contains polished English orthography following a detailed style guide,
including proper casing, punctuation, and denormalized non-standard words such as numbers
and acronyms, making SPGISpeech suited for training fully formatted end-to-end models.

Official reference:

O’Neill, P.K., Lavrukhin, V., Majumdar, S., Noroozi, V., Zhang, Y., Kuchaiev, O., Balam,
J., Dovzhenko, Y., Freyberg, K., Shulman, M.D., Ginsburg, B., Watanabe, S., & Kucsko, G.
(2021). SPGISpeech: 5, 000 hours of transcribed financial audio for fully formatted
end-to-end speech recognition. ArXiv, abs/2104.02014.

ArXiv link: https://arxiv.org/abs/2104.02014
"""

import logging
import string
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.cut import CutSet
from lhotse.parallel import parallel_map
from lhotse.recipes.utils import (
    create_cuts_per_recording,
    manifests_exist,
    read_manifests_if_cached,
)
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds


def download_spgispeech(
    target_dir: Pathlike = ".",
) -> None:
    """
    Download and untar the dataset.

    NOTE: This function just returns with a message since SPGISpeech is not available
    for direct download.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    """
    logging.info(
        "SPGISpeech is not available for direct download. Please fill out the form at"
        " https://datasets.kensho.com/datasets/spgispeech to download the corpus."
    )


def normalize(text: str) -> str:
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert all upper case to lower case
    text = text.lower()
    return text


def _read_utterance(
    utterance: Tuple[Path, str, str],
) -> Tuple[Recording, SupervisionSegment]:
    audio_path, recording_id, text = utterance
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        text=text,
        speaker=recording_id.split("_")[0],
        start=0,
        duration=recording.duration,
        language="English",
    )
    return recording, segment


def _iter_transcripts(
    csv_path: Path, audio_dir: Path, normalize_text: bool
) -> Iterator[Tuple[Path, str, str]]:
    """Yields (audio path, recording ID, text) for every row of the transcripts CSV."""
    with open(csv_path, "r") as f:
        # Skip the header
        next(f)
        for line in f:
            parts = line.strip().split("|")
            # 07a785e9237c389c1354bb60abca42d5/1.wav -> 07a785e9237c389c1354bb60abca42d5_1
            recording_id = parts[0].replace("/", "_").replace(".wav", "")
            text = parts[2]
            if normalize_text:
                text = normalize(text)
            yield audio_dir / parts[0], recording_id, text


def prepare_spgispeech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    normalize_text: bool = True,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet, CutSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    The transcripts CSV drives the preparation: every row is turned into a recording, a supervision and a cut
    in a single pass. When ``output_dir`` is given, the manifests are streamed to disk to avoid OOM issues,
    since it is a large dataset, and returned lazily. Otherwise, eager manifests are returned.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests. Can be omitted to avoid writing.
    :param normalize_text: Bool, if True, normalize the text (similar to ESPNet recipe).
    :param num_jobs: int, the number of jobs to use for parallel processing.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys "recordings", "supervisions", and "cuts".

    .. caution::
        The `normalize_text` option removes all punctuation and converts all upper case to lower case.
        This includes removing possibly important punctuations such as dashes and apostrophes.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    audio_dir = (
        corpus_dir if (corpus_dir / "train").is_dir() else corpus_dir / "spgispeech"
    )

    dataset_parts = ["train", "val"]
    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for part in dataset_parts:
        logging.info(f"Processing SPGISpeech subset: {part}")
        if manifests_exist(part=part, output_dir=output_dir, prefix="spgispeech"):
            logging.info(f"SPGISpeech subset: {part} already prepared - skipping.")
            manifests.update(
                read_manifests_if_cached(
                    dataset_parts=[part],
                    output_dir=output_dir,
                    prefix="spgispeech",
                    lazy=True,
                )
            )
            continue

        utterances = tqdm(
            parallel_map(
                _read_utterance,
                _iter_transcripts(
                    corpus_dir / f"{part}.csv", audio_dir / part, normalize_text
                ),
                num_jobs=num_jobs,
            ),
            desc="Processing SPGISpeech utterances",
        )

        if output_dir is None:
            recordings, supervisions = zip(*utterances)
            manifests[part] = {
                "recordings": RecordingSet.from_recordings(recordings),
                "supervisions": SupervisionSet.from_segments(supervisions),
            }
            manifests[part]["cuts"] = create_cuts_per_recording(
                manifests[part]["recordings"], manifests[part]["supervisions"]
            )
            continue

        paths = {
            name: output_dir / f"spgispeech_{name}_{part}.jsonl.gz"
            for name in ("recordings", "supervisions", "cuts")
        }
        with (
            RecordingSet.open_writer(paths["recordings"], overwrite=True) as rec_writer,
            SupervisionSet.open_writer(
                paths["supervisions"], overwrite=True
            ) as sup_writer,
            CutSet.open_writer(paths["cuts"], overwrite=True) as cut_writer,
        ):
            for recording, segment in utterances:
                rec_writer.write(recording)
                sup_writer.write(segment)
                cut = recording.to_cut()
                cut.supervisions = [segment]
                cut_writer.write(cut)
        manifests[part] = {
            "recordings": RecordingSet.from_jsonl_lazy(paths["recordings"]),
            "supervisions": SupervisionSet.from_jsonl_lazy(paths["supervisions"]),
            "cuts": CutSet.from_jsonl_lazy(paths["cuts"]),
        }

    if output_dir is not None:
        logging.warning(
            "Manifests are lazily materialized. You may want to call `lhotse.qa.fix_manifests()`"
            " to ensure that all supervisions fall within the corresponding recordings."
        )
    return manifests
