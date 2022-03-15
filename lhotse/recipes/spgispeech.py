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
from typing import Dict, Union

from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.parallel import parallel_map
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSet, SupervisionSegment
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


def prepare_spgispeech(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    normalize_text: bool = True,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param normalize_text: Bool, if True, normalize the text (similar to ESPNet recipe).
    :param num_jobs: int, the number of jobs to use for parallel processing.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.

    .. note::
        Unlike other recipes, output_dir is not Optional here because we write the manifests
        to the output directory while processing to avoid OOM issues, since it is a large dataset.

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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Maybe the manifests already exist: we can read them and save a bit of preparation time.
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=output_dir,
        prefix="spgispeech",
        suffix="jsonl.gz",
        lazy=True,
    )

    for part in dataset_parts:
        logging.info(f"Processing SPGISpeech subset: {part}")
        if manifests_exist(
            part=part, output_dir=output_dir, prefix="spgispeech", suffix="jsonl.gz"
        ):
            logging.info(f"SPGISpeech subset: {part} already prepared - skipping.")
            continue

        # Read the recordings and write them into manifest. We additionally store the
        # duration of the recordings in a dict which will be used later to create the
        # supervisions.
        global audio_read_worker
        durations = {}

        def audio_read_worker(p: Path) -> Recording:
            r = Recording.from_file(p, recording_id=f"{p.parent.stem}_{p.stem}")
            durations[r.id] = r.duration
            return r

        with RecordingSet.open_writer(
            output_dir / f"spgispeech_recordings_{part}.jsonl.gz"
        ) as rec_writer:
            for recording in tqdm(
                parallel_map(
                    audio_read_worker,
                    (audio_dir / part).rglob("*.wav"),
                    num_jobs=num_jobs,
                ),
                desc="Processing SPGISpeech recordings",
            ):
                rec_writer.write(recording)

        # Read supervisions and write them to manifest
        with SupervisionSet.open_writer(
            output_dir / f"spgispeech_supervisions_{part}.jsonl.gz"
        ) as sup_writer, open(corpus_dir / f"{part}.csv", "r") as f:
            # Skip the header
            next(f)
            for line in tqdm(f, desc="Processing utterances"):
                parts = line.strip().split("|")
                # 07a785e9237c389c1354bb60abca42d5/1.wav -> 07a785e9237c389c1354bb60abca42d5_1
                recording_id = parts[0].replace("/", "_").replace(".wav", "")
                text = parts[2]
                if normalize_text:
                    text = normalize(text)
                spkid = recording_id.split("_")[0]
                segment = SupervisionSegment(
                    id=recording_id,
                    recording_id=recording_id,
                    text=text,
                    speaker=spkid,
                    start=0,
                    duration=durations[recording_id],
                    language="English",
                )
                sup_writer.write(segment)

        manifests[part] = {
            "recordings": RecordingSet.from_jsonl_lazy(rec_writer.path),
            "supervisions": SupervisionSet.from_jsonl_lazy(sup_writer.path),
        }

    return manifests
