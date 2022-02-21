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
from concurrent.futures.thread import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSet, SupervisionSegment
from lhotse.utils import Pathlike


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


def normalize_text(text: str) -> str:
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert all upper case to lower case
    text = text.lower()
    return text


def prepare_spgispeech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
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
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    dataset_parts = ["train", "val"]
    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir
        )

    for part in tqdm(dataset_parts, desc="Dataset parts"):
        logging.info(f"Processing SPGISpeech subset: {part}")
        if manifests_exist(part=part, output_dir=output_dir):
            logging.info(f"SPGISpeech subset: {part} already prepared - skipping.")
            continue

        # Get recordings
        futures = []
        with ProcessPoolExecutor(num_jobs) as ex:
            for wav in tqdm(
                corpus_dir.rglob(f"{part}/*/*.wav"),
                desc=f"Processing {part} recordings",
            ):
                futures.append(
                    ex.submit(
                        Recording.from_file,
                        wav,
                        recording_id=f"{wav.parent.stem}_{wav.stem}",
                    )
                )

        for future in tqdm(futures, desc=f"Processing {part} recordings", leave=False):
            recordings = future.result()

        recording_set = RecordingSet.from_recordings(recordings)

        supervisions = []
        with open(corpus_dir / f"{part}.csv", "r") as f:
            # Skip the header
            next(f)
            for line in f:
                parts = line.strip().split("|")
                # 07a785e9237c389c1354bb60abca42d5/1.wav -> 07a785e9237c389c1354bb60abca42d5_1
                recording_id = parts[0].replace("/", "_").replace(".wav", "")
                text = parts[2]
                if normalize_text:
                    text = normalize_text(text)
                spkid = recording_id.split("_")[0]
                supervisions.append(
                    SupervisionSegment(
                        id=recording_id,
                        recording_id=recording_id,
                        text=text,
                        speaker=spkid,
                        start=0,
                        duration=recording_set[recording_id].duration,
                        language="English",
                    )
                )

        supervision_set = SupervisionSet.from_segments(supervisions)

        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(output_dir / f"supervisions_{part}.json")
            recording_set.to_file(output_dir / f"recordings_{part}.json")

        manifests[part] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests
