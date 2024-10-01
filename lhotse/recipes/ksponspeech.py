"""
KsponSpeech is a large-scale spontaneous speech corpus of Korean.
This corpus contains 969 hours of open-domain dialogue utterances,
spoken by about 2,000 native Korean speakers in a clean environment.

All data were constructed by recording the dialogue between two people
freely conversing on a variety of topics and manually transcribing the utterances.

The transcription provides a dual transcription consisting of orthography and pronunciation,
and disfluency tags for the spontaneity of speech, such as filler words, repeated words, and word fragments.

The original audio data has a PCM extension.
During preprocessing, it is converted into a file in the FLAC extension and saved anew.

KsponSpeech is publicly available on an open data hub site of the Korea government.
The dataset must be downloaded manually.

For more details, please visit:

Dataset: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123
Paper: https://www.mdpi.com/2076-3417/10/19/6936
"""

import logging
import re
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import soundfile as sf
from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

KSPONSPEECH = (
    "train",
    "dev",
    "eval_clean",
    "eval_other",
)


def normalize(
    raw_content: str,
    normalize_text: str = "default",
) -> Tuple[str, str]:
    """
    Normalizing KsponSpeech text datasets with '.trn' extension.
    Perform the following processing.

    1. Separate file name and text labeling from raw content using separator ' :: ';
    2. Remove noise labeling characters (e.g. `o/`, `b/`...);
    3. Remove the actual pronunciation from the text labeling; use the spelling content;
    4. Remove other special characters and double spaces from text labeling.

    :param raw_content: a raw text labeling content containing file name and text labeling.
    :param normalize_text: str, the text normalization type, "default" or "none".
    :return: a tuple with file name and normalized text labeling.
    """
    if len(raw_content) == 0:
        return ""

    original_content_id, content = raw_content.split(" :: ")

    if normalize_text == "none":
        return original_content_id, content

    elif normalize_text == "default":
        content = re.sub(r"[a-z]/", "", content)
        content = re.sub(r"\((.*?)\)/\((.*?)\)", r"\1", content)
        content = content.replace("*", "")
        content = content.replace("+", "")
        content = content.replace("/", "")
        content = re.sub(r"\s+", " ", content)

        return original_content_id, content.strip()


def prepare_ksponspeech(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "all",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
    normalize_text: str = "default",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train', 'dev'.
        By default, we will infer all parts.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: int, number of parallel threads used for 'parse_utterance' calls.
    :param normalize_text: str, the text normalization type, "default" or "none".
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if dataset_parts == "all":
        dataset_parts = set(KSPONSPEECH)

    elif isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts,
            output_dir=output_dir,
            prefix="ksponspeech",
            suffix="jsonl.gz",
            lazy=True,
        )

    with ThreadPoolExecutor(num_jobs) as ex:
        for part in tqdm(dataset_parts, desc="Dataset parts"):
            logging.info(f"Processing KsponSpeech subset: {part}")
            if manifests_exist(
                part=part,
                output_dir=output_dir,
                prefix="ksponspeech",
                suffix="jsonl.gz",
            ):
                logging.info(f"KsponSpeech subset: {part} already prepared - skipping.")
                continue

            recordings = []
            supervisions = []
            futures = []

            trans_path = corpus_dir / f"{part}.trn"
            with open(trans_path) as f:
                for line in f:
                    futures.append(
                        ex.submit(
                            parse_utterance, corpus_dir, part, line, normalize_text
                        )
                    )

            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.append(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            recording_set, supervision_set = fix_manifests(
                recording_set, supervision_set
            )
            validate_recordings_and_supervisions(recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_file(
                    output_dir / f"ksponspeech_supervisions_{part}.jsonl.gz"
                )
                recording_set.to_file(
                    output_dir / f"ksponspeech_recordings_{part}.jsonl.gz"
                )

            manifests[part] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }

    return manifests


def pcm_to_flac(
    pcm_path: Union[str, Path],
    flac_path: Union[str, Path],
    sample_rate: Optional[int] = 16000,
    channels: Optional[int] = 1,
    bit_depth: Optional[int] = 16,
) -> Path:
    # typecasting
    pcm_path = Path(pcm_path)
    flac_path = Path(flac_path)

    data, _ = sf.read(
        pcm_path,
        channels=channels,
        samplerate=sample_rate,
        format="RAW",
        subtype="PCM_16",
    )

    sf.write(flac_path, data, sample_rate, format="FLAC")
    return flac_path


def parse_utterance(
    corpus_dir: Pathlike,
    part: str,
    line: str,
    normalize_text: str = "default",
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    corpus_dir = Path(corpus_dir)
    audio_path, normalized_line = normalize(line, normalize_text)
    if "eval" in part:
        audio_path = audio_path.split("/", maxsplit=1)[1]

    audio_path = corpus_dir / audio_path
    recording_id = audio_path.stem

    # Create the Recording first
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None
    flac_path = audio_path.with_suffix(".flac")
    flac_path = pcm_to_flac(audio_path, flac_path)
    recording = Recording.from_file(flac_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="Korean",
        text=normalized_line,
    )
    return recording, segment
