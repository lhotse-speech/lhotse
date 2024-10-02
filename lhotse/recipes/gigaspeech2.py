"""
Description taken from the abstract of paper:
"GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages with Automated Crawling, Transcription and Refinement"
https://arxiv.org/abs/2406.11546

The evolution of speech technology has been spurred by the rapid increase in dataset sizes. Traditional speech models generally depend on a large amount of labeled training data, which is scarce for low-resource languages. This paper presents GigaSpeech 2, a large-scale, multi-domain, multilingual speech recognition corpus. It is designed for low-resource languages and does not rely on paired speech and text data. GigaSpeech 2 comprises about 30,000 hours of automatically transcribed speech, including Thai, Indonesian, and Vietnamese, gathered from unlabeled YouTube videos. We also introduce an automated pipeline for data crawling, transcription, and label refinement. Specifically, this pipeline uses Whisper for initial transcription and TorchAudio for forced alignment, combined with multi-dimensional filtering for data quality assurance. A modified Noisy Student Training is developed to further refine flawed pseudo labels iteratively, thus enhancing model performance. Experimental results on our manually transcribed evaluation set and two public test sets from Common Voice and FLEURS confirm our corpus’s high quality and broad applicability. Notably, ASR models trained on GigaSpeech 2 can reduce the word error rate for Thai, Indonesian, and Vietnamese on our challenging and realistic YouTube test set by 25% to 40% compared to the Whisper large-v3 model, with merely 10% model parameters. Furthermore, our ASR models trained on Gigaspeech 2 yield superior performance compared to commercial services. We believe that our newly introduced corpus and pipeline will open a new avenue for low-resource speech recognition and significantly facilitate research in this area.
"""

import logging
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import load_manifest
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

GIGASPEECH2_URL = "https://huggingface.co/datasets/speechcolab/gigaspeech2"

GIGASPEECH2_LANGS = ("th", "id", "vi")
GIGASPEECH2_SPLITS = ("train_raw", "train_refined", "dev", "test")


def _read_manifests_if_cached(
    output_dir: Optional[Pathlike],
    language: str,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns:
        {
          "train_raw": {"recordings": ..., "supervisions": ...},
          "train_refined": ...,
          "dev": ...,
          "test": ...,
        }
    """
    if output_dir is None:
        return {}
    manifests = defaultdict(dict)
    for part in ["train_raw", "train_refined", "dev", "test"]:
        for manifest in ["recordings", "supervisions"]:
            path = output_dir / f"gigaspeech2-{language}_{manifest}_{part}.jsonl.gz"
            if not path.is_file():
                continue
            manifests[part][manifest] = load_manifest(path)
    return manifests


def _parse_utterance(
    lang: str,
    part_dir: Pathlike,
    audio_info: Pathlike,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    segment_id, text = audio_info.split("\t")
    audio_path = part_dir.joinpath(*segment_id.split("-")[:-1]) / f"{segment_id}.wav"
    audio_path = audio_path.resolve()

    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(
        path=audio_path,
        recording_id=segment_id,
    )

    segment = SupervisionSegment(
        id=segment_id,
        recording_id=segment_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=lang,
        text=text.strip(),
    )

    return recording, segment


def _prepare_subset(
    lang: str,
    part: str,
    lang_dir: Pathlike,
    num_jobs: int = 1,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Returns the RecodingSet and SupervisionSet given a dataset part.

    :param lang: string language code (e.g., "th").
    :param part: str, the name of the subset.
    :param lang_dir: Pathlike, the path of the data dir for a specific language.
    :return: the RecodingSet and SupervisionSet for train and valid.
    """
    lang_dir = Path(lang_dir)
    part_dir = lang_dir / part.replace("_raw", "").replace("_refined", "")
    tsv_path = lang_dir / f"{part}.tsv"

    audio_infos = []
    with open(tsv_path) as f:
        audio_infos = f.read().splitlines()

    with ProcessPoolExecutor(num_jobs) as ex:
        futures = []
        recordings = []
        supervisions = []
        for audio_info in tqdm(audio_infos, desc="Distributing tasks"):
            futures.append(ex.submit(_parse_utterance, lang, part_dir, audio_info))

        for future in tqdm(futures, desc="Processing"):
            result = future.result()
            if result is None:
                continue
            recording, segment = result
            recordings.append(recording)
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)

        # Fix manifests
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

    return recording_set, supervision_set


def prepare_gigaspeech2(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    languages: Union[str, Sequence[str]] = "auto",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the GigaSpeech 2 dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param languages: 'auto' (prepare all discovered data) or a list of language codes.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    corpus_dir = Path(corpus_dir) / "data"

    if languages == "auto":
        languages = set(GIGASPEECH2_LANGS).intersection(
            path.name for path in corpus_dir.glob("*")
        )
        if not languages:
            raise ValueError(
                f"Could not find any of GigaSpeech 2 languages in: {corpus_dir}"
            )
    elif isinstance(languages, str):
        languages = [languages]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for lang in tqdm(languages, desc="Processing GigaSpeech 2 languages"):
        logging.info(f"Language: {lang}")
        lang_dir = corpus_dir / lang

        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        lang_manifests = _read_manifests_if_cached(output_dir=output_dir, language=lang)

        for part in tqdm(GIGASPEECH2_SPLITS, desc="Processing GigaSpeech 2 subset"):
            logging.info(f"Processing GigaSpeech 2 subset: {part}")
            if part in lang_manifests:
                logging.info(f"GigaSpeech 2 {lang} {part} already prepared - skipping.")
                continue

            recording_set, supervision_set = _prepare_subset(
                lang=lang,
                part=part,
                lang_dir=lang_dir,
                num_jobs=num_jobs,
            )

            if output_dir is not None:
                supervision_set.to_file(
                    output_dir / f"gigaspeech2-{lang}_supervisions_{part}.jsonl.gz"
                )
                recording_set.to_file(
                    output_dir / f"gigaspeech2-{lang}_recordings_{part}.jsonl.gz"
                )

            lang_manifests[part] = {
                "supervisions": supervision_set,
                "recordings": recording_set,
            }

        manifests[lang] = lang_manifests
