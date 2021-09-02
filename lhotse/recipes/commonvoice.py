import logging
import shutil
import tarfile
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import load_manifest, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, urlretrieve_progress

DEFAULT_COMMONVOICE_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com"
DEFAULT_COMMONVOICE_RELEASE = "cv-corpus-5.1-2020-06-22"


COMMONVOICE_LANGS = "en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk".split()
COMMONVOICE_SPLITS = ("train", "dev", "test")

# TODO: a list of mapping from language codes (e.g., "en") to actual language names (e.g., "US English")


def download_commonvoice(
    target_dir: Pathlike = ".",
    languages: Union[str, Iterable[str]] = "all",
    force_download: bool = False,
    base_url: str = DEFAULT_COMMONVOICE_URL,
    release: str = DEFAULT_COMMONVOICE_RELEASE,
) -> None:
    """
    Download and untar the CommonVoice dataset.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param languages: one of: 'all' (downloads all known languages); a single language code (e.g., 'en'),
        or a list of language codes.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the base URL for CommonVoice.
    :param release: str, the name of the CommonVoice release (e.g., "cv-corpus-5.1-2020-06-22").
        It is used as part of the download URL.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}/{release}"

    if languages == "all":
        languages = COMMONVOICE_LANGS
    elif isinstance(languages, str):
        languages = [languages]
    else:
        languages = list(languages)

    logging.info(
        f"About to download {len(languages)} CommonVoice languages: {languages}"
    )
    for lang in tqdm(languages, desc="Downloading CommonVoice languages"):
        logging.info(f"Language: {lang}")
        # Split directory exists and seem valid? Skip this split.
        part_dir = target_dir / release / lang
        completed_detector = part_dir / ".completed"
        if completed_detector.is_file():
            logging.info(f"Skipping {lang} because {completed_detector} exists.")
            continue
        # Maybe-download the archive.
        tar_name = f"{lang}.tar.gz"
        tar_path = target_dir / tar_name
        if force_download or not tar_path.is_file():
            urlretrieve_progress(url, filename=tar_path, desc=f"Downloading {tar_name}")
            logging.info(f"Downloading finished: {lang}")
        # Remove partial unpacked files, if any, and unpack everything.
        logging.info(f"Unpacking archive: {lang}")
        shutil.rmtree(part_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=target_dir)
        completed_detector.touch()


def prepare_commonvoice(
    corpus_dir: Pathlike,
    languages: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    Returns a dict with 3-level structure (lang -> split -> manifest-type)::

        >>> {'en/fr/pl/...': {'train/dev/test': {'recordings/supervisions': manifest}}}

    :param corpus_dir: Pathlike, the path to the downloaded corpus.
    :param languages: 'auto' (prepare all discovered data) or a list of language codes.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: How many concurrent workers to use for scanning of the audio files.
    :return: a dict with manifests for all specified languagues and their train/dev/test splits.
    """
    if not is_module_available("pandas"):
        raise ValueError(
            "To prepare CommonVoice data, please 'pip install pandas' first."
        )
    import pandas as pd

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if languages == "auto":
        languages = set(COMMONVOICE_LANGS).intersection(
            path.name for path in corpus_dir.glob("*")
        )
        if not languages:
            raise ValueError(
                f"Could not find any of CommonVoice languages in: {corpus_dir}"
            )
    elif isinstance(languages, str):
        languages = [languages]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(num_jobs) as ex:
        for lang in tqdm(languages, desc="Processing CommonVoice languages"):
            logging.info(f"Language: {lang}")
            recordings = []
            supervisions = []
            part_path = corpus_dir / lang

            # Maybe the manifests already exist: we can read them and save a bit of preparation time.
            # Pattern: "cv_recordings_en_train.jsonl.gz" / "cv_supervisions_en_train.jsonl.gz"
            lang_manifests = read_cv_manifests_if_cached(
                output_dir=output_dir, language=lang
            )

            for part in COMMONVOICE_SPLITS:
                logging.info(f"Split: {part}")

                if part in lang_manifests:
                    logging.info(
                        f"CommonVoice language: {lang} already prepared - skipping."
                    )
                    continue

                # Read the metadata
                df = pd.read_csv(part_path / f"{part}.tsv", sep="\t")

                # Scan all the audio files
                futures = []
                for idx, row in tqdm(
                    df.iterrows(), desc="Processing audio files", leave=False
                ):
                    futures.append(ex.submit(parse_utterance, row, part_path, lang))

                for future in tqdm(futures, desc="Collecting results", leave=False):
                    result = future.result()
                    if result is None:
                        continue
                    recording, segment = result
                    recordings.append(recording)
                    supervisions.append(segment)

                recording_set = RecordingSet.from_recordings(recordings)
                supervision_set = SupervisionSet.from_segments(supervisions)

                validate_recordings_and_supervisions(recording_set, supervision_set)

                if output_dir is not None:
                    supervision_set.to_file(
                        output_dir / f"cv_supervisions_{lang}_{part}.jsonl.gz"
                    )
                    recording_set.to_file(
                        output_dir / f"cv_recordings_{lang}_{part}.jsonl.gz"
                    )

                lang_manifests[part] = {
                    'supervisions': supervision_set,
                    'recordings': recording_set
                }

            manifests[lang] = lang_manifests

    return manifests


def parse_utterance(
    row: Any, dataset_split_path: Path, language: str
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    # Create the Recording first
    audio_path = dataset_split_path / "clips" / row.path
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None
    recording_id = Path(row.path).stem
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=language,
        speaker=row.client_id,
        text=row.sentence.strip(),
        gender=row.gender,
        custom={"age": row.age, "accent": row.accent},
    )
    return recording, segment


def read_cv_manifests_if_cached(
    output_dir: Optional[Pathlike],
    language: str,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns:
        {'train': {'recordings': ..., 'supervisions': ...}, 'dev': ..., 'test': ...}
    """
    if output_dir is None:
        return {}
    manifests = {}
    for part in ["train", "dev", "test"]:
        for manifest in ["recordings", "supervisions"]:
            path = output_dir / f"cv_{manifest}_{language}_{part}.jsonl.gz"
            if not path.is_file():
                continue
            manifests[part][manifest] = load_manifest(path)
    return manifests
