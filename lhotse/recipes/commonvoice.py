"""
Official description from the "about" page of the Mozilla CommonVoice project
(source link: https://commonvoice.mozilla.org/en/about)

Why Common Voice?
Mozilla Common Voice is an initiative to help teach machines how real people speak.
This project is an effort to bridge the digital speech divide. Voice recognition technologies bring a human dimension to our devices, but developers need an enormous amount of voice data to build them. Currently, most of that data is expensive and proprietary. We want to make voice data freely and publicly available, and make sure the data represents the diversity of real people. Together we can make voice recognition better for everyone.

How does it work?
We’re crowdsourcing an open-source dataset of voices. Donate your voice, validate the accuracy of other people’s clips, make the dataset better for everyone.
"""
import logging
import shutil
import tarfile
import warnings
from collections import defaultdict
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
COMMONVOICE_SPLITS = ("train", "dev", "test", "validated", "invalidated", "other")
COMMONVOICE_DEFAULT_SPLITS = ("train", "dev", "test")

# TODO: a list of mapping from language codes (e.g., "en") to actual language names (e.g., "US English")
COMMONVOICE_CODE2LANG = {}


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
    # note(pzelasko): This code should work in general if we supply the right URL,
    # but the URL stopped working during the development of this script --
    # I'm not going to fight this, maybe somebody else would be interested to pick it up.
    raise NotImplementedError(
        "CommonVoice requires you to enter e-mail to download the data"
        "-- please download it manually for now. "
        "We are open to contributions to support downloading CV via lhotse."
    )
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
    output_dir: Pathlike,
    languages: Union[str, Sequence[str]] = "auto",
    splits: Union[str, Sequence[str]] = COMMONVOICE_DEFAULT_SPLITS,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    This function expects the input directory structure of::

        >>> metadata_path = corpus_dir / language_code / "{train,dev,test}.tsv"
        >>> # e.g. pl_train_metadata_path = "/path/to/cv-corpus-7.0-2021-07-21/pl/train.tsv"
        >>> audio_path = corpus_dir / language_code / "clips"
        >>> # e.g. pl_audio_path = "/path/to/cv-corpus-7.0-2021-07-21/pl/clips"

    Returns a dict with 3-level structure (lang -> split -> manifest-type)::

        >>> {'en/fr/pl/...': {'train/dev/test': {'recordings/supervisions': manifest}}}

    :param corpus_dir: Pathlike, the path to the downloaded corpus.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param languages: 'auto' (prepare all discovered data) or a list of language codes.
    :param splits: by default ``['train', 'dev', 'test']``, can also include
        ``'validated'``, ``'invalidated'``, and ``'other'``.
    :param num_jobs: How many concurrent workers to use for scanning of the audio files.
    :return: a dict with manifests for all specified languagues and their train/dev/test splits.
    """
    if not is_module_available("pandas"):
        raise ValueError(
            "To prepare CommonVoice data, please 'pip install pandas' first."
        )
    if num_jobs > 1:
        warnings.warn(
            "num_jobs>1 currently not supported for CommonVoice data prep;"
            "setting to 1."
        )

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    assert output_dir is not None, (
        "CommonVoice recipe requires to specify the output "
        "manifest directory (output_dir cannot be None)."
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    for lang in tqdm(languages, desc="Processing CommonVoice languages"):
        logging.info(f"Language: {lang}")
        lang_path = corpus_dir / lang

        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        # Pattern: "cv_recordings_en_train.jsonl.gz" / "cv_supervisions_en_train.jsonl.gz"
        lang_manifests = read_cv_manifests_if_cached(
            output_dir=output_dir, language=lang
        )

        for part in splits:
            logging.info(f"Split: {part}")
            if part in lang_manifests:
                logging.info(
                    f"CommonVoice language: {lang} already prepared - skipping."
                )
                continue
            recording_set, supervision_set = prepare_single_commonvoice_tsv(
                lang=lang,
                part=part,
                output_dir=output_dir,
                lang_path=lang_path,
            )
            lang_manifests[part] = {
                "supervisions": supervision_set,
                "recordings": recording_set,
            }

        manifests[lang] = lang_manifests

    return manifests


def prepare_single_commonvoice_tsv(
    lang: str,
    part: str,
    output_dir: Pathlike,
    lang_path: Pathlike,
) -> Tuple[RecordingSet, SupervisionSet]:
    """
    Prepares part of CommonVoice data from a single TSV file.

    :param lang: string language code (e.g., "en").
    :param part: which split to prepare (e.g., "train", "validated", etc.).
    :param output_dir: path to directory where we will store the manifests.
    :param lang_path: path to a CommonVoice directory for a specific language
        (e.g., "/path/to/cv-corpus-7.0-2021-07-21/pl").
    :return: a tuple of (RecordingSet, SupervisionSet) objects opened in lazy mode,
        as CommonVoice manifests may be fairly large in memory.
    """
    if not is_module_available("pandas"):
        raise ValueError(
            "To prepare CommonVoice data, please 'pip install pandas' first."
        )
    import pandas as pd

    lang_path = Path(lang_path)
    output_dir = Path(output_dir)
    tsv_path = lang_path / f"{part}.tsv"

    # Read the metadata
    df = pd.read_csv(tsv_path, sep="\t")
    # Scan all the audio files
    with RecordingSet.open_writer(
        output_dir / f"cv_recordings_{lang}_{part}.jsonl.gz",
        overwrite=False,
    ) as recs_writer, SupervisionSet.open_writer(
        output_dir / f"cv_supervisions_{lang}_{part}.jsonl.gz",
        overwrite=False,
    ) as sups_writer:
        for idx, row in tqdm(
            df.iterrows(),
            desc="Processing audio files",
            total=len(df),
        ):
            try:
                result = parse_utterance(row, lang_path, lang)
                if result is None:
                    continue
                recording, segment = result
                validate_recordings_and_supervisions(recording, segment)
                recs_writer.write(recording)
                sups_writer.write(segment)
            except Exception as e:
                logging.error(
                    f"Error when processing TSV file: line no. {idx}: '{row}'.\n"
                    f"Original error type: '{type(e)}' and message: {e}"
                )
                continue
    recordings = RecordingSet.from_jsonl_lazy(recs_writer.path)
    supervisions = SupervisionSet.from_jsonl_lazy(sups_writer.path)
    return recordings, supervisions


def parse_utterance(
    row: Any, lang_path: Path, language: str
) -> Tuple[Recording, SupervisionSegment]:
    # Create the Recording first
    audio_path = lang_path / "clips" / row.path
    if not audio_path.is_file():
        raise ValueError(f"No such file: {audio_path}")
    recording_id = Path(row.path).stem
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        # Look up language code => language name mapping (it is empty at the time of writing this comment)
        # if the language code is unknown, fall back to using the language code.
        language=COMMONVOICE_CODE2LANG.get(language, language),
        speaker=row.client_id,
        text=row.sentence.strip(),
        gender=row.gender if row.gender != "nan" else None,
        custom={
            "age": row.age if row.age != "nan" else None,
            "accent": row.accent if row.accent != "nan" else None,
        },
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
    manifests = defaultdict(dict)
    for part in ["train", "dev", "test"]:
        for manifest in ["recordings", "supervisions"]:
            path = output_dir / f"cv_{manifest}_{language}_{part}.jsonl.gz"
            if not path.is_file():
                continue
            manifests[part][manifest] = load_manifest(path)
    return manifests
