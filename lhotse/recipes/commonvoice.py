import logging
import re
import shutil
import tarfile
import zipfile
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, urlretrieve_progress


DEFAULT_COMMONVOICE_URL = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22"

COMMONVOICE_LANGS = "en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk".split()


def download_commonvoice(
        target_dir: Pathlike = '.',
        languages: Union[str, Iterable[str]] = 'all',
        force_download: bool = False,
        base_url: str = DEFAULT_COMMONVOICE_URL,
) -> None:
    """
    Download and untar the CommonVoice dataset.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param languages: one of: 'all' (downloads all known languages); a single language code (e.g., 'en'),
        or a list of language codes.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the base URL for CommonVoice.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if languages == "all":
        languages = COMMONVOICE_LANGS
    elif isinstance(languages, str):
        languages = [languages]
    else:
        languages = list(languages)

    logging.info(f'About to download {len(languages)} CommonVoice languages: {languages}')
    for lang in tqdm(languages, desc='Downloading CommonVoice languages'):
        logging.info(f'Language: {lang}')
        # Split directory exists and seem valid? Skip this split.
        part_dir = target_dir / 'CommonVoice' / lang
        completed_detector = part_dir / '.completed'
        if completed_detector.is_file():
            logging.info(f'Skipping {lang} because {completed_detector} exists.')
            continue
        # Maybe-download the archive.
        tar_name = f'{lang}.tar.gz'
        tar_path = target_dir / tar_name
        if force_download or not tar_path.is_file():
            urlretrieve_progress(f'{base_url}/{tar_name}', filename=tar_path, desc=f'Downloading {tar_name}')
        # Remove partial unpacked files, if any, and unpack everything.
        shutil.rmtree(part_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=target_dir)
        completed_detector.touch()


def prepare_commonvoice(
        corpus_dir: Pathlike,
        languages: Union[str, Sequence[str]] = 'auto',
        output_dir: Optional[Pathlike] = None,
        num_jobs: int = 1
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param languages: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f'No such directory: {corpus_dir}'

    if languages == 'auto':
        languages = set(COMMONVOICE_LANGS).intersection(path.name for path in corpus_dir.glob('*'))
        if not languages:
            raise ValueError(f"Could not find any of CommonVoice languages in: {corpus_dir}")
    elif isinstance(languages, str):
        languages = [languages]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        # Pattern: "cv_recordings_en.jsonl.gz" / "cv_supervisions_en.jsonl.gz"
        manifests = read_manifests_if_cached(
            dataset_parts=languages,
            output_dir=output_dir,
            prefix="cv",
            suffix=".jsonl.gz",
        )

    with ThreadPoolExecutor(num_jobs) as ex:
        for lang in tqdm(languages, desc='Processing CommonVoice languages'):
            logging.info(f'Language: {lang}')
            if manifests_exist(part=lang, output_dir=output_dir):
                logging.info(f'CommonVoice language: {lang} already prepared - skipping.')
                continue
            recordings = []
            supervisions = []
            part_path = corpus_dir / lang
            futures = []
            for trans_path in tqdm(part_path.rglob('*.trans.txt'), desc='Distributing tasks', leave=False):
                alignments = {}
                ali_path = trans_path.parent / (trans_path.stem.split('.')[0] + '.alignment.txt')
                print(ali_path)
                if ali_path.exists():
                    alignments = parse_alignments(ali_path)
                # "trans_path" file contains lines like:
                #
                #   121-121726-0000 ALSO A POPULAR CONTRIVANCE
                #   121-121726-0001 HARANGUE THE TIRESOME PRODUCT OF A TIRELESS TONGUE
                #   121-121726-0002 ANGOR PAIN PAINFUL TO HEAR
                #
                # We will create a separate Recording and SupervisionSegment for those.
                with open(trans_path) as f:
                    for line in f:
                        futures.append(ex.submit(parse_utterance, part_path, line, alignments))

            for future in tqdm(futures, desc='Processing', leave=False):
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
                supervision_set.to_json(output_dir / f'supervisions_{lang}.json')
                recording_set.to_json(output_dir / f'recordings_{lang}.json')

            manifests[lang] = {
                'recordings': recording_set,
                'supervisions': supervision_set
            }

    return manifests


def parse_utterance(
        dataset_split_path: Path,
        line: str,
        alignments: Dict[str, List[AlignmentItem]],
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recording_id, text = line.strip().split(maxsplit=1)
    # Create the Recording first
    audio_path = dataset_split_path / Path(recording_id.replace('-', '/')).parent / f'{recording_id}.flac'
    if not audio_path.is_file():
        logging.warning(f'No such file: {audio_path}')
        return None
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language='English',
        speaker=re.sub(r'-.*', r'', recording.id),
        text=text.strip(),
        alignment={"word": alignments[recording_id]} if recording_id in alignments else None
    )
    return recording, segment


def parse_alignments(ali_path: Pathlike) -> Dict[str, List[AlignmentItem]]:
    alignments = {}
    for line in Path(ali_path).read_text().splitlines():
        utt_id, words, timestamps = line.split()
        words = words.replace('"', '').split(',')
        timestamps = [0.0] + list(map(float, timestamps.replace('"', '').split(',')))
        alignments[utt_id] = [
            AlignmentItem(symbol=word, start=start, duration=round(end - start, ndigits=8))
            for word, start, end in zip(words, timestamps, timestamps[1:])
        ]
    return alignments
