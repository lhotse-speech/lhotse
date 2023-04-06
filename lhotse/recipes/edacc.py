"""
The Edinburgh International Accents of English Corpus

Citation

Sanabria, Ramon; Markl, Nina; Carmantini, Andrea; Klejch, Ondrej; Bell, Peter; Bogoychev, Nikolay. (2023). The Edinburgh International Accents of English Corpus, [dataset]. University of Edinburgh. School of Informatics. The Institute for Language, Cognition and Computation. The Centre for Speech Technology Research. https://doi.org/10.7488/ds/3832.

Description

English is the most widely spoken language in the world, used daily by millions of people as a first or second language in many different contexts. As a result, there are many varieties of English. Although the great many advances in English automatic speech recognition (ASR) over the past decades, results are usually reported based on test datasets which fail to represent the diversity of English as spoken today around the globe. We present the first release of The Edinburgh International Accents of English Corpus (EdAcc). This dataset attempts to better represent the wide diversity of English, encompassing almost 40 hours of dyadic video call conversations between friends. Unlike other datasets, EdAcc includes a wide range of first and second-language varieties of English and a linguistic background profile of each speaker. Results on latest public, and commercial models show that EdAcc highlights shortcomings of current English ASR models. The best performing model, trained on 680 thousand hours of transcribed data, obtains an average of 19.7% WER -- in contrast to the the 2.7% WER obtained when evaluated on US English clean read speech. Across all models, we observe a drop in performance on Jamaican, Indonesian, Nigerian, and Kenyan English speakers. Recordings, linguistic backgrounds, data statement, and evaluation scripts are released on our website under CC-BY-SA.

Source: https://datashare.ed.ac.uk/handle/10283/4836
"""
import logging
import re
import shutil
import tarfile
import zipfile
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
    is_module_available,
    safe_extract,
    urlretrieve_progress,
)


def download_edacc(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    base_url: str = "https://datashare.ed.ac.uk/download/",
) -> Path:
    """
    Download and extract the EDACC dataset.

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param force_download: Bool, if True, download the data even if it exists.
    :param base_url: str, the url of the website used to fetch the archive from.
    :return: the path to downloaded and extracted directory with data.
    """
    archive_name = "DS_10283_4836.zip"

    target_dir = Path(target_dir)
    corpus_dir = target_dir / "edacc"
    target_dir.mkdir(parents=True, exist_ok=True)

    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping EDACC download because {completed_detector} exists.")
        return corpus_dir

    # Maybe-download the archive.
    archive_path = target_dir / archive_name
    if force_download or not archive_path.is_file():
        urlretrieve_progress(
            f"{base_url}/{archive_name}",
            filename=archive_path,
            desc=f"Downloading {archive_name}",
        )

    # Remove partial unpacked files, if any, and unpack everything.
    shutil.rmtree(corpus_dir, ignore_errors=True)
    with zipfile.ZipFile(archive_path) as zip:
        zip.extractall(path=corpus_dir)
    completed_detector.touch()

    return corpus_dir


def prepare_edacc(
    corpus_dir: Pathlike,
    alignments_dir: Optional[Pathlike] = None,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param alignments_dir: Pathlike, the path of the alignments dir. By default, it is
        the same as ``corpus_dir``.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    raise NotImplementedError()
    # corpus_dir = Path(corpus_dir)
    # alignments_dir = Path(alignments_dir) if alignments_dir is not None else corpus_dir
    # assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    #
    # if dataset_parts == "mini_librispeech":
    #     dataset_parts = set(MINI_LIBRISPEECH).intersection(
    #         path.name for path in corpus_dir.glob("*")
    #     )
    # elif dataset_parts == "auto":
    #     dataset_parts = (
    #         set(LIBRISPEECH)
    #         .union(MINI_LIBRISPEECH)
    #         .intersection(path.name for path in corpus_dir.glob("*"))
    #     )
    #     if not dataset_parts:
    #         raise ValueError(
    #             f"Could not find any of librispeech or mini_librispeech splits in: {corpus_dir}"
    #         )
    # elif isinstance(dataset_parts, str):
    #     dataset_parts = [dataset_parts]
    #
    # manifests = {}
    #
    # if output_dir is not None:
    #     output_dir = Path(output_dir)
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     # Maybe the manifests already exist: we can read them and save a bit of preparation time.
    #     manifests = read_manifests_if_cached(
    #         dataset_parts=dataset_parts, output_dir=output_dir
    #     )
    #
    # with ThreadPoolExecutor(num_jobs) as ex:
    #     for part in tqdm(dataset_parts, desc="Dataset parts"):
    #         logging.info(f"Processing LibriSpeech subset: {part}")
    #         if manifests_exist(part=part, output_dir=output_dir):
    #             logging.info(f"LibriSpeech subset: {part} already prepared - skipping.")
    #             continue
    #         recordings = []
    #         supervisions = []
    #         part_path = corpus_dir / part
    #         futures = []
    #         for trans_path in tqdm(
    #             part_path.rglob("*.trans.txt"), desc="Distributing tasks", leave=False
    #         ):
    #             alignments = {}
    #             ali_path = (
    #                 alignments_dir
    #                 / trans_path.parent.relative_to(corpus_dir)
    #                 / (trans_path.stem.split(".")[0] + ".alignment.txt")
    #             )
    #             if ali_path.exists():
    #                 alignments = parse_alignments(ali_path)
    #             # "trans_path" file contains lines like:
    #             #
    #             #   121-121726-0000 ALSO A POPULAR CONTRIVANCE
    #             #   121-121726-0001 HARANGUE THE TIRESOME PRODUCT OF A TIRELESS TONGUE
    #             #   121-121726-0002 ANGOR PAIN PAINFUL TO HEAR
    #             #
    #             # We will create a separate Recording and SupervisionSegment for those.
    #             with open(trans_path) as f:
    #                 for line in f:
    #                     futures.append(
    #                         ex.submit(parse_utterance, part_path, line, alignments)
    #                     )
    #
    #         for future in tqdm(futures, desc="Processing", leave=False):
    #             result = future.result()
    #             if result is None:
    #                 continue
    #             recording, segment = result
    #             recordings.append(recording)
    #             supervisions.append(segment)
    #
    #         recording_set = RecordingSet.from_recordings(recordings)
    #         supervision_set = SupervisionSet.from_segments(supervisions)
    #
    #         validate_recordings_and_supervisions(recording_set, supervision_set)
    #
    #         if output_dir is not None:
    #             supervision_set.to_file(
    #                 output_dir / f"librispeech_supervisions_{part}.jsonl.gz"
    #             )
    #             recording_set.to_file(
    #                 output_dir / f"librispeech_recordings_{part}.jsonl.gz"
    #             )
    #
    #         manifests[part] = {
    #             "recordings": recording_set,
    #             "supervisions": supervision_set,
    #         }
    #
    # return manifests
