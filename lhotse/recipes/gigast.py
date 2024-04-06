"""
About GigaST corpus:
GigaST: A large-scale speech translation corpus
https://arxiv.org/abs/2204.03939

GigaST is a large-scale speech translation corpus, by translating the transcriptions in GigaSpeech, a multi-domain English speech recognition corpus with 10,000 hours of labeled audio. The training data is translated by a strong machine translation system and the test data is produced by professional human translators.

Note:
This recipe assume you have downloaded and prepared GigaSpeech by lhotse.
We assure manifests_dir contains the following files:
  - gigaspeech_recordings_TEST.jsonl.gz
  - gigaspeech_recordings_XL.jsonl.gz
  - gigaspeech_supervisions_TEST.jsonl.gz
  - gigaspeech_supervisions_XL.jsonl.gz
"""
import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import CutSet
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available, resumable_download

GIGASPEECH_PARTS = ("XL", "L", "M", "S", "XS", "DEV", "TEST")
GIGAST_LANGS = ("de", "zh")


class GigaST:
    def __init__(self, corpus_dir: Pathlike, lang: str):
        with open(corpus_dir / f"GigaST.{lang}.json") as f:
            self.audio_generator = iter(json.load(f)["audios"])
        self.segment_generator = iter(next(self.audio_generator)["segments"])

    def get_next_line(self):
        try:
            return next(self.segment_generator)
        except StopIteration:
            self.segment_generator = iter(next(self.audio_generator)["segments"])
            return next(self.segment_generator)


def download_gigast(
    target_dir: Pathlike = ".",
    languages: Union[str, Sequence[str]] = "all",
    force_download: bool = False,
) -> Path:
    """
    Download GigaST dataset

    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param languages: one of: 'all' (downloads all known languages); a single language code (e.g., 'en'), or a list of language codes.
    :param force_download: bool, if True, download the archive even if it already exists.

    :return: the path to downloaded with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if languages == "all":
        languages = GIGAST_LANGS
    elif isinstance(languages, str):
        languages = [languages]
    else:
        languages = list(languages)

    for lang in tqdm(languages, desc=f"Downloading GigaST"):
        logging.info(f"Download language: {lang}")
        completed_detector = target_dir / f".{lang}_completed"
        if completed_detector.is_file():
            logging.info(f"Skipping {lang} because {completed_detector} exists.")
            continue
        # Process the archive.
        json_name = f"GigaST.{lang}.json"
        json_path = target_dir / json_name
        resumable_download(
            f"https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/datasets/GigaST/{json_name}",
            filename=json_path,
            force_download=force_download,
        )
        completed_detector.touch()

    return target_dir


def prepare_gigast(
    corpus_dir: Pathlike,
    manifests_dir: Pathlike,
    output_dir: Optional[Pathlike],
    languages: Union[str, Sequence[str]] = "auto",
    dataset_parts: Union[str, Sequence[str]] = "auto",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the GigaST dataset
    :param manifests_dir: Path to the GigaSpeech manifests
    :param output_dir: Pathlike, the path where to write the manifests.
    :param languages: 'auto' (prepare all languages) or a list of language codes.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    manifests_dir = Path(manifests_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing GigaST...")

    languages = GIGAST_LANGS if languages == "auto" else languages
    if isinstance(languages, str):
        languages = [languages]

    dataset_parts = ("XL", "TEST") if dataset_parts == "auto" else dataset_parts
    if isinstance(dataset_parts, str):
        dataset_parts = [dataset_parts]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading manifest")
    prefix = "gigaspeech"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=manifests_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    for lang in tqdm(languages, desc="Processing GigaST"):
        assert lang in GIGAST_LANGS, (lang, GIGAST_LANGS)
        logging.info(f"Loading GigaST.{lang}.json")
        gigast = GigaST(corpus_dir, lang)

        for partition, m in manifests.items():
            supervisions = []
            logging.info(f"Processing {partition}")
            if manifests_exist(
                part=partition,
                output_dir=output_dir,
                prefix="gigast-de",
                suffix="jsonl.gz",
            ):
                logging.info(
                    f"GigaST {lang} subset: {partition} already prepared - skipping."
                )
                continue

            cur_line = gigast.get_next_line()
            for sup in tqdm(
                m["supervisions"], desc="Generate the extented supervisions"
            ):
                if cur_line["sid"] == sup.id:
                    new_sup = sup
                    if partition != "TEST":
                        new_sup.custom = {
                            "text_raw": cur_line["text_raw"],
                            "extra": cur_line["extra"],
                        }
                    else:
                        new_sup.custom = {"text_raw": cur_line["text_raw"]}
                    supervisions.append(new_sup)
                    try:
                        cur_line = gigast.get_next_line()
                    except StopIteration:
                        break

            logging.info(f"Saving GigaST {lang} subset: {partition}")
            supervisionset = SupervisionSet.from_segments(supervisions)
            supervisionset.to_file(
                output_dir / f"gigast-{lang}_supervisions_{partition}.jsonl.gz"
            )
