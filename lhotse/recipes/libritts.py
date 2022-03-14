"""
LibriTTS is a multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz sampling rate, prepared by Heiga Zen with the assistance of Google Speech and Google Brain team members. The LibriTTS corpus is designed for TTS research. It is derived from the original materials (mp3 audio files from LibriVox and text files from Project Gutenberg) of the LibriSpeech corpus. The main differences from the LibriSpeech corpus are listed below:
The audio files are at 24kHz sampling rate.
The speech is split at sentence breaks.
Both original and normalized texts are included.
Contextual information (e.g., neighbouring sentences) can be extracted.
Utterances with significant background noise are excluded.
For more information, refer to the paper "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech", Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu, arXiv, 2019. If you use the LibriTTS corpus in your work, please cite this paper where it was introduced.
"""
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm import tqdm

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike, urlretrieve_progress

LIBRITTS = (
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)


def download_libritts(
    target_dir: Pathlike = ".",
    dataset_parts: Optional[Union[str, Sequence[str]]] = "all",
    force_download: Optional[bool] = False,
    base_url: Optional[str] = "http://www.openslr.org/resources",
) -> Path:
    """
    Download and untar the dataset, supporting both LibriSpeech and MiniLibrispeech

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param dataset_parts: "librispeech", "mini_librispeech",
        or a list of splits (e.g. "dev-clean") to download.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to downloaded and extracted directory with data.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if dataset_parts == "all":
        dataset_parts = LIBRITTS

    for part in tqdm(dataset_parts, desc="Downloading LibriSpeech parts"):
        if part not in LIBRITTS:
            logging.warning(f"Skipping invalid dataset part name: {part}")
        url = f"{base_url}/60"
        tar_name = f"{part}.tar.gz"
        tar_path = target_dir / tar_name
        part_dir = target_dir / f"LibriTTS/{part}"
        completed_detector = part_dir / ".completed"
        if completed_detector.is_file():
            logging.info(f"Skipping {part} because {completed_detector} exists.")
            continue
        if force_download or not tar_path.is_file():
            urlretrieve_progress(
                f"{url}/{tar_name}", filename=tar_path, desc=f"Downloading {tar_name}"
            )
        shutil.rmtree(part_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=target_dir)
        completed_detector.touch()

    return target_dir


def prepare_libritts(
    corpus_dir: Pathlike,
    dataset_parts: Union[str, Sequence[str]] = "auto",
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
    link_previous_utt: bool = False,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: the number of parallel workers parsing the data.
    :param link_previous_utt: If true adds previous utterance id to supervisions.
        Useful for reconstructing chains of utterances as they were read.
        If previous utterance was skipped from LibriTTS datasets previous_utt label is None.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if dataset_parts == "auto":
        dataset_parts = LIBRITTS
    elif isinstance(dataset_parts, str):
        assert dataset_parts in LIBRITTS
        dataset_parts = [dataset_parts]

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts, output_dir=output_dir, prefix="libritts"
        )

    # Contents of the file
    #   ;ID  |SEX| SUBSET           |MINUTES| NAME
    #   14   | F | train-clean-360  | 25.03 | ...
    #   16   | F | train-clean-360  | 25.11 | ...
    #   17   | M | train-clean-360  | 25.04 | ...
    spk2gender = {
        spk_id.strip(): gender.strip()
        for spk_id, gender, *_ in (
            line.split("|")
            for line in (corpus_dir / "SPEAKERS.txt").read_text().splitlines()
            if not line.startswith(";")
        )
    }

    for part in tqdm(dataset_parts, desc="Preparing LibriTTS parts"):
        if manifests_exist(part=part, output_dir=output_dir, prefix="libritts"):
            logging.info(f"LibriTTS subset: {part} already prepared - skipping.")
            continue
        part_path = corpus_dir / part
        recordings = RecordingSet.from_dir(part_path, "*.wav", num_jobs=num_jobs)
        supervisions = []
        for trans_path in tqdm(
            part_path.rglob("*.trans.tsv"),
            desc="Scanning transcript files (progbar per speaker)",
            leave=False,
        ):
            # The trans.tsv files contain only the recordings that were kept for LibriTTS.
            # Example path to a file:
            #   /export/corpora5/LibriTTS/dev-clean/84/121123/84_121123.trans.tsv
            #
            # Example content:
            #   84_121123_000007_000001 Maximilian.     Maximilian.
            #   84_121123_000008_000000 Villefort rose, half ashamed of being surprised in such a paroxysm of grief.    Villefort rose, half ashamed of being surprised in such a paroxysm of grief.

            # book.tsv contains additional metadata
            utt2snr = [
                (rec_id, float(snr))
                for rec_id, *_, snr in map(
                    str.split,
                    (
                        trans_path.parent
                        / trans_path.name.replace(".trans.tsv", ".book.tsv")
                    )
                    .read_text()
                    .splitlines(),
                )
            ]
            # keeps the order of uttids as they appear in book.tsv
            uttids = [r for r, _ in utt2snr]
            utt2snr = dict(utt2snr)

            if link_previous_utt:
                # Using the property of sorted keys to find previous utterance
                # The keys has structure speaker_book_x_y e.g. 1089_134691_000004_000001
                utt2prevutt = dict(zip(uttids + [None], [None] + uttids))

            prev_rec_id = None
            for line in trans_path.read_text().splitlines():
                rec_id, orig_text, norm_text = line.split("\t")
                spk_id = rec_id.split("_")[0]
                customd = {"orig_text": orig_text, "snr": utt2snr[rec_id]}
                if link_previous_utt:
                    # all recordings ids should be in the book.csv
                    # but they are some missing e.g. 446_123502_000030_000003
                    prev_utt = utt2prevutt.get(rec_id, None)
                    # previous utterance has to be present in trans.csv - otherwise it was skipped
                    prev_utt = prev_utt if prev_utt == prev_rec_id else None
                    customd["prev_utt"] = prev_utt
                    prev_rec_id = rec_id
                supervisions.append(
                    SupervisionSegment(
                        id=rec_id,
                        recording_id=rec_id,
                        start=0.0,
                        duration=recordings[rec_id].duration,
                        channel=0,
                        text=norm_text,
                        language="English",
                        speaker=spk_id,
                        gender=spk2gender[spk_id],
                        custom=customd,
                    )
                )

        supervisions = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        if output_dir is not None:
            supervisions.to_json(output_dir / f"libritts_supervisions_{part}.json")
            recordings.to_json(output_dir / f"libritts_recordings_{part}.json")

        manifests[part] = {"recordings": recordings, "supervisions": supervisions}

    return manifests
