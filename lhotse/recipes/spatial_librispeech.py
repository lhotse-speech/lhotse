import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download

SPATIAL_LIBRISPEECH = ("train", "test")
BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/spatial-librispeech/v1"
META_DATA_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/spatial-librispeech/v1/metadata.parquet"


def _download_and_save_audio(target_file: Pathlike, url: str):
    # Implementation from https://github.com/apple/ml-spatial-librispeech/pull/1/
    # Use the requests module to avoid the 403 forbidden error
    def _download_file(url: str) -> bytes:
        """This function downloads and returns the content of the given url
        Args:
            url (str): the url of the file to be downloaded
        Raises:
            e: The exception that is raised by the request module
        Returns:
            file_content (bytes): The file content downloaded from the url
        """

        try:
            import requests
        except ImportError:
            raise ImportError(
                "The Spatial LibriSpeech recipe requires requests dependency to download the dataset. You can install the dependency using: pip install requests"
            )

        try:
            file_content = requests.get(url, allow_redirects=True).content
            return file_content
        except requests.exceptions.RequestException as e:
            raise e

    # Implementation from https://github.com/apple/ml-spatial-librispeech/pull/1/
    def _save_audio_content(target_file: str, file_content: bytes):
        """This function saves the downloaded content passed via `file_content' in the `target_file'
        Args:
            target_file (str): the target path for the file content to be saved to
            file_content (bytes): the content to be saved

        Raises:
            e: the IOError raised by the writing operation
        """
        try:
            with open(target_file, "wb") as file:
                file.write(file_content)
        except IOError as e:
            raise e

    file_content = _download_file(url)
    _save_audio_content(target_file, file_content)


def download_spatial_librispeech(
    target_dir: Pathlike = ".",
    dataset_parts: Union[str, Sequence[str]] = SPATIAL_LIBRISPEECH,
    force_download: bool = False,
    base_url: str = BASE_URL,
    num_jobs: int = 1,
) -> Path:
    """
    Download the Spatial-LibriSpeech dataset.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param dataset_parts: "all" or a list of splits (e.g. ["train", "test"]) to download.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the resource.
    :return: the path to downloaded and extracted directory with data.
    """

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "The Spatial LibriSpeech recipe requires pandas, pyarrow and fastparquet dependency to parse parquet formatted metadata. You can install the dependencies using: pip install pandas pyarrow fastparquet"
        )

    def _download_spatial_librispeech_audio_files(
        target_dir: Pathlike,
        dataset_parts: Sequence[str],
        metadata: pd.DataFrame,
        base_url: str,
        force_download: bool = False,
        num_jobs: int = 1,
    ):
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        audio_url = f"{base_url}/ambisonics"
        from concurrent.futures.thread import ThreadPoolExecutor

        for part in dataset_parts:
            part_dir = target_dir / part
            part_dir.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(num_jobs) as ex:
            for sample_id, split in tqdm(
                zip(metadata["sample_id"], metadata["split"]),
                total=len(metadata["sample_id"]),
            ):
                if split not in dataset_parts:
                    continue
                recording_path = target_dir / split / f"{sample_id:06}.flac"
                recording_url = f"{audio_url}/{sample_id:06}.flac"
                if not recording_path.exists() or force_download:
                    ex.submit(_download_and_save_audio, recording_path, recording_url)

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if dataset_parts == "all":
        dataset_parts = SPATIAL_LIBRISPEECH
    else:
        dataset_parts = (
            [dataset_parts] if isinstance(dataset_parts, str) else dataset_parts
        )
    for part in dataset_parts:
        assert part in SPATIAL_LIBRISPEECH, f"Unknown dataset part: {part}"

    corpus_dir = target_dir / "Spatial-LibriSpeech"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping download, found {completed_detector}.")
        return corpus_dir

    metadata_path = corpus_dir / "metadata.parquet"
    if not metadata_path.is_file() or force_download:
        resumable_download(META_DATA_URL, metadata_path, force_download=force_download)
    elif metadata_path.is_file():
        logging.info(f"Skipping download, found {metadata_path}.")

    metadata = pd.read_parquet(metadata_path)
    try:
        _download_spatial_librispeech_audio_files(
            target_dir=corpus_dir / "audio_files",
            dataset_parts=dataset_parts,
            metadata=metadata,
            base_url=base_url,
            force_download=force_download,
            num_jobs=num_jobs,
        )
    except Exception as e:
        logging.error(f"Failed to download audio files: {e}")
        raise e

    completed_detector.touch()
    return corpus_dir


def prepare_spatial_librispeech(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    dataset_parts: Union[str, Sequence[str]] = SPATIAL_LIBRISPEECH,
    normalize_text: str = "none",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train', 'test'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param normalize_text: str, "none" or "lower",
        for "lower" the transcripts are converted to lower-case.
    :param num_jobs: int, number of parallel threads used for 'parse_utterance' calls.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "The Spatial LibriSpeech recipe requires pandas, pyarrow and fastparquet dependency to parse parquet formatted metadata. You can install the dependencies using: pip install pandas pyarrow fastparquet"
        )

    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir) if output_dir is not None else corpus_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if dataset_parts == "all":
        dataset_parts = SPATIAL_LIBRISPEECH
    else:
        dataset_parts = (
            [dataset_parts] if isinstance(dataset_parts, str) else dataset_parts
        )
    for part in dataset_parts:
        assert part in SPATIAL_LIBRISPEECH, f"Unknown dataset part: {part}"

    metadata_path = corpus_dir / "metadata.parquet"
    assert metadata_path.is_file(), f"{metadata_path} not found"
    metadata = pd.read_parquet(metadata_path)

    manifests = {}

    for part in dataset_parts:
        assert part in SPATIAL_LIBRISPEECH, f"Unknown dataset part: {part}"
        logging.info(f"Processing {part} split...")
        part_dir = corpus_dir / "audio_files" / part
        recording_set = RecordingSet.from_dir(
            part_dir,
            pattern="*.flac",
            num_jobs=num_jobs,
            recording_id=lambda x: x.stem,
        )

        supervision_segments = []
        part_metadata = metadata[metadata["split"] == part]
        for _, row in tqdm(
            part_metadata.iterrows(),
            total=len(part_metadata["sample_id"]),
            desc=f"Processing supervision segments for split: {part}",
        ):
            recording_id = f"{row['sample_id']:06}"
            start = 0
            duration = recording_set[recording_id].duration
            channel = recording_set[recording_id].channel_ids
            text = row["speech/librispeech_metadata/transcription"]
            speaker = row["speech/librispeech_metadata/reader_id"]
            gender = row["speech/librispeech_metadata/reader_sex"]
            segment = SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=start,
                duration=duration,
                channel=channel,
                text=text,
                gender=gender,
                speaker=speaker,
            )
            supervision_segments.append(segment)
        supervision_set = SupervisionSet.from_segments(supervision_segments)

        # Normalize text to lowercase
        if normalize_text == "lower":
            to_lower = lambda text: text.lower()
            supervision_set = SupervisionSet.from_segments(
                [s.transform_text(to_lower) for s in supervision_set]
            )

        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            recording_set.to_file(
                output_dir / f"spatial-librispeech_recordings_{part}.jsonl.gz"
            )
            supervision_set.to_file(
                output_dir / f"spatial-librispeech_supervisions_{part}.jsonl.gz"
            )

        manifests[part] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }

    return manifests
