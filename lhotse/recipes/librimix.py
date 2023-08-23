import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union
from zipfile import ZipFile

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, resumable_download


def download_librimix(
    target_dir: Pathlike = ".",
    force_download: Optional[bool] = False,
    url: Optional[str] = "https://zenodo.org/record/3871592/files/MiniLibriMix.zip",
) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "MiniLibriMix.zip"
    unzipped_dir = target_dir / "MiniLibriMix"
    completed_detector = unzipped_dir / ".completed"
    if completed_detector.is_file():
        logging.info(f"Skipping {zip_path} because {completed_detector} exists.")
        return unzipped_dir
    resumable_download(
        url,
        filename=zip_path,
        force_download=force_download,
    )
    shutil.rmtree(unzipped_dir, ignore_errors=True)
    with ZipFile(zip_path) as zf:
        zf.extractall(path=target_dir)
    completed_detector.touch()
    return unzipped_dir


def prepare_librimix(
    librimix_csv: Pathlike,
    output_dir: Optional[Pathlike] = None,
    with_precomputed_mixtures: bool = False,
    sampling_rate: int = 16000,
    min_segment_seconds: Seconds = 3.0,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    import pandas as pd

    assert Path(librimix_csv).is_file(), f"No such file: {librimix_csv}"
    df = pd.read_csv(librimix_csv)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    # First, create the audio manifest that specifies the pairs of source recordings
    # to be mixed together.
    audio_sources = RecordingSet.from_recordings(
        Recording(
            id=row["mixture_ID"],
            sources=[
                AudioSource(type="file", channels=[0], source=row["source_1_path"]),
                AudioSource(type="file", channels=[1], source=row["source_2_path"]),
            ],
            sampling_rate=sampling_rate,
            num_samples=int(row["length"]),
            duration=row["length"] / sampling_rate,
        )
        for idx, row in df.iterrows()
        if row["length"] / sampling_rate > min_segment_seconds
    )
    supervision_sources = make_corresponding_supervisions(audio_sources)

    # Fix manifests and validate them
    audio_sources, supervision_sources = fix_manifests(
        audio_sources, supervision_sources
    )
    validate_recordings_and_supervisions(audio_sources, supervision_sources)
    if output_dir is not None:
        audio_sources.to_file(output_dir / "librimix_recordings_sources.jsonl.gz")
        supervision_sources.to_file(
            output_dir / "librimix_supervisions_sources.jsonl.gz"
        )
    manifests["sources"] = {
        "recordings": audio_sources,
        "supervisions": supervision_sources,
    }

    # When requested, create an audio manifest for the pre-computed mixtures.
    # A different way of performing the mix would be using Lhotse's on-the-fly
    # overlaying of audio Cuts.
    if with_precomputed_mixtures:
        audio_mix = RecordingSet.from_recordings(
            Recording(
                id=row["mixture_ID"],
                sources=[
                    AudioSource(type="file", channels=[0], source=row["mixture_path"]),
                ],
                sampling_rate=sampling_rate,
                num_samples=int(row["length"]),
                duration=row["length"] / sampling_rate,
            )
            for idx, row in df.iterrows()
            if row["length"] / sampling_rate > min_segment_seconds
        )
        supervision_mix = make_corresponding_supervisions(audio_mix)
        audio_mix, supervision_mix = fix_manifests(audio_mix, supervision_mix)
        validate_recordings_and_supervisions(audio_mix, supervision_mix)
        if output_dir is not None:
            audio_mix.to_file(output_dir / "librimix_recordings_mix.jsonl.gz")
            supervision_mix.to_file(output_dir / "librimix_supervisions_mix.jsonl.gz")
        manifests["premixed"] = {
            "recordings": audio_mix,
            "supervisions": supervision_mix,
        }

    # When the LibriMix CSV specifies noises, we create a separate RecordingSet for them,
    # so that we can extract their features and overlay them as Cuts later.
    if "noise_path" in df:
        audio_noise = RecordingSet.from_recordings(
            Recording(
                id=row["mixture_ID"],
                sources=[
                    AudioSource(type="file", channels=[0], source=row["noise_path"]),
                ],
                sampling_rate=sampling_rate,
                num_samples=int(row["length"]),
                duration=row["length"] / sampling_rate,
            )
            for idx, row in df.iterrows()
            if row["length"] / sampling_rate > min_segment_seconds
        )
        supervision_noise = make_corresponding_supervisions(audio_noise)
        audio_noise, supervision_noise = fix_manifests(audio_noise, supervision_noise)
        validate_recordings_and_supervisions(audio_noise, supervision_noise)
        if output_dir is not None:
            audio_noise.to_file(output_dir / "librimix_recordings_noise.jsonl.gz")
            supervision_noise.to_file(
                output_dir / "libirmix_supervisions_noise.jsonl.gz"
            )
        manifests["noise"] = {
            "recordings": audio_noise,
            "supervisions": supervision_noise,
        }

    return manifests


def make_corresponding_supervisions(audio: RecordingSet) -> SupervisionSet:
    """
    Prepare a supervision set - in this case it just describes
    which segments are available in the corpus, as the actual supervisions for
    speech separation come from the source recordings.
    """
    return SupervisionSet.from_segments(
        SupervisionSegment(
            id=f"{recording.id}-c{source.channels[0]}",
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            channel=source.channels[0],
        )
        for recording in audio
        for source in recording.sources
    )
