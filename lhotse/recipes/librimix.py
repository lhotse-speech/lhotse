import zipfile
import urllib.request
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Union

import pandas as pd

from lhotse.audio import RecordingSet, Recording, AudioSource
from lhotse.supervision import SupervisionSet, SupervisionSegment
from lhotse.utils import Pathlike, Seconds


def download_and_unzip(
        target_dir: Pathlike = '.',
        force_download: Optional[bool] = False,
        url: Optional[str] = 'https://zenodo.org/record/3871592/files/MiniLibriMix.zip'
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / 'MiniLibriMix.zip'
    if force_download or not zip_path.is_file():
        urllib.request.urlretrieve(url, filename=zip_path)
    unzipped_dir = target_dir / 'MiniLibriMix'
    completed_detector = unzipped_dir / '.completed'
    if not completed_detector.is_file():
        shutil.rmtree(unzipped_dir, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(path=target_dir)
            completed_detector.touch()


def prepare_librimix(
        librimix_csv: Pathlike,
        output_dir: Pathlike,
        with_precomputed_mixtures: bool = False,
        sampling_rate: int = 16000,
        min_segment_seconds: Seconds = 3.0
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    df = pd.read_csv(librimix_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    # First, create the audio manifest that specifies the pairs of source recordings
    # to be mixed together.
    audio_sources = RecordingSet.from_recordings(
        Recording(
            id=row['mixture_ID'],
            sources=[
                AudioSource(
                    type='file',
                    channel_ids=[0],
                    source=row['source_1_path']
                ),
                AudioSource(
                    type='file',
                    channel_ids=[1],
                    source=row['source_2_path']
                )
            ],
            sampling_rate=sampling_rate,
            num_samples=int(row['length']),
            duration_seconds=row['length'] / sampling_rate
        )
        for idx, row in df.iterrows()
        if row['length'] / sampling_rate > min_segment_seconds
    )
    audio_sources.to_yaml(output_dir / 'audio_sources.yml')
    supervision_sources = make_corresponding_supervisions(audio_sources)
    supervision_sources.to_yaml(output_dir / 'supervisions_sources.yml')

    manifests['sources'] = {
        'audio': audio_sources,
        'supervisions': supervision_sources
    }

    # When requested, create an audio manifest for the pre-computed mixtures.
    # A different way of performing the mix would be using Lhotse's on-the-fly
    # overlaying of audio Cuts.
    if with_precomputed_mixtures:
        audio_mix = RecordingSet.from_recordings(
            Recording(
                id=row['mixture_ID'],
                sources=[
                    AudioSource(
                        type='file',
                        channel_ids=[0],
                        source=row['mixture_path']
                    ),
                ],
                sampling_rate=sampling_rate,
                num_samples=int(row['length']),
                duration_seconds=row['length'] / sampling_rate
            )
            for idx, row in df.iterrows()
            if row['length'] / sampling_rate > min_segment_seconds
        )
        audio_mix.to_yaml(output_dir / 'audio_mix.yml')
        supervision_mix = make_corresponding_supervisions(audio_mix)
        supervision_mix.to_yaml(output_dir / 'supervisions_mix.yml')
        manifests['premixed'] = {
            'audio': audio_mix,
            'supervisions': supervision_mix
        }

    # When the LibriMix CSV specifies noises, we create a separate RecordingSet for them,
    # so that we can extract their features and overlay them as Cuts later.
    if 'noise_path' in df:
        audio_noise = RecordingSet.from_recordings(
            Recording(
                id=row['mixture_ID'],
                sources=[
                    AudioSource(
                        type='file',
                        channel_ids=[0],
                        source=row['noise_path']
                    ),
                ],
                sampling_rate=sampling_rate,
                num_samples=int(row['length']),
                duration_seconds=row['length'] / sampling_rate
            )
            for idx, row in df.iterrows()
            if row['length'] / sampling_rate > min_segment_seconds
        )
        audio_noise.to_yaml(output_dir / 'audio_noise.yml')
        supervision_noise = make_corresponding_supervisions(audio_noise)
        supervision_noise.to_yaml(output_dir / 'supervisions_noise.yml')
        manifests['noise'] = {
            'audio': audio_noise,
            'supervisions': supervision_noise
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
            id=f'{recording.id}-c{source.channel_ids[0]}',
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration_seconds,
            channel_id=source.channel_ids[0],
        )
        for recording in audio
        for source in recording.sources
    )
