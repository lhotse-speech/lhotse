from pathlib import Path

import click
import pandas as pd

from lhotse.audio import AudioSet, Recording, AudioSource
from lhotse.bin.modes import recipe
from lhotse.supervision import SupervisionSet, SupervisionSegment
from lhotse.utils import Pathlike

__all__ = ['librimix']


@recipe.command(context_settings=dict(show_default=True))
@click.argument('librimix-csv', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option('--sampling-rate', type=int, default=16000, help='Sampling rate to set in the AudioSet manifest.')
@click.option(
    '--min-segment-seconds', type=float, default=3.0,
    help='Remove segments shorter than MIN_SEGMENT_SECONDS.'
)
@click.option(
    '--with-precomputed-mixtures/--no-precomputed-mixtures', type=bool, default=False,
    help='Optionally create an AudioSet manifest including the precomputed LibriMix mixtures.'
)
def librimix(
        librimix_csv: Pathlike,
        output_dir: Pathlike,
        sampling_rate: int,
        min_segment_seconds: float,
        with_precomputed_mixtures: bool
):
    """Recipe to prepare the manifests for LibrMix source separation task."""
    df = pd.read_csv(librimix_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, create the audio manifest that specifies the pairs of source recordings
    # to be mixed together.
    audio_sources = AudioSet(recordings={
        row['mixture_ID']: Recording(
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
    })
    audio_sources.to_yaml(output_dir / 'audio_sources.yml')
    make_corresponding_supervisions(audio_sources).to_yaml(output_dir / 'supervisions_sources.yml')

    # When requested, create an audio manifest for the pre-computed mixtures.
    # A different way of performing the mix would be using Lhotse's on-the-fly
    # overlaying of audio Cuts.
    if with_precomputed_mixtures:
        audio_mix = AudioSet(recordings={
            row['mixture_ID']: Recording(
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
        })
        audio_mix.to_yaml(output_dir / 'audio_mix.yml')
        make_corresponding_supervisions(audio_mix).to_yaml(output_dir / 'supervisions_mix.yml')

    # When the LibriMix CSV specifies noises, we create a separate AudioSet for them,
    # so that we can extract their features and overlay them as Cuts later.
    if 'noise_path' in df:
        audio_noise = AudioSet(recordings={
            row['mixture_ID']: Recording(
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
        })
        audio_noise.to_yaml(output_dir / 'audio_noise.yml')
        make_corresponding_supervisions(audio_noise).to_yaml(output_dir / 'supervisions_noise.yml')


def make_corresponding_supervisions(audio: AudioSet) -> SupervisionSet:
    """
    Prepare a supervision set - in this case it just describes
    which segments are available in the corpus, as the actual supervisions for
    speech separation come from the source recordings.
    """
    return SupervisionSet(segments={
        f'{recording.id}-c{source.channel_ids[0]}': SupervisionSegment(
            id=f'{recording.id}-c{source.channel_ids[0]}',
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration_seconds,
            channel_id=source.channel_ids[0],
        )
        for recording in audio
        for source in recording.sources
    })
