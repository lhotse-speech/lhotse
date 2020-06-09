from pathlib import Path

import click
import pandas as pd

from lhotse.audio import AudioSet, Recording, AudioSource
from lhotse.bin.modes import recipe
from lhotse.supervision import SupervisionSet, SupervisionSegment
from lhotse.utils import Pathlike

__all__ = ['librimix']


@recipe.command()
@click.argument('librimix-csv', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option('--sampling-rate', type=int, default=16000)
@click.option('--min-segment-seconds', type=float, default=3.0)
def librimix(librimix_csv: Pathlike, output_dir: Pathlike, sampling_rate: int, min_segment_seconds: float):
    """Recipe to prepare the manifests for LibrMix source separation task."""
    df = pd.read_csv(librimix_csv)

    audio = AudioSet(recordings={
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

    supervisions = SupervisionSet(segments={
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio.to_yaml(output_dir / 'audio.yml')
    supervisions.to_yaml(output_dir / 'supervisions.yml')
