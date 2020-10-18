from pathlib import Path
from typing import Optional

import click

from lhotse.audio import RecordingSet
from lhotse.augmentation import available_wav_augmentations, WavAugmenter
from lhotse.bin.modes.cli_base import cli
from lhotse.features import FeatureExtractor, FeatureSetBuilder, create_default_feature_extractor, Fbank
from lhotse.features.io import available_storage_backends, get_writer
from lhotse.utils import Pathlike


@cli.group()
def feat():
    """Feature extraction related commands."""
    pass


@feat.command(context_settings=dict(show_default=True))
@click.argument('output_config', type=click.Path())
@click.option('-f', '--feature-type', type=click.Choice(['fbank', 'mfcc', 'spectrogram']), default='fbank',
              help='Which feature extractor type to use.')
def write_default_config(output_config: Pathlike, feature_type: str):
    """Save a default feature extraction config to OUTPUT_CONFIG."""
    create_default_feature_extractor(feature_type).to_yaml(output_config)


@feat.command(context_settings=dict(show_default=True))
@click.argument('recording_manifest', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option('-a', '--augmentation', type=click.Choice(available_wav_augmentations()),
              default=None, help='Optional time-domain data augmentation effect chain to apply.')
@click.option('-f', '--feature-manifest', type=click.Path(exists=True, dir_okay=False),
              help='Optional manifest specifying feature extractor configuration.')
@click.option('--storage-type', type=click.Choice(available_storage_backends()),
              default='lilcom_files',
              help='Select a storage backend for the feature matrices.')
@click.option('-t', '--lilcom-tick-power', type=int, default=-5,
              help='Determines the compression accuracy; '
                   'the input will be compressed to integer multiples of 2^tick_power')
@click.option('-r', '--root-dir', type=click.Path(exists=True, file_okay=False), default=None,
              help='Root directory - all paths in the manifest will use this as prefix.')
@click.option('-j', '--num-jobs', type=int, default=1, help='Number of parallel processes.')
def extract(
        recording_manifest: Pathlike,
        output_dir: Pathlike,
        augmentation: str,
        feature_manifest: Optional[Pathlike],
        storage_type: str,
        lilcom_tick_power: int,
        root_dir: Optional[Pathlike],
        num_jobs: int
):
    """
    Extract features for recordings in a given AUDIO_MANIFEST. The features are stored in OUTPUT_DIR,
    with one file per recording (or segment).
    """
    recordings: RecordingSet = RecordingSet.from_json(recording_manifest)
    if root_dir is not None:
        recordings = recordings.with_path_prefix(root_dir)

    feature_extractor = (FeatureExtractor.from_yaml(feature_manifest)
                         if feature_manifest is not None else Fbank())

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    storage_path = output_dir / 'feats.h5' if 'hdf5' in storage_type else output_dir / 'storage'

    augmenter = None
    if augmentation is not None:
        sampling_rate = next(iter(recordings)).sampling_rate
        assert all(rec.sampling_rate == sampling_rate for rec in recordings), \
            "Wav augmentation effect chains expect all the recordings to have the same sampling rate at this time."
        augmenter = WavAugmenter.create_predefined(name=augmentation, sampling_rate=sampling_rate)

    with get_writer(storage_type)(storage_path, tick_power=lilcom_tick_power) as storage:
        feature_set_builder = FeatureSetBuilder(
            feature_extractor=feature_extractor,
            storage=storage,
            augmenter=augmenter
        )
        feature_set_builder.process_and_store_recordings(
            recordings=recordings,
            output_manifest=output_dir / 'feature_manifest.json.gz',
            num_jobs=num_jobs
        )
