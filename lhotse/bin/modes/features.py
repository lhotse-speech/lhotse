from pathlib import Path
from typing import Optional

import click

from lhotse.audio import RecordingSet
from lhotse.augmentation import available_wav_augmentations, WavAugmenter
from lhotse.bin.modes.cli_base import cli
from lhotse.features import FeatureExtractor, FeatureSetBuilder, create_default_feature_extractor, Fbank
from lhotse.supervision import SupervisionSet
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
@click.option('-s', '--segmentation-manifest', type=click.Path(exists=True, dir_okay=False),
              help='Optional manifest specifying the regions, for which features are to be extracted. '
                   'When not specified, features will be extracted for the entire recording. '
                   'Supervision manifest can be used here.')
@click.option('-a', '--augmentation', type=click.Choice(available_wav_augmentations()),
              default=None, help='Optional time-domain data augmentation effect chain to apply.')
@click.option('-f', '--feature-manifest', type=click.Path(exists=True, dir_okay=False),
              help='Optional manifest specifying feature extractor configuration.')
@click.option('--compressed/--not-compressed', default=True, help='Enable/disable lilcom for features compression.')
@click.option('-t', '--lilcom-tick-power', type=int, default=-5,
              help='Determines the compression accuracy; '
                   'the input will be compressed to integer multiples of 2^tick_power')
@click.option('-r', '--root-dir', type=click.Path(exists=True, file_okay=False), default=None,
              help='Root directory - all paths in the manifest will use this as prefix.')
@click.option('-j', '--num-jobs', type=int, default=1, help='Number of parallel processes.')
def extract(
        recording_manifest: Pathlike,
        output_dir: Pathlike,
        segmentation_manifest: Optional[Pathlike],
        augmentation: str,
        feature_manifest: Optional[Pathlike],
        compressed: bool,
        lilcom_tick_power: int,
        root_dir: Optional[Pathlike],
        num_jobs: int
):
    """
    Extract features for recordings in a given AUDIO_MANIFEST. The features are stored in OUTPUT_DIR,
    with one file per recording (or segment).
    """
    recordings = RecordingSet.from_json(recording_manifest)

    feature_extractor = (FeatureExtractor.from_yaml(feature_manifest)
                         if feature_manifest is not None else Fbank())

    # TODO: to be used (actually, only the segmentation info will be used, and all supervision info will be ignored)
    supervision_set = (SupervisionSet.from_json(segmentation_manifest)
                       if segmentation_manifest is not None else None)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    augmenter = None
    if augmentation is not None:
        sampling_rate = next(iter(recordings)).sampling_rate
        assert all(rec.sampling_rate == sampling_rate for rec in recordings), \
            "Wav augmentation effect chains expect all the recordings to have the same sampling rate at this time."
        augmenter = WavAugmenter.create_predefined(name=augmentation, sampling_rate=sampling_rate)

    feature_set_builder = FeatureSetBuilder(
        feature_extractor=feature_extractor,
        output_dir=output_dir,
        augmenter=augmenter
    )
    feature_set_builder.process_and_store_recordings(
        recordings=recordings,
        segmentation=None,  # TODO: implement and use
        compressed=compressed,
        lilcom_tick_power=lilcom_tick_power,
        num_jobs=num_jobs
    )
