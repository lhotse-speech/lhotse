from pathlib import Path
from typing import Optional

import click

from lhotse.audio import RecordingSet
from lhotse.bin.modes.cli_base import cli
from lhotse.features import FeatureExtractor, FeatureSetBuilder
from lhotse.supervision import SupervisionSet
from lhotse.utils import Pathlike

__all__ = ['write_default_feature_config', 'make_feats']


@cli.command()
@click.argument('output_config', type=click.Path())
def write_default_feature_config(output_config):
    """Save a default feature extraction config to OUTPUT_CONFIG."""
    FeatureExtractor().to_yaml(output_config, include_defaults=True)


@cli.command()
@click.argument('audio_manifest', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path())
@click.option('-s', '--segmentation-manifest', type=click.Path(exists=True, dir_okay=False),
              help='Optional manifest specifying the regions, for which features are to be extracted. '
                   'When not specified, features will be extracted for the entire recording. '
                   'Supervision manifest can be used here.')
@click.option('-a', '--augmentation-manifest', type=click.Path(exists=True, dir_okay=False),
              help='Optional manifest specifying augmentation transforms that can be applied to recordings.')
@click.option('-f', '--feature-manifest', type=click.Path(exists=True, dir_okay=False),
              help='Optional manifest specifying feature extractor configuration.')
@click.option('--compressed/--not-compressed', default=True, help='Enable/disable lilcom for features compression.')
@click.option('-t', '--lilcom-tick-power', type=int, default=-8,
              help='Determines the compression accuracy; '
                   'the input will be compressed to integer multiples of 2^tick_power')
@click.option('-r', '--root-dir', type=click.Path(exists=True, file_okay=False), default=None,
              help='Root directory - all paths in the manifest will use this as prefix.')
@click.option('-j', '--num-jobs', type=int, default=1, help='Number of parallel processes.')
def make_feats(
        audio_manifest: Pathlike,
        output_dir: Pathlike,
        segmentation_manifest: Optional[Pathlike],
        # TODO: augmentation manifest should specify a number of transforms and probability of their application
        # e.g.:
        # "add_noise", "prob": 0.5, "noise_recordings": ["path1.wav", "path2.wav"]
        # "reverberate", "prob": 0.2, "rirs": ["rir1.wav", "rir2.wav"] (or however the RIRs are stored like... can be params for simulation)
        augmentation_manifest: Optional[Pathlike],
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
    audio_set = RecordingSet.from_yaml(audio_manifest)

    feature_extractor = (FeatureExtractor.from_yaml(feature_manifest)
                         if feature_manifest is not None else FeatureExtractor())

    # TODO: to be used (actually, only the segmentation info will be used, and all supervision info will be ignored)
    supervision_set = (SupervisionSet.from_yaml(segmentation_manifest)
                       if segmentation_manifest is not None else None)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    feature_set_builder = FeatureSetBuilder(
        feature_extractor=feature_extractor,
        output_dir=output_dir,
        root_dir=root_dir,
        augmentation_manifest=augmentation_manifest
    )
    feature_set_builder.process_and_store_recordings(
        recordings=audio_set,
        segmentation=None,  # TODO: implement and use
        compressed=compressed,
        lilcom_tick_power=lilcom_tick_power,
        num_jobs=num_jobs
    )
