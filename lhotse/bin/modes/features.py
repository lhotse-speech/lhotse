from pathlib import Path
from typing import Optional

import click

from lhotse import FeatureSet, Features, LilcomURLWriter, Seconds
from lhotse.audio import RecordingSet
from lhotse.bin.modes.cli_base import cli
from lhotse.features import (
    Fbank,
    FeatureExtractor,
    FeatureSetBuilder,
    create_default_feature_extractor,
)
from lhotse.features.base import FEATURE_EXTRACTORS
from lhotse.features.io import available_storage_backends, get_writer
from lhotse.utils import Pathlike, fastcopy


@cli.group()
def feat():
    """Feature extraction related commands."""
    pass


@feat.command(context_settings=dict(show_default=True))
@click.argument("output_config", type=click.Path())
@click.option(
    "-f",
    "--feature-type",
    type=click.Choice(list(FEATURE_EXTRACTORS)),
    default="fbank",
    help="Which feature extractor type to use.",
)
def write_default_config(output_config: Pathlike, feature_type: str):
    """Save a default feature extraction config to OUTPUT_CONFIG."""
    create_default_feature_extractor(feature_type).to_yaml(output_config)


@feat.command(context_settings=dict(show_default=True))
@click.argument("recording_manifest", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-f",
    "--feature-manifest",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional manifest specifying feature extractor configuration.",
)
@click.option(
    "--storage-type",
    type=click.Choice(available_storage_backends()),
    default="lilcom_files",
    help="Select a storage backend for the feature matrices.",
)
@click.option(
    "-t",
    "--lilcom-tick-power",
    type=int,
    default=-5,
    help="Determines the compression accuracy; "
    "the input will be compressed to integer multiples of 2^tick_power",
)
@click.option(
    "-r",
    "--root-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Root directory - all paths in the manifest will use this as prefix.",
)
@click.option(
    "-j", "--num-jobs", type=int, default=1, help="Number of parallel processes."
)
def extract(
    recording_manifest: Pathlike,
    output_dir: Pathlike,
    feature_manifest: Optional[Pathlike],
    storage_type: str,
    lilcom_tick_power: int,
    root_dir: Optional[Pathlike],
    num_jobs: int,
):
    """
    Extract features for recordings in a given AUDIO_MANIFEST. The features are stored in OUTPUT_DIR,
    with one file per recording (or segment).
    """
    recordings: RecordingSet = RecordingSet.from_json(recording_manifest)
    if root_dir is not None:
        recordings = recordings.with_path_prefix(root_dir)

    feature_extractor = (
        FeatureExtractor.from_yaml(feature_manifest)
        if feature_manifest is not None
        else Fbank()
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    storage_path = (
        output_dir / "feats.h5" if "hdf5" in storage_type else output_dir / "storage"
    )

    with get_writer(storage_type)(
        storage_path, tick_power=lilcom_tick_power
    ) as storage:
        feature_set_builder = FeatureSetBuilder(
            feature_extractor=feature_extractor,
            storage=storage,
        )
        feature_set_builder.process_and_store_recordings(
            recordings=recordings,
            output_manifest=output_dir / "feature_manifest.json.gz",
            num_jobs=num_jobs,
        )


@feat.command(context_settings=dict(show_default=True))
@click.argument("cutset", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_cutset", type=click.Path())
@click.argument("storage_path", type=click.Path())
@click.option(
    "-f",
    "--feature-manifest",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional manifest specifying feature extractor configuration.",
)
@click.option(
    "--storage-type",
    type=click.Choice(available_storage_backends()),
    default="lilcom_hdf5",
    help="Select a storage backend for the feature matrices.",
)
@click.option(
    "-j", "--num-jobs", type=int, default=1, help="Number of parallel processes."
)
def extract_cuts(
    cutset: Pathlike,
    output_cutset: Pathlike,
    storage_path: Pathlike,
    feature_manifest: Optional[Pathlike],
    storage_type: str,
    num_jobs: int,
):
    """
    Extract features for cuts in a given CUTSET manifest.
    The features are stored in STORAGE_PATH, and the output manifest
    with features is stored in OUTPUT_CUTSET.
    """
    from lhotse import CutSet

    cuts: CutSet = CutSet.from_file(cutset)
    feature_extractor = (
        FeatureExtractor.from_yaml(feature_manifest)
        if feature_manifest is not None
        else Fbank()
    )
    cuts = cuts.compute_and_store_features(
        extractor=feature_extractor,
        storage_path=storage_path,
        num_jobs=num_jobs,
        storage_type=get_writer(storage_type),
    )
    Path(output_cutset).parent.mkdir(parents=True, exist_ok=True)
    cuts.to_file(output_cutset)


@feat.command(context_settings=dict(show_default=True))
@click.argument("cutset", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_cutset", type=click.Path())
@click.argument("storage_path", type=click.Path())
@click.option(
    "-f",
    "--feature-manifest",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional manifest specifying feature extractor configuration. "
    "If you want to use CUDA, you should specify the device in this "
    "config.",
)
@click.option(
    "--storage-type",
    type=click.Choice(available_storage_backends()),
    default="lilcom_hdf5",
    help="Select a storage backend for the feature matrices.",
)
@click.option(
    "-j", "--num-jobs", type=int, default=4, help="Number of dataloader workers."
)
@click.option(
    "-b",
    "--batch-duration",
    type=float,
    default=600.0,
    help="At most this many seconds of audio will be processed in each batch.",
)
def extract_cuts_batch(
    cutset: Pathlike,
    output_cutset: Pathlike,
    storage_path: Pathlike,
    feature_manifest: Optional[Pathlike],
    storage_type: str,
    num_jobs: int,
    batch_duration: Seconds,
):
    """
    Extract features for cuts in a given CUTSET manifest.
    The features are stored in STORAGE_PATH, and the output manifest
    with features is stored in OUTPUT_CUTSET.

    This version enables CUDA acceleration for feature extractors
    that support it (e.g., kaldifeat extractors).

    \b
    Example usage of kaldifeat fbank with CUDA:

        $ pip install kaldifeat  # note: ensure it's compiled with CUDA

        $ lhotse feat write-default-config -f kaldifeat-fbank feat.yml

        $ sed 's/device: cpu/device: cuda/' feat.yml feat-cuda.yml

        $ lhotse feat extract-cuts-batch -f feat-cuda.yml cuts.jsonl cuts_with_feats.jsonl feats.h5
    """
    from lhotse import CutSet

    cuts: CutSet = CutSet.from_file(cutset)
    feature_extractor = (
        FeatureExtractor.from_yaml(feature_manifest)
        if feature_manifest is not None
        else Fbank()
    )
    cuts = cuts.compute_and_store_features_batch(
        extractor=feature_extractor,
        storage_path=storage_path,
        batch_duration=batch_duration,
        num_workers=num_jobs,
        storage_type=get_writer(storage_type),
    )
    Path(output_cutset).parent.mkdir(parents=True, exist_ok=True)
    cuts.to_file(output_cutset)


@feat.command(context_settings=dict(show_default=True))
@click.argument("feature_manifest", type=click.Path(exists=True, dir_okay=False))
@click.argument("url")
@click.argument("output_manifest", type=click.Path())
@click.option("--num-jobs", "-j", type=int, default=1)
def upload(
    feature_manifest: Pathlike, url: str, output_manifest: Pathlike, num_jobs: int
):
    """
    Read an existing FEATURE_MANIFEST, upload the feature matrices it contains to a URL location,
    and save a new feature OUTPUT_MANIFEST that refers to the uploaded features.

    The URL can refer to endpoints such as AWS S3, GCP, Azure, etc.
    For example: "s3://my-bucket/my-features" is a valid URL.

    This script does not currently support credentials,
    and assumes that you have the write permissions.
    """
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    output_manifest = Path(output_manifest)
    assert (
        ".jsonl" in output_manifest.suffixes
    ), "This mode only supports writing to JSONL feature manifests."

    local_features: FeatureSet = FeatureSet.from_file(feature_manifest)

    with FeatureSet.open_writer(
        output_manifest
    ) as manifest_writer, ProcessPoolExecutor(num_jobs) as ex:
        futures = []
        for item in tqdm(local_features, desc="Submitting parallel uploading tasks..."):
            futures.append(ex.submit(_upload_one, item, url))
        for item in tqdm(futures, desc=f"Uploading features to {url}"):
            manifest_writer.write(item.result())


def _upload_one(item: Features, url: str) -> Features:
    feats_mtx = item.load()
    feats_writer = LilcomURLWriter(url)
    new_key = feats_writer.write(key=item.storage_key, value=feats_mtx)
    return fastcopy(
        item, storage_path=url, storage_key=new_key, storage_type=feats_writer.name
    )
