import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Optional

import click
import tqdm

from lhotse import CutSet, Fbank, FeatureExtractor
from lhotse.bin.modes.cli_base import cli
from lhotse.features.io import MemoryRawWriter
from lhotse.shar import ArrayTarWriter
from lhotse.utils import Pathlike


@cli.group()
def shar():
    """Lhotse Shar format for optimized I/O commands"""
    pass


@shar.command(context_settings=dict(show_default=True))
@click.argument("cuts", type=click.Path(exists=True, dir_okay=False))
@click.argument("outdir", type=click.Path())
@click.option(
    "-a",
    "--audio",
    default="none",
    type=click.Choice(["none", "wav", "flac", "mp3", "opus", "original"]),
    help="Format in which to export audio. Original will save in the same format as the original audio (disabled by default, enabling will make a copy of the data)",
)
@click.option(
    "-f",
    "--features",
    default="none",
    type=click.Choice(["none", "lilcom", "numpy"]),
    help="Format in which to export features (disabled by default, enabling will make a copy of the data)",
)
@click.option(
    "-c",
    "--custom",
    multiple=True,
    default=[],
    help="Custom fields to export. Use syntax NAME:FORMAT, e.g.: -c target_recording:flac -c embedding:numpy. Use format options for audio and features depending on the custom fields type, or 'jsonl' for metadata.",
)
@click.option(
    "-s",
    "--shard-size",
    type=int,
    default=1000,
    help="The number of cuts in a single shard.",
)
@click.option(
    "--shuffle/--no-shuffle",
    default=True,
    help="Should we shuffle the cuts before splitting into shards.",
)
@click.option(
    "--fault-tolerant/--fast-fail",
    default=False,
    help="Should we skip over cuts that failed to load data or raise an error.",
)
@click.option("--seed", default=0, type=int, help="Random seed.")
@click.option(
    "-j",
    "--num-jobs",
    default=1,
    type=int,
    help="Number of parallel workers. We recommend to keep this number low on machines "
    "with slow disks as the speed of I/O will likely be the bottleneck.",
)
@click.option("-v", "--verbose", count=True)
def export(
    cuts: str,
    outdir: str,
    audio: str,
    features: str,
    custom: List[str],
    shard_size: int,
    shuffle: bool,
    fault_tolerant: bool,
    seed: int,
    num_jobs: int,
    verbose: bool,
):
    """
    Export CutSet from CUTS into Lhotse Shar format in OUTDIR.

    This script partitions the input manifest into smaller pieces called shards
    with SHARD_SIZE cuts per shard. The input is optionally shuffled.
    In addition to sharding, the user can choose to export AUDIO or FEATURES
    into sequentially readable tar files with a selected compression type.
    This typically yields very high speedups vs random read formats such as HDF5,
    especially on slower disks or clusters, at the expense of a data copy.

    The result is readable in Python using: CutSet.from_shar(OUTDIR)
    """
    cuts: CutSet = CutSet.from_file(cuts)

    if shuffle:
        cuts = cuts.shuffle(rng=random.Random(seed))

    fields = {}
    if audio != "none":
        fields["recording"] = audio
    if features != "none":
        fields["features"] = features
    if custom:
        for item in custom:
            key, fmt = item.split(":")
            fields[key] = fmt

    Path(outdir).mkdir(parents=True, exist_ok=True)
    cuts.to_shar(
        output_dir=outdir,
        fields=fields,
        shard_size=shard_size,
        num_jobs=num_jobs,
        fault_tolerant=fault_tolerant,
        verbose=verbose,
    )


@shar.command(context_settings=dict(show_default=True))
@click.argument("shar_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-f",
    "--feature-config",
    type=click.Path(exists=True, dir_okay=False),
    help="Optional manifest specifying feature extractor configuration (use Fbank by default).",
)
@click.option(
    "-c",
    "--compression",
    type=click.Choice(["lilcom", "numpy"]),
    default="lilcom",
    help="Which compression to use (lilcom is lossy, numpy is lossless).",
)
@click.option(
    "-j", "--num-jobs", default=1, type=int, help="Number of parallel workers."
)
@click.option("-v", "--verbose", count=True)
def compute_features(
    shar_dir: str,
    feature_config: Optional[str],
    compression: str,
    num_jobs: int,
    verbose: int,
):
    """
    Compute features for Lhotse Shar cuts stored in SHAR_DIR.

    The features are computed sequentially on CPU within shards,
    and parallelized across shards up to NUM_JOBS concurrent workers.

    FEATURE_CONFIG defines the feature extractor type and settings.
    You can generate default feature extractor settings with:
    lhotse feat write-default-config --help
    """
    shards = [
        {
            "cuts": [p],
            "recording": [p.with_name("".join(["recording", p.suffixes[0], ".tar"]))],
        }
        for p in Path(shar_dir).glob("cuts.*.jsonl*")
    ]
    progbar = lambda x: x
    if verbose:
        click.echo(f"Computing features for {len(shards)} shards.")
        progbar = partial(tqdm.tqdm, desc="Shard progress", total=len(shards))

    futures = []
    with ProcessPoolExecutor(num_jobs) as ex:
        for shard in shards:
            cuts_path = shard["cuts"][0]
            shard_idx = cuts_path.name.split(".")[1]
            output_path = cuts_path.with_name(f"features.{shard_idx}.tar")
            futures.append(
                ex.submit(
                    compute_features_one_shard,
                    cuts=CutSet.from_shar(shard),
                    feature_config=feature_config,
                    output_path=output_path,
                    compression=compression,
                )
            )
        for f in progbar(as_completed(futures)):
            f.result()


def compute_features_one_shard(
    cuts: CutSet, feature_config: Pathlike, output_path: Pathlike, compression: str
):
    extractor = (
        FeatureExtractor.from_yaml(feature_config)
        if feature_config is not None
        else Fbank()
    )
    in_memory = MemoryRawWriter()
    with ArrayTarWriter(
        output_path, shard_size=None, compression=compression
    ) as writer:
        for cut in cuts:
            cut = cut.compute_and_store_features(extractor, in_memory)
            writer.write(key=cut.id, value=cut.load_features(), manifest=cut.features)
