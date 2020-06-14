from pathlib import Path

import click

from lhotse.bin.modes.cli_base import cli
from lhotse.manipulation import (
    load_manifest,
    split as split_manifest,
    combine as combine_manifests
)
from lhotse.utils import Pathlike

__all__ = ['split', 'combine']


@cli.command()
@click.argument('num_splits', type=int)
@click.argument('manifest', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_dir', type=click.Path())
def split(num_splits: int, manifest: Pathlike, output_dir: Pathlike):
    """Load MANIFEST, split it into NUM_SPLITS equal parts and save as separate manifests in OUTPUT_DIR. """
    output_dir = Path(output_dir)
    manifest = Path(manifest)
    data_set = load_manifest(manifest)
    parts = split_manifest(manifest=data_set, num_splits=num_splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, part in enumerate(parts):
        part.to_yaml(output_dir / f'{manifest.stem}.{idx + 1}.yml')


@cli.command()
@click.argument('manifests', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.argument('output_manifest', type=click.Path())
def combine(manifests: Pathlike, output_manifest: Pathlike):
    """Load MANIFESTS, combine them into a single one, and write it to OUTPUT_MANIFEST."""
    data_set = combine_manifests(*[load_manifest(m) for m in manifests])
    data_set.to_yaml(output_manifest)
