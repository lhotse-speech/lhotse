import operator
import re
from math import isclose
from pathlib import Path

import click
from cytoolz.functoolz import complement

from lhotse.bin.modes.cli_base import cli
from lhotse.manipulation import (
    combine as combine_manifests,
    load_manifest,
    split as split_manifest,
    to_manifest,
)
from lhotse.utils import Pathlike

__all__ = ['split', 'combine']


@cli.group()
def manifest():
    """Generic commands working with all or most manifest types."""
    pass


@manifest.command()
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
        part.to_json(output_dir / f'{manifest.stem}.{idx + 1}.json')


@manifest.command()
@click.argument('manifests', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.argument('output_manifest', type=click.Path())
def combine(manifests: Pathlike, output_manifest: Pathlike):
    """Load MANIFESTS, combine them into a single one, and write it to OUTPUT_MANIFEST."""
    data_set = combine_manifests(*[load_manifest(m) for m in manifests])
    data_set.to_json(output_manifest)


@manifest.command()
@click.argument('predicate')
@click.argument('manifest', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_manifest', type=click.Path())
def filter(predicate: str, manifest: Pathlike, output_manifest: Pathlike):
    """
    Filter a MANIFEST according to the rule specified in PREDICATE, and save the result to OUTPUT_MANIFEST.
    It is intended to work generically with most manifest types - it supports RecordingSet, SupervisionSet and CutSet.

    \b
    The PREDICATE specifies which attribute is used for item selection. Some examples:
    lhotse manifest filter 'duration>4.5' supervision.json output.json
    lhotse manifest filter 'num_frames<600' cuts.json output.json
    lhotse manifest filter 'start=0' cuts.json output.json
    lhotse manifest filter 'channel!=0' audio.json output.json

    It currently only supports comparison of numerical manifest item attributes, such as:
    start, duration, end, channel, num_frames, num_features, etc.
    """
    data_set = load_manifest(manifest)

    predicate_pattern = re.compile(r'(?P<key>\w+)(?P<op>=|==|!=|>|<|>=|<=)(?P<value>[0-9.]+)')
    match = predicate_pattern.match(predicate)
    if match is None:
        raise ValueError("Invalid predicate! Run with --help option to learn what predicates are allowed.")

    compare = {
        '<': operator.lt,
        '>': operator.gt,
        '>=': operator.ge,
        '<=': operator.le,
        '=': isclose,
        '==': isclose,
        '!=': complement(isclose)

    }[match.group('op')]
    try:
        value = int(match.group('value'))
    except ValueError:
        value = float(match.group('value'))

    retained_items = []
    try:
        for item in data_set:
            attr = getattr(item, match.group('key'))
            if compare(attr, value):
                retained_items.append(item)
    except AttributeError:
        click.echo(f'Invalid predicate! Items in "{manifest}" do not have the attribute "{match.group("key")}"',
                   err=True)
        exit(1)

    filtered_data_set = to_manifest(retained_items)
    if filtered_data_set is None:
        click.echo('No items satisfying the predicate.', err=True)
        exit(0)
    filtered_data_set.to_json(output_manifest)
