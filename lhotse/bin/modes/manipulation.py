import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
from pathlib import Path
from typing import Optional

import click

from lhotse import (
    FeatureSet,
    available_storage_backends,
)
from lhotse.bin.modes.cli_base import cli
from lhotse.cut import CutSet
from lhotse.features.io import get_writer
from lhotse.utils import Pathlike

__all__ = ["split", "combine", "subset", "filter"]


@cli.command()
@click.argument("input_manifest", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_manifest", type=click.Path())
def copy(input_manifest, output_manifest):
    """
    Load INPUT_MANIFEST and store it to OUTPUT_MANIFEST.
    Useful for conversion between different serialization formats (e.g. JSON, JSONL, YAML).
    Automatically supports gzip compression when '.gz' suffix is detected.
    """
    from lhotse import load_manifest

    data = load_manifest(input_manifest)
    data.to_file(output_manifest)


@cli.command()
@click.argument("input_manifest", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_manifest", type=click.Path())
@click.argument("storage_path", type=str)
@click.option(
    "-t",
    "--storage-type",
    type=click.Choice(available_storage_backends()),
    default="lilcom_chunky",
    help="Which storage backend should we use for writing the copied features.",
)
@click.option(
    "-j",
    "--max-jobs",
    default=-1,
    type=int,
    help="Maximum number of parallel copying processes. "
    "By default, one process is spawned for every existing feature file in the "
    "INPUT_MANIFEST (e.g., if the features were extracted with 20 jobs, "
    "there will typically be 20 files).",
)
def copy_feats(
    input_manifest: Pathlike,
    output_manifest: Pathlike,
    storage_path: str,
    storage_type: str,
    max_jobs: int,
) -> None:
    """
    Load INPUT_MANIFEST of type :class:`lhotse.FeatureSet` or `lhotse.CutSet`,
    read every feature matrix using ``features.load()`` or ``cut.load_features()``,
    save them in STORAGE_PATH and save the updated manifest to OUTPUT_MANIFEST.
    """
    from lhotse.serialization import load_manifest_lazy_or_eager
    from lhotse.manipulation import combine as combine_manifests

    manifests = load_manifest_lazy_or_eager(input_manifest)

    if isinstance(manifests, FeatureSet):
        with get_writer(storage_type)(storage_path) as w:
            # FeatureSet is copied in-memory and written (TODO: make it incremental if needed)
            manifests = manifests.copy_feats(writer=w)
            manifests.to_file(output_manifest)

    elif isinstance(manifests, CutSet):
        # Group cuts by their underlying feature files.
        manifests = sorted(manifests, key=lambda cut: cut.features.storage_path)
        subsets = groupby(manifests, lambda cut: cut.features.storage_path)
        unique_storage_paths, subsets = zip(
            *[(k, CutSet.from_cuts(grp)) for k, grp in subsets]
        )

        # Create paths for new feature files and subset cutsets.
        tot_items = len(unique_storage_paths)
        new_storage_paths = [f"{storage_path}/feats-{i}" for i in range(tot_items)]
        partial_manifest_paths = [
            f"{storage_path}/cuts-{i}.jsonl.gz" for i in range(tot_items)
        ]

        num_jobs = len(unique_storage_paths)
        if max_jobs > 0:
            num_jobs = min(num_jobs, max_jobs)

        # Create directory if needed (storage_path might be an URL)
        if Path(storage_path).parent.is_dir():
            Path(storage_path).mkdir(exist_ok=True)

        # Copy each partition in parallel and combine lazily opened manifests.
        with ProcessPoolExecutor(num_jobs) as ex:
            futures = []
            for cs, nsp, pmp in zip(subsets, new_storage_paths, partial_manifest_paths):
                futures.append(ex.submit(copy_feats_worker, cs, nsp, storage_type, pmp))

            all_cuts = combine_manifests((f.result() for f in as_completed(futures)))

        # Combine and save subset cutsets into the final file.
        with CutSet.open_writer(output_manifest) as w:
            for c in all_cuts:
                w.write(c)
    else:
        raise ValueError(
            f"Unsupported manifest type ({type(manifests)}) at: {input_manifest}"
        )


def copy_feats_worker(
    cuts: CutSet, storage_path: Pathlike, storage_type: str, output_manifest: Path
) -> CutSet:
    with get_writer(storage_type)(storage_path) as w:
        # CutSet has an incremental reading API
        return cuts.copy_feats(writer=w, output_path=output_manifest)


@cli.command()
@click.argument("num_splits", type=int)
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-s",
    "--shuffle",
    is_flag=True,
    help="Optionally shuffle the sequence before splitting.",
)
def split(num_splits: int, manifest: Pathlike, output_dir: Pathlike, shuffle: bool):
    """
    Load MANIFEST, split it into NUM_SPLITS equal parts and save as separate manifests in OUTPUT_DIR.

    When your manifests are very large, prefer to use "lhotse split-lazy" instead.
    """
    from lhotse.serialization import load_manifest_lazy_or_eager

    output_dir = Path(output_dir)
    manifest = Path(manifest)
    suffix = "".join(manifest.suffixes)
    any_set = load_manifest_lazy_or_eager(manifest)
    parts = any_set.split(num_splits=num_splits, shuffle=shuffle)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_digits = len(str(num_splits))
    for idx, part in enumerate(parts):
        idx = f"{idx + 1}".zfill(num_digits)
        part.to_file((output_dir / manifest.stem).with_suffix(f".{idx}{suffix}"))


@cli.command()
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path())
@click.argument("chunk_size", type=int)
def split_lazy(manifest: Pathlike, output_dir: Pathlike, chunk_size: int):
    """
    Load MANIFEST (lazily if in JSONL format) and split it into parts,
    each with CHUNK_SIZE items.
    The parts are saved to separate files with pattern "{output_dir}/{chunk_idx}.jsonl.gz".

    Prefer this to "lhotse split" when your manifests are very large.
    """
    from lhotse.serialization import load_manifest_lazy_or_eager

    output_dir = Path(output_dir)
    manifest = Path(manifest)
    any_set = load_manifest_lazy_or_eager(manifest)
    any_set.split_lazy(output_dir=output_dir, chunk_size=chunk_size)


@cli.command()
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_manifest", type=click.Path())
@click.option("--first", type=int)
@click.option("--last", type=int)
@click.option(
    "--cutids",
    type=str,
    help=(
        "A json string or path to json file containing array of cutids strings. "
        'E.g. --cutids \'["cutid1", "cutid2"]\'.'
    ),
)
def subset(
    manifest: Pathlike,
    output_manifest: Pathlike,
    first: Optional[int],
    last: Optional[int],
    cutids: Optional[str],
):
    """Load MANIFEST, select the FIRST or LAST number of items and store it in OUTPUT_MANIFEST."""
    from lhotse import load_manifest

    output_manifest = Path(output_manifest)
    manifest = Path(manifest)
    any_set = load_manifest(manifest)

    cids = None
    if cutids is not None:
        if os.path.exists(cutids):
            with open(cutids, "rt") as r:
                cids = json.load(r)
        else:
            cids = json.loads(cutids)

    if isinstance(any_set, CutSet):
        a_subset = any_set.subset(first=first, last=last, cut_ids=cids)
    else:
        if cutids is not None:
            raise ValueError(
                f"Expected a CutSet manifest with cut_ids argument; got {type(any_set)}"
            )
        a_subset = any_set.subset(first=first, last=last)

    a_subset.to_file(output_manifest)


@cli.command()
@click.argument("manifests", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.argument("output_manifest", type=click.Path())
def combine(manifests: Pathlike, output_manifest: Pathlike):
    """Load MANIFESTS, combine them into a single one, and write it to OUTPUT_MANIFEST."""
    from lhotse.serialization import load_manifest_lazy_or_eager
    from lhotse.manipulation import combine as combine_manifests

    data_set = combine_manifests(*[load_manifest_lazy_or_eager(m) for m in manifests])
    data_set.to_file(output_manifest)


@cli.command()
@click.argument("predicate")
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_manifest", type=click.Path())
def filter(predicate: str, manifest: Pathlike, output_manifest: Pathlike):
    """
    Filter a MANIFEST according to the rule specified in PREDICATE, and save the result to OUTPUT_MANIFEST.
    It is intended to work generically with most manifest types - it supports RecordingSet, SupervisionSet and CutSet.

    \b
    The PREDICATE specifies which attribute is used for item selection. Some examples:
    lhotse filter 'duration>4.5' supervision.json output.json
    lhotse filter 'num_frames<600' cuts.json output.json
    lhotse filter 'start=0' cuts.json output.json
    lhotse filter 'channel!=0' audio.json output.json

    It currently only supports comparison of numerical manifest item attributes, such as:
    start, duration, end, channel, num_frames, num_features, etc.
    """
    import operator
    import re
    from math import isclose
    from cytoolz.functoolz import complement
    from lhotse import load_manifest
    from lhotse.manipulation import to_manifest

    data_set = load_manifest(manifest)

    predicate_pattern = re.compile(
        r"(?P<key>\w+)(?P<op>=|==|!=|>|<|>=|<=)(?P<value>[0-9.]+)"
    )
    match = predicate_pattern.match(predicate)
    if match is None:
        raise ValueError(
            "Invalid predicate! Run with --help option to learn what predicates are allowed."
        )

    compare = {
        "<": operator.lt,
        ">": operator.gt,
        ">=": operator.ge,
        "<=": operator.le,
        "=": isclose,
        "==": isclose,
        "!=": complement(isclose),
    }[match.group("op")]
    try:
        value = int(match.group("value"))
    except ValueError:
        value = float(match.group("value"))

    retained_items = []
    try:
        for item in data_set:
            attr = getattr(item, match.group("key"))
            if compare(attr, value):
                retained_items.append(item)
    except AttributeError:
        click.echo(
            f'Invalid predicate! Items in "{manifest}" do not have the attribute "{match.group("key")}"',
            err=True,
        )
        exit(1)

    filtered_data_set = to_manifest(retained_items)
    if filtered_data_set is None:
        click.echo("No items satisfying the predicate.", err=True)
        exit(0)
    filtered_data_set.to_file(output_manifest)
